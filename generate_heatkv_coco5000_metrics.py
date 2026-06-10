#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from enable_real_pruning_fast import enable_real_pruning_scale_specific_fused_segmented
from tools import run_infinity
from tools.utils import get_dataset, get_prompt, orders


def _build_model_args(model: str, seed: int) -> argparse.Namespace:
    if model == "2b":
        return argparse.Namespace(
            pn="1M",
            model_path="weights/infinity_2b_reg.pth",
            cfg_insertion_layer=0,
            vae_type=32,
            vae_path="weights/infinity_vae_d32reg.pth",
            add_lvl_embeding_only_first_block=1,
            use_bit_label=1,
            model_type="infinity_2b",
            rope2d_each_sa_layer=1,
            rope2d_normalized_by_hw=2,
            use_scale_schedule_embedding=0,
            sampling_per_bits=1,
            text_encoder_ckpt="weights/flan-t5-xl",
            text_channels=2048,
            apply_spatial_patchify=0,
            h_div_w_template=1.000,
            use_flex_attn=0,
            cache_dir="/dev/shm",
            checkpoint_type="torch",
            seed=seed,
            bf16=1,
            enable_model_cache=1,
        )
    if model == "8b":
        return argparse.Namespace(
            pn="1M",
            model_path="weights/infinity_8b_weights",
            cfg_insertion_layer=0,
            vae_type=14,
            vae_path="weights/infinity_vae_d56_f8_14_patchify.pth",
            add_lvl_embeding_only_first_block=1,
            use_bit_label=1,
            model_type="infinity_8b",
            rope2d_each_sa_layer=1,
            rope2d_normalized_by_hw=2,
            use_scale_schedule_embedding=0,
            sampling_per_bits=1,
            text_encoder_ckpt="weights/flan-t5-xl",
            text_channels=2048,
            apply_spatial_patchify=1,
            h_div_w_template=1.000,
            use_flex_attn=0,
            cache_dir="/dev/shm",
            checkpoint_type="torch_shard",
            seed=seed,
            bf16=1,
            enable_model_cache=0,
        )
    raise ValueError(f"unsupported model: {model}")


def _scale_schedule(args: argparse.Namespace) -> list[tuple[int, int, int]]:
    schedule = run_infinity.dynamic_resolution_h_w[args.h_div_w_template][args.pn]["scales"]
    return [(1, h, w) for (_, h, w) in schedule]


def _collect_prompts(dataset_name: str, count: int, seed: int, manifest_path: Path) -> list[dict]:
    if manifest_path.exists():
        prompts = []
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    prompts.append(json.loads(line))
        if len(prompts) >= count:
            return prompts[:count]

    prompts = []
    for entry in get_dataset(dataset_name):
        prompt = get_prompt(entry, dataset_name)
        if not prompt:
            continue
        idx = len(prompts)
        prompts.append({"index": idx, "seed": seed + idx, "prompt": prompt})
        if len(prompts) >= count:
            break

    if len(prompts) < count:
        raise RuntimeError(f"Only found {len(prompts)} prompts in dataset {dataset_name!r}; wanted {count}.")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for item in prompts:
            handle.write(json.dumps(item, ensure_ascii=True) + "\n")
    return prompts


def _load_model(vae, model_args, *, heatkv: bool, heatkv_budget: float, heatkv_initial_scales: int, model_type: int):
    model = run_infinity.load_transformer(vae, model_args)
    if not heatkv:
        return model
    return enable_real_pruning_scale_specific_fused_segmented(
        model,
        heatkv_budget,
        orders(heatkv_initial_scales, model_type=model_type),
        protected_prefix_scales=heatkv_initial_scales,
        source_scale_start=heatkv_initial_scales,
        requested_generation_batch_size=1,
    )


def _generate_folder(
    *,
    label: str,
    output_dir: Path,
    prompts: list[dict],
    model,
    vae,
    text_tokenizer,
    text_encoder,
    model_args,
    scale_schedule,
    overwrite: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    progress = tqdm(prompts, desc=f"Generating {label}", unit="img")
    for item in progress:
        image_path = output_dir / f"{item['index']:06d}.png"
        if image_path.exists() and not overwrite:
            continue
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
            generated_image = run_infinity.gen_one_img(
                model,
                vae,
                text_tokenizer,
                text_encoder,
                item["prompt"],
                g_seed=item["seed"],
                gt_leak=0,
                gt_ls_Bl=None,
                cfg_list=3,
                tau_list=1.0,
                scale_schedule=scale_schedule,
                cfg_insertion_layer=[model_args.cfg_insertion_layer],
                vae_type=model_args.vae_type,
                sampling_per_bits=model_args.sampling_per_bits,
                enable_positive_prompt=0,
            )
        cv2.imwrite(str(image_path), generated_image.cpu().numpy())
        del generated_image
        torch.cuda.empty_cache()


def _clear_model(model) -> None:
    del model
    gc.collect()
    torch.cuda.empty_cache()


class _PairedImageDataset(Dataset):
    def __init__(self, base_dir: Path, heatkv_dir: Path):
        base_files = {path.name: path for path in base_dir.glob("*.png")}
        heatkv_files = {path.name: path for path in heatkv_dir.glob("*.png")}
        names = sorted(base_files.keys() & heatkv_files.keys())
        if not names:
            raise RuntimeError(f"No paired PNG files found in {base_dir} and {heatkv_dir}.")
        self.pairs = [(base_files[name], heatkv_files[name]) for name in names]

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        base_path, heatkv_path = self.pairs[index]
        base = _load_image_tensor(base_path)
        heatkv = _load_image_tensor(heatkv_path)
        return base, heatkv


def _load_image_tensor(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.uint8)
    return torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)


def _compute_pair_metrics(base_dir: Path, heatkv_dir: Path, *, batch_size: int, device: torch.device) -> dict:
    from torchmetrics.image import LearnedPerceptualImagePatchSimilarity, PeakSignalNoiseRatio

    dataset = _PairedImageDataset(base_dir, heatkv_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    psnr = PeakSignalNoiseRatio(
        data_range=(0, 1),
        reduction="elementwise_mean",
        dim=(1, 2, 3),
    ).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(device)

    psnr_total = 0.0
    lpips_total = 0.0
    image_count = 0
    for base, heatkv in tqdm(loader, desc="Computing PSNR/LPIPS", unit="batch"):
        base = base.to(device, non_blocking=True)
        heatkv = heatkv.to(device, non_blocking=True)
        current_batch_size = int(base.shape[0])
        psnr_total += float(psnr(heatkv, base).detach().float().cpu().item()) * current_batch_size
        lpips_total += float(lpips(heatkv, base).detach().float().cpu().item()) * current_batch_size
        image_count += current_batch_size
        psnr.reset()
        lpips.reset()

    return {
        "paired_images": len(dataset),
        "psnr": psnr_total / image_count,
        "lpips": lpips_total / image_count,
    }


def _compute_fid(base_dir: Path, heatkv_dir: Path) -> float:
    from cleanfid import fid

    return float(fid.compute_fid(str(base_dir), str(heatkv_dir)))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate paired base/HeatKV COCO2017 images and compute PSNR, LPIPS, and FID."
    )
    parser.add_argument("--dataset", default="test", choices=("test", "coco2017"))
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--model", choices=("2b", "8b"), default="2b")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-root", default="outputs/coco2017_heatkv_5000")
    parser.add_argument("--base-folder", default="base")
    parser.add_argument("--heatkv-folder", default="heatkv")
    parser.add_argument("--heatkv-budget", type=float, default=0.1)
    parser.add_argument("--heatkv-initial-scales", type=int, default=3)
    parser.add_argument("--metric-batch-size", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-metrics", action="store_true")
    args = parser.parse_args()

    if args.count <= 0:
        parser.error("--count must be positive")
    if args.heatkv_initial_scales < 2:
        parser.error("--heatkv-initial-scales must be at least 2")

    output_root = Path(args.output_root)
    base_dir = output_root / args.base_folder
    heatkv_dir = output_root / args.heatkv_folder
    manifest_path = output_root / "prompts.jsonl"
    metrics_path = output_root / "metrics.json"
    output_root.mkdir(parents=True, exist_ok=True)

    prompts = _collect_prompts(args.dataset, args.count, args.seed, manifest_path)
    model_args = _build_model_args(args.model, args.seed)
    model_type = 8 if args.model == "8b" else 2

    if not args.skip_generation:
        torch.cuda.set_device(args.gpu)
        text_tokenizer, text_encoder = run_infinity.load_tokenizer(t5_path=model_args.text_encoder_ckpt)
        vae = run_infinity.load_visual_tokenizer(model_args)
        scale_schedule = _scale_schedule(model_args)

        model = _load_model(
            vae,
            model_args,
            heatkv=False,
            heatkv_budget=args.heatkv_budget,
            heatkv_initial_scales=args.heatkv_initial_scales,
            model_type=model_type,
        )
        _generate_folder(
            label="base",
            output_dir=base_dir,
            prompts=prompts,
            model=model,
            vae=vae,
            text_tokenizer=text_tokenizer,
            text_encoder=text_encoder,
            model_args=model_args,
            scale_schedule=scale_schedule,
            overwrite=args.overwrite,
        )
        _clear_model(model)

        model = _load_model(
            vae,
            model_args,
            heatkv=True,
            heatkv_budget=args.heatkv_budget,
            heatkv_initial_scales=args.heatkv_initial_scales,
            model_type=model_type,
        )
        _generate_folder(
            label="HeatKV",
            output_dir=heatkv_dir,
            prompts=prompts,
            model=model,
            vae=vae,
            text_tokenizer=text_tokenizer,
            text_encoder=text_encoder,
            model_args=model_args,
            scale_schedule=scale_schedule,
            overwrite=args.overwrite,
        )
        _clear_model(model)

    if args.skip_metrics:
        print(f"Base images: {base_dir.resolve()}")
        print(f"HeatKV images: {heatkv_dir.resolve()}")
        return 0

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    metrics = _compute_pair_metrics(base_dir, heatkv_dir, batch_size=args.metric_batch_size, device=device)
    metrics["fid"] = _compute_fid(base_dir, heatkv_dir)
    metrics["base_dir"] = str(base_dir)
    metrics["heatkv_dir"] = str(heatkv_dir)
    metrics["dataset"] = args.dataset
    metrics["count_requested"] = args.count
    metrics["model"] = args.model

    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Saved metrics: {metrics_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
