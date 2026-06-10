#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch

from enable_real_pruning_fast import enable_real_pruning_scale_specific_fused_segmented
from tools import run_infinity
from tools.utils import orders


def _build_model_args(seed: int) -> argparse.Namespace:
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


def _scale_schedule(args: argparse.Namespace) -> list[tuple[int, int, int]]:
    schedule = run_infinity.dynamic_resolution_h_w[args.h_div_w_template][args.pn]["scales"]
    return [(1, h, w) for (_, h, w) in schedule]


def _generate(model, vae, text_tokenizer, text_encoder, prompt: str, seed: int, args, scale_schedule):
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        return run_infinity.gen_one_img(
            model,
            vae,
            text_tokenizer,
            text_encoder,
            prompt,
            g_seed=seed,
            gt_leak=0,
            gt_ls_Bl=None,
            cfg_list=3,
            tau_list=1.0,
            scale_schedule=scale_schedule,
            cfg_insertion_layer=[args.cfg_insertion_layer],
            vae_type=args.vae_type,
            sampling_per_bits=args.sampling_per_bits,
            enable_positive_prompt=0,
        )


def _clear_model(model) -> None:
    del model
    gc.collect()
    torch.cuda.empty_cache()


def _save_image(path: Path, image: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image.cpu().numpy())


def _save_side_by_side(path: Path, baseline: torch.Tensor, heatkv: torch.Tensor) -> None:
    left = baseline.cpu().numpy()
    right = heatkv.cpu().numpy()
    cv2.imwrite(str(path), np.concatenate([left, right], axis=1))


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate 2B Infinity images with and without HeatKV.")
    parser.add_argument("--prompt", default="A photo of a red sports car on a mountain road")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/heatkv_2b_comparison")
    parser.add_argument("--heatkv-budget", type=float, default=0.1)
    parser.add_argument("--heatkv-initial-scales", type=int, default=3)
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    model_args = _build_model_args(args.seed)
    scale_schedule = _scale_schedule(model_args)

    text_tokenizer, text_encoder = run_infinity.load_tokenizer(t5_path=model_args.text_encoder_ckpt)
    vae = run_infinity.load_visual_tokenizer(model_args)

    output_dir = Path(args.output_dir)
    baseline_path = output_dir / "2b_without_heatkv.png"
    heatkv_path = output_dir / "2b_with_heatkv.png"
    comparison_path = output_dir / "2b_comparison.png"

    model = run_infinity.load_transformer(vae, model_args)
    baseline = _generate(model, vae, text_tokenizer, text_encoder, args.prompt, args.seed, model_args, scale_schedule)
    _save_image(baseline_path, baseline)
    _clear_model(model)

    model = run_infinity.load_transformer(vae, model_args)
    model = enable_real_pruning_scale_specific_fused_segmented(
        model,
        args.heatkv_budget,
        orders(args.heatkv_initial_scales, model_type=2),
        protected_prefix_scales=args.heatkv_initial_scales,
        source_scale_start=args.heatkv_initial_scales,
        requested_generation_batch_size=1,
    )
    heatkv = _generate(model, vae, text_tokenizer, text_encoder, args.prompt, args.seed, model_args, scale_schedule)
    _save_image(heatkv_path, heatkv)
    _save_side_by_side(comparison_path, baseline, heatkv)
    _clear_model(model)

    print(f"Saved baseline: {baseline_path.resolve()}")
    print(f"Saved HeatKV: {heatkv_path.resolve()}")
    print(f"Saved comparison: {comparison_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
