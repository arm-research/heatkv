#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

# ROOT = Path(__file__).resolve().parent
# if str(ROOT) not in sys.path:
#     sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from tqdm import tqdm

from infinity.models.basic import SelfAttention, apply_rotary_emb
from tools import run_infinity


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


def _scale_schedule(model_args: argparse.Namespace) -> list[tuple[int, int, int]]:
    schedule = run_infinity.dynamic_resolution_h_w[model_args.h_div_w_template][model_args.pn]["scales"]
    return [(1, h, w) for (_, h, w) in schedule]


def _scale_token_bounds(scale_schedule: list[tuple[int, int, int]]) -> list[tuple[int, int]]:
    bounds = []
    cursor = 0
    for t, h, w in scale_schedule:
        tokens = int(t) * int(h) * int(w)
        bounds.append((cursor, cursor + tokens))
        cursor += tokens
    return bounds


def _read_prompts(path: Path, max_prompts: int | None) -> list[str]:
    prompts = []
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if suffix == ".jsonl":
                item = json.loads(line)
                prompt = item.get("prompt") or item.get("caption")
                if prompt is None and isinstance(item.get("captions"), list) and item["captions"]:
                    prompt = item["captions"][0]
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError(f"{path}:{line_number} does not contain a usable prompt")
                prompts.append(prompt.strip())
            else:
                prompts.append(line)
            if max_prompts is not None and len(prompts) >= max_prompts:
                break
    if not prompts:
        raise ValueError(f"no prompts found in {path}")
    return prompts


class ScasAccumulator:
    def __init__(
        self,
        *,
        num_layers: int,
        num_heads: int,
        scale_schedule: list[tuple[int, int, int]],
        sink_scales: int,
        device: torch.device,
    ):
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.scale_schedule = scale_schedule
        self.sink_scales = int(sink_scales)
        self.num_scales = len(scale_schedule)
        self.bounds = _scale_token_bounds(scale_schedule)
        self.accum = torch.zeros(
            (self.num_layers, self.num_heads, self.num_scales),
            dtype=torch.float64,
            device=device,
        )

    def update(self, *, layer_idx: int, scale_ind: int, probs: torch.Tensor) -> None:
        if scale_ind <= self.sink_scales:
            return
        if scale_ind >= self.num_scales:
            return

        probs = probs[:1].float()
        layer_idx = int(layer_idx)
        source_end = min(scale_ind, self.num_scales - 1)
        for source_scale_idx in range(self.sink_scales, source_end):
            start, end = self.bounds[source_scale_idx]
            if end > probs.shape[-1]:
                continue
            mass_by_head = probs[..., start:end].sum(dim=-1).mean(dim=(0, 2))
            self.accum[layer_idx, :, source_scale_idx] += mass_by_head.to(dtype=torch.float64)

    def scores(self, num_prompts: int) -> torch.Tensor:
        out = torch.full_like(self.accum, float("nan"))
        for source_scale_idx in range(self.sink_scales, self.num_scales - 1):
            future_count = self.num_scales - source_scale_idx - 1
            if future_count <= 0:
                continue
            out[:, :, source_scale_idx] = self.accum[:, :, source_scale_idx] / (
                float(num_prompts) * float(future_count)
            )
        return out

    def orders_from_scores(self, scores: torch.Tensor) -> list[list[tuple[int, int]]]:
        scores_cpu = scores.detach().cpu()
        orders = []
        for source_scale_idx in range(self.sink_scales, self.num_scales - 1):
            entries = []
            for layer_idx in range(self.num_layers):
                for head_idx in range(self.num_heads):
                    score = float(scores_cpu[layer_idx, head_idx, source_scale_idx])
                    entries.append((score, head_idx, layer_idx))
            entries.sort(key=lambda item: (item[0], item[1], item[2]))
            orders.append([(head_idx, layer_idx) for _, head_idx, layer_idx in entries])
        return orders


def _scas_forward(
    self,
    x,
    attn_bias_or_two_vector,
    attn_fn=None,
    scale_schedule=None,
    rope2d_freqs_grid=None,
    scale_ind=0,
):
    if attn_bias_or_two_vector is not None:
        raise ValueError("S-CAS capture supports inference attention only")

    original_forward = self._scas_original_forward

    batch_size, query_len, _ = x.shape
    qkv_bias = torch.cat((self.q_bias, self.zero_k_bias, self.v_bias))
    qkv = F.linear(
        input=x,
        weight=self.mat_qkv.weight,
        bias=qkv_bias,
    ).view(batch_size, query_len, 3, self.num_heads, self.head_dim)

    if self.using_flash:
        q, k, v = qkv.unbind(dim=2)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
    else:
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)

    if self.cos_attn:
        if self.using_flash:
            scale_mul = (
                self.scale_mul_1H11.clamp_max(self.max_scale_mul)
                .exp()
                .permute(0, 2, 1, 3)
            )
        else:
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()
        q = F.normalize(q, dim=-1, eps=1e-12).mul(scale_mul).contiguous()
        k = F.normalize(k, dim=-1, eps=1e-12).contiguous()
        v = v.contiguous()
    else:
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

    if rope2d_freqs_grid is not None:
        q, k = apply_rotary_emb(
            q,
            k,
            scale_schedule,
            rope2d_freqs_grid,
            self.pad_to_multiplier,
            self.rope2d_normalized_by_hw,
            scale_ind,
        )

    if self.caching and self.cached_k is not None:
        k = torch.cat((self.cached_k, k), dim=2)
        v = torch.cat((self.cached_v, v), dim=2)

    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * float(self.scale)
    probs = torch.softmax(scores, dim=-1)
    self._scas_accumulator.update(
        layer_idx=self.layer_idx,
        scale_ind=int(scale_ind),
        probs=probs,
    )
    del scores, probs, q, k, v, qkv

    return original_forward(
        x,
        attn_bias_or_two_vector,
        attn_fn=attn_fn,
        scale_schedule=scale_schedule,
        rope2d_freqs_grid=rope2d_freqs_grid,
        scale_ind=scale_ind,
    )


def _patch_self_attention(model, accumulator: ScasAccumulator) -> list[tuple[SelfAttention, types.MethodType]]:
    modules = [module for module in model.modules() if isinstance(module, SelfAttention)]
    if not modules:
        raise ValueError("model does not contain any SelfAttention modules")
    num_heads = modules[0].num_heads
    if any(module.num_heads != num_heads for module in modules):
        raise ValueError("S-CAS capture requires all SelfAttention modules to use the same num_heads")

    originals = []
    for layer_idx, module in enumerate(modules):
        module.layer_idx = layer_idx
        module._scas_accumulator = accumulator
        module._scas_original_forward = module.forward
        originals.append((module, module.forward))
        module.forward = types.MethodType(_scas_forward, module)
    return originals


def _restore_self_attention(originals: list[tuple[SelfAttention, types.MethodType]]) -> None:
    for module, original_forward in originals:
        module.forward = original_forward
        if hasattr(module, "_scas_accumulator"):
            delattr(module, "_scas_accumulator")
        if hasattr(module, "_scas_original_forward"):
            delattr(module, "_scas_original_forward")


def _model_shape(model) -> tuple[int, int]:
    modules = [module for module in model.modules() if isinstance(module, SelfAttention)]
    if not modules:
        raise ValueError("model does not contain any SelfAttention modules")
    num_heads = modules[0].num_heads
    if any(module.num_heads != num_heads for module in modules):
        raise ValueError("S-CAS capture requires all SelfAttention modules to use the same num_heads")
    return len(modules), int(num_heads)


def _run_prompt(
    *,
    model,
    vae,
    text_tokenizer,
    text_encoder,
    prompt: str,
    seed: int,
    model_args: argparse.Namespace,
    scale_schedule: list[tuple[int, int, int]],
    cfg: float,
    tau: float,
) -> None:
    kv_compact, lens, cu_seqlens_k, max_seqlen_k = run_infinity.encode_prompt(
        text_tokenizer,
        text_encoder,
        prompt,
        enable_positive_prompt=False,
    )
    text_cond_tuple = (kv_compact.to(dtype=torch.bfloat16), lens, cu_seqlens_k, max_seqlen_k)
    cfg_list = [float(cfg)] * len(scale_schedule)
    tau_list = [float(tau)] * len(scale_schedule)
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True):
        model.autoregressive_infer_cfg(
            vae=vae,
            scale_schedule=scale_schedule,
            label_B_or_BLT=text_cond_tuple,
            g_seed=seed,
            B=1,
            negative_label_B_or_BLT=None,
            force_gt_Bhw=None,
            cfg_sc=float(cfg),
            cfg_list=cfg_list,
            tau_list=tau_list,
            top_k=900,
            top_p=0.97,
            returns_vemb=1,
            ratio_Bl1=None,
            gumbel=0,
            norm_cfg=False,
            cfg_exp_k=0.0,
            cfg_insertion_layer=[model_args.cfg_insertion_layer],
            vae_type=model_args.vae_type,
            softmax_merge_topk=-1,
            ret_img=False,
            trunk_scale=1000,
            gt_leak=0,
            gt_ls_Bl=None,
            inference_mode=True,
            sampling_per_bits=model_args.sampling_per_bits,
        )


def _jsonable_orders(orders: list[list[tuple[int, int]]]) -> list[list[list[int]]]:
    return [[[int(head), int(layer)] for head, layer in order] for order in orders]


def _finite_score_summary(values: torch.Tensor) -> dict[str, float]:
    finite_values = values[torch.isfinite(values)]
    if finite_values.numel() == 0:
        return {"min": float("nan"), "mean": float("nan"), "max": float("nan")}
    return {
        "min": float(finite_values.min().item()),
        "mean": float(finite_values.mean().item()),
        "max": float(finite_values.max().item()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute 2B Infinity S-CAS scores on the fly.")
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--sink-scales", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/scas_2b"))
    parser.add_argument("--max-prompts", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--tau", type=float, default=1.0)
    args = parser.parse_args()

    if args.sink_scales < 0:
        parser.error("--sink-scales must be non-negative")
    if args.max_prompts is not None and args.max_prompts <= 0:
        parser.error("--max-prompts must be positive")

    prompts = _read_prompts(args.prompt_file, args.max_prompts)
    model_args = _build_model_args(args.seed)
    scale_schedule = _scale_schedule(model_args)
    if args.sink_scales >= len(scale_schedule) - 1:
        parser.error("--sink-scales must leave at least one eligible source scale")

    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    text_tokenizer, text_encoder = run_infinity.load_tokenizer(t5_path=model_args.text_encoder_ckpt)
    vae = run_infinity.load_visual_tokenizer(model_args)
    model = run_infinity.load_transformer(vae, model_args)
    model.eval()

    num_layers, num_heads = _model_shape(model)
    accumulator = ScasAccumulator(
        num_layers=num_layers,
        num_heads=num_heads,
        scale_schedule=scale_schedule,
        sink_scales=args.sink_scales,
        device=device,
    )

    originals = _patch_self_attention(model, accumulator)
    try:
        with torch.no_grad():
            for idx, prompt in enumerate(tqdm(prompts, desc="Computing S-CAS", unit="prompt")):
                _run_prompt(
                    model=model,
                    vae=vae,
                    text_tokenizer=text_tokenizer,
                    text_encoder=text_encoder,
                    prompt=prompt,
                    seed=args.seed + idx,
                    model_args=model_args,
                    scale_schedule=scale_schedule,
                    cfg=args.cfg,
                    tau=args.tau,
                )
                torch.cuda.empty_cache()
    finally:
        _restore_self_attention(originals)

    scores = accumulator.scores(len(prompts)).detach().cpu()
    scas_orders = accumulator.orders_from_scores(scores)
    eligible_source_scales = list(range(args.sink_scales, len(scale_schedule) - 1))
    metadata = {
        "model": "infinity_2b",
        "num_prompts": len(prompts),
        "seed": args.seed,
        "cfg": args.cfg,
        "tau": args.tau,
        "sink_scales": args.sink_scales,
        "eligible_source_scales": eligible_source_scales,
        "scale_schedule": scale_schedule,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "order_format": "(head, layer), ascending S-CAS; lowest future dependency first",
        "prompt_file": str(args.prompt_file),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "scores": scores,
            "metadata": metadata,
        },
        args.output_dir / "scas_scores.pt",
    )
    torch.save(
        {
            "orders": scas_orders,
            "metadata": metadata,
        },
        args.output_dir / "scas_orders.pt",
    )

    score_summary = {}
    for source_scale_idx in eligible_source_scales:
        values = scores[:, :, source_scale_idx]
        score_summary[str(source_scale_idx)] = _finite_score_summary(values)
    (args.output_dir / "scas_scores.json").write_text(
        json.dumps(
            {
                "metadata": metadata,
                "summary_by_source_scale": score_summary,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (args.output_dir / "scas_orders.json").write_text(
        json.dumps(
            {
                "metadata": metadata,
                "orders": _jsonable_orders(scas_orders),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print(f"Saved S-CAS scores: {(args.output_dir / 'scas_scores.pt').resolve()}")
    print(f"Saved S-CAS orders: {(args.output_dir / 'scas_orders.pt').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
