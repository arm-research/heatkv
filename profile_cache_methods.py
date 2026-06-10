#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F


def _parse_batch_sizes(value: str) -> list[int]:
    out = []
    for item in value.split(','):
        item = item.strip()
        if item:
            out.append(int(item))
    if not out:
        raise argparse.ArgumentTypeError('expected at least one batch size')
    if any(batch <= 0 for batch in out):
        raise argparse.ArgumentTypeError('batch sizes must be positive')
    return out


def _normalize_method(method: str) -> str:
    aliases = {
        'fused-segmented': 'real_scale_fused_segmented',
        'fused_segmented': 'real_scale_fused_segmented',
    }
    return aliases.get(method, method)


def _check_gpu_busy(*, fail_on_busy: bool) -> dict:
    try:
        proc = subprocess.run(
            [
                'nvidia-smi',
                '--query-compute-apps=pid,process_name,used_memory',
                '--format=csv,noheader,nounits',
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return {'available': False, 'processes': []}
    rows = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(',')]
        if len(parts) >= 3:
            rows.append({'pid': parts[0], 'name': parts[1], 'used_memory_mib': parts[2]})
    if fail_on_busy and rows:
        raise SystemExit(f'GPU busy; refusing to benchmark: {rows}')
    return {'available': True, 'processes': rows}


def _build_model_args(seed: int):
    return argparse.Namespace(
        pn='1M',
        model_path='weights/infinity_2b_reg.pth',
        cfg_insertion_layer=0,
        vae_type=32,
        vae_path='weights/infinity_vae_d32reg.pth',
        add_lvl_embeding_only_first_block=1,
        use_bit_label=1,
        model_type='infinity_2b',
        rope2d_each_sa_layer=1,
        rope2d_normalized_by_hw=2,
        use_scale_schedule_embedding=0,
        sampling_per_bits=1,
        text_encoder_ckpt='weights/flan-t5-xl',
        text_channels=2048,
        apply_spatial_patchify=0,
        h_div_w_template=1.000,
        use_flex_attn=0,
        cache_dir='/dev/shm',
        checkpoint_type='torch',
        seed=seed,
        bf16=1,
        enable_model_cache=1,
    )


def _scale_schedule(run_infinity_module, model_args):
    h_div_w = 1.0
    template = run_infinity_module.h_div_w_templates[
        np.argmin(np.abs(run_infinity_module.h_div_w_templates - h_div_w))
    ]
    schedule = run_infinity_module.dynamic_resolution_h_w[template][model_args.pn]['scales']
    return [(1, h, w) for (_, h, w) in schedule]


def _encode_prompts(text_tokenizer, text_encoder, prompt: str, batch_size: int):
    captions = [prompt] * int(batch_size)
    tokens = text_tokenizer(
        text=captions,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
    )
    input_ids = tokens.input_ids.cuda(non_blocking=True)
    mask = tokens.attention_mask.cuda(non_blocking=True)
    text_features = text_encoder(input_ids=input_ids, attention_mask=mask)['last_hidden_state'].float()
    lens = mask.sum(dim=-1).tolist()
    cu_seqlens_k = F.pad(mask.sum(dim=-1).to(dtype=torch.int32).cumsum_(0), (1, 0))
    max_seqlen_k = max(lens)
    kv_compact = torch.cat([feat[:length] for feat, length in zip(text_features.unbind(0), lens)], dim=0)
    return kv_compact, lens, cu_seqlens_k, max_seqlen_k


def _apply_method(model, method: str, *, budget: float, initial_scales: int, batch_size: int):
    method = _normalize_method(method)
    if method == 'baseline':
        return model
    if method == 'real_scale':
        from enable_real_pruning import enable_real_pruning_scale_specific
        from tools.utils import orders
        return enable_real_pruning_scale_specific(
            model,
            budget,
            orders(initial_scales),
            protected_prefix_scales=initial_scales,
            source_scale_start=initial_scales,
        )
    if method == 'real_scale_fused_segmented':
        from enable_real_pruning_fast import enable_real_pruning_scale_specific_fused_segmented
        from tools.utils import orders
        return enable_real_pruning_scale_specific_fused_segmented(
            model,
            budget,
            orders(initial_scales),
            protected_prefix_scales=initial_scales,
            source_scale_start=initial_scales,
            requested_generation_batch_size=batch_size,
        )
    raise ValueError(f'unsupported method: {method}')


def _run_once(model, vae, text_cond_tuple, *, batch_size: int, scale_schedule, seed: int, decode: bool):
    cfg = 3
    tau = 1.0
    cfg_list = [cfg] * len(scale_schedule)
    tau_list = [tau] * len(scale_schedule)
    with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16, cache_enabled=True):
        return model.autoregressive_infer_cfg(
            vae=vae,
            scale_schedule=scale_schedule,
            label_B_or_BLT=text_cond_tuple,
            g_seed=seed,
            B=batch_size,
            negative_label_B_or_BLT=None,
            force_gt_Bhw=None,
            cfg_sc=cfg,
            cfg_list=cfg_list,
            tau_list=tau_list,
            top_k=900,
            top_p=0.97,
            returns_vemb=1,
            ratio_Bl1=None,
            gumbel=0,
            norm_cfg=False,
            cfg_exp_k=0.0,
            cfg_insertion_layer=[0],
            vae_type=32,
            softmax_merge_topk=-1,
            ret_img=decode,
            trunk_scale=1000,
            gt_leak=0,
            gt_ls_Bl=None,
            inference_mode=True,
            sampling_per_bits=1,
        )


def _profile_one(method: str, batch_size: int, args, components) -> dict:
    run_infinity, text_tokenizer, text_encoder, vae = components
    model_args = _build_model_args(args.seed)
    scale_schedule = _scale_schedule(run_infinity, model_args)
    text_cond_tuple = _encode_prompts(text_tokenizer, text_encoder, args.prompt, batch_size)
    model = run_infinity.load_transformer(vae, model_args)
    model = _apply_method(
        model,
        method,
        budget=args.head_specific_budget,
        initial_scales=args.head_specific_initial_scales,
        batch_size=batch_size,
    )
    model.eval()

    fast_profile = None
    if _normalize_method(method) == 'real_scale_fused_segmented' and not args.disable_phase_profile:
        import enable_real_pruning_fast as fast_profile
        fast_profile.set_real_pruning_fast_phase_profile_enabled(True)

    timings = []
    phases = []
    peak_allocated = []
    peak_reserved = []
    total_runs = args.warmup + args.repeats
    for run_idx in range(total_runs):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        if fast_profile is not None:
            fast_profile.reset_real_pruning_fast_phase_profile()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _run_once(
            model,
            vae,
            text_cond_tuple,
            batch_size=batch_size,
            scale_schedule=scale_schedule,
            seed=args.seed,
            decode=args.decode,
        )
        end.record()
        end.synchronize()
        elapsed_ms = float(start.elapsed_time(end))
        if run_idx >= args.warmup:
            timings.append(elapsed_ms)
            peak_allocated.append(int(torch.cuda.max_memory_allocated()))
            peak_reserved.append(int(torch.cuda.max_memory_reserved()))
            if fast_profile is not None:
                phases.append(fast_profile.snapshot_real_pruning_fast_phase_profile(reset=True))

    if fast_profile is not None:
        fast_profile.set_real_pruning_fast_phase_profile_enabled(False)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        'method': _normalize_method(method),
        'batch_size': int(batch_size),
        'decode': bool(args.decode),
        'warmup': int(args.warmup),
        'repeats': int(args.repeats),
        'latency_ms': timings,
        'median_ms': statistics.median(timings),
        'mean_ms': statistics.fmean(timings),
        'per_image_median_ms': statistics.median(timings) / float(batch_size),
        'peak_allocated_bytes': max(peak_allocated) if peak_allocated else 0,
        'peak_reserved_bytes': max(peak_reserved) if peak_reserved else 0,
        'phase_profiles': phases,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', nargs='+', default=['baseline', 'real_scale_fused_segmented'])
    parser.add_argument('--batch-sizes', type=_parse_batch_sizes, default=[1])
    parser.add_argument('--prompt', default='A photo of a red sports car on a mountain road')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--decode', action='store_true', help='Include VAE image decode in the timed region.')
    parser.add_argument('--disable-phase-profile', action='store_true')
    parser.add_argument('--fail-on-busy-gpu', action='store_true')
    parser.add_argument('--head-specific-budget', type=float, default=0.1)
    parser.add_argument('--head-specific-initial-scales', type=int, default=3)
    parser.add_argument('--json-out', type=str, default=None)
    args = parser.parse_args(argv)

    if args.warmup < 0 or args.repeats <= 0:
        parser.error('--warmup must be >= 0 and --repeats must be > 0')

    busy = _check_gpu_busy(fail_on_busy=args.fail_on_busy_gpu)
    torch.cuda.set_device(0)

    from tools import run_infinity

    model_args = _build_model_args(args.seed)
    text_tokenizer, text_encoder = run_infinity.load_tokenizer(t5_path=model_args.text_encoder_ckpt)
    vae = run_infinity.load_visual_tokenizer(model_args)
    components = (run_infinity, text_tokenizer, text_encoder, vae)

    results = []
    for batch_size in args.batch_sizes:
        for method in args.methods:
            result = _profile_one(method, batch_size, args, components)
            results.append(result)
            print(
                f"{result['method']} b{batch_size}: "
                f"median={result['median_ms']:.1f} ms "
                f"per_image={result['per_image_median_ms']:.1f} ms "
                f"peak_alloc={result['peak_allocated_bytes'] / 2**30:.2f} GiB "
                f"peak_reserved={result['peak_reserved_bytes'] / 2**30:.2f} GiB"
            )

    payload = {
        'config': {
            'methods': [_normalize_method(m) for m in args.methods],
            'batch_sizes': args.batch_sizes,
            'prompt': args.prompt,
            'seed': args.seed,
            'decode': args.decode,
            'warmup': args.warmup,
            'repeats': args.repeats,
            'head_specific_budget': args.head_specific_budget,
            'head_specific_initial_scales': args.head_specific_initial_scales,
        },
        'gpu_busy_preflight': busy,
        'results': results,
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
