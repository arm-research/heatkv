# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT
from __future__ import annotations

"""Slim fused-segmented attention kernels for scale-specific real pruning.

This module intentionally contains only the fused-segmented path needed by the
NoK branch:
- K/V segment packing into a flat slab.
- Triton attention over flat old K/V segments plus current K/V.

It does not include page-backed/page-native experiments, CUDA graph helpers, or
other local research-only paths.
"""

import math
import os
from typing import Optional

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - depends on target runtime
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _pack_flat_segments_kv_kernel(
        src_k_ptr,
        src_v_ptr,
        dst_k_ptr,
        dst_v_ptr,
        src_starts_ptr,
        dst_starts_ptr,
        lengths_ptr,
        src_batch_stride: tl.constexpr,
        dst_batch_stride: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_item = tl.program_id(1)
        pid_t = tl.program_id(2)

        offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        offs_d = tl.arange(0, BLOCK_D)

        src_start = tl.load(src_starts_ptr + pid_item)
        dst_start = tl.load(dst_starts_ptr + pid_item)
        length = tl.load(lengths_ptr + pid_item)

        mask = (offs_t[:, None] < length) & (offs_d[None, :] < HEAD_DIM)
        src_offsets = (
            pid_b * src_batch_stride
            + (src_start + offs_t[:, None]) * HEAD_DIM
            + offs_d[None, :]
        )
        dst_offsets = (
            pid_b * dst_batch_stride
            + (dst_start + offs_t[:, None]) * HEAD_DIM
            + offs_d[None, :]
        )
        k_values = tl.load(src_k_ptr + src_offsets, mask=mask, other=0.0)
        v_values = tl.load(src_v_ptr + src_offsets, mask=mask, other=0.0)
        tl.store(dst_k_ptr + dst_offsets, k_values, mask=mask)
        tl.store(dst_v_ptr + dst_offsets, v_values, mask=mask)


    @triton.jit
    def _fused_real_pruning_segmented_attention_kernel(
        q_ptr,
        persisted_k_ptr,
        persisted_v_ptr,
        segment_starts_ptr,
        segment_ends_ptr,
        segment_counts_ptr,
        current_k_ptr,
        current_v_ptr,
        output_ptr,
        softmax_scale,
        NUM_HEADS: tl.constexpr,
        CURRENT_LEN: tl.constexpr,
        OLD_CAPACITY: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        MAX_SEGMENTS: tl.constexpr,
        FLAT_LAYOUT: tl.constexpr,
        OUTPUT_FLAT: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        USE_EXP2: tl.constexpr,
        QK_DOT_MODE: tl.constexpr,
        VALUE_DOT_MODE: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_b = tl.program_id(2)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, HEAD_DIM)

        batch_head = pid_b * NUM_HEADS + pid_h
        current_base = batch_head * CURRENT_LEN
        if FLAT_LAYOUT:
            old_base = pid_b * OLD_CAPACITY
        else:
            old_base = batch_head * OLD_CAPACITY

        q_ptrs = q_ptr + (current_base + offs_m[:, None]) * HEAD_DIM + offs_d[None, :]
        q = tl.load(q_ptrs, mask=offs_m[:, None] < CURRENT_LEN, other=0.0)

        acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
        m_i = tl.full((BLOCK_M,), -float("inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

        segment_count = tl.load(segment_counts_ptr + pid_h)
        segment_idx = 0
        while segment_idx < segment_count:
            segment_base = pid_h * MAX_SEGMENTS + segment_idx
            segment_start = tl.load(segment_starts_ptr + segment_base)
            segment_end = tl.load(segment_ends_ptr + segment_base)
            old_start = segment_start
            while old_start < segment_end:
                old_offsets = old_start + offs_n
                key_valid = old_offsets < segment_end

                k_ptrs = (
                    persisted_k_ptr
                    + (old_base + old_offsets[None, :]) * HEAD_DIM
                    + offs_d[:, None]
                )
                k = tl.load(k_ptrs, mask=key_valid[None, :], other=0.0)

                if QK_DOT_MODE == 0:
                    scores = tl.dot(q, k) * softmax_scale
                elif QK_DOT_MODE == 1:
                    scores = tl.dot(q.to(tl.float32), k.to(tl.float32)) * softmax_scale
                else:
                    scores = (
                        tl.dot(
                            q.to(tl.float32),
                            k.to(tl.float32),
                            input_precision="ieee",
                        )
                        * softmax_scale
                    )
                scores = tl.where(key_valid[None, :], scores, -float("inf"))

                block_has_valid = tl.sum(tl.where(key_valid, 1, 0), axis=0) > 0
                m_ij = tl.max(scores, axis=1)
                m_new = tl.where(block_has_valid, tl.maximum(m_i, m_ij), m_i)
                safe_m = tl.where(block_has_valid, m_new, 0.0)
                if USE_EXP2:
                    alpha = tl.where(block_has_valid, tl.exp2(m_i - safe_m), 1.0)
                else:
                    alpha = tl.where(block_has_valid, tl.exp(m_i - safe_m), 1.0)
                safe_scores = tl.where(key_valid[None, :], scores, safe_m[:, None])
                if USE_EXP2:
                    p = tl.exp2(safe_scores - safe_m[:, None])
                else:
                    p = tl.exp(safe_scores - safe_m[:, None])
                p = tl.where(key_valid[None, :] & block_has_valid, p, 0.0)

                v_ptrs = (
                    persisted_v_ptr
                    + (old_base + old_offsets[:, None]) * HEAD_DIM
                    + offs_d[None, :]
                )
                v = tl.load(v_ptrs, mask=key_valid[:, None], other=0.0)

                l_new = l_i * alpha + tl.sum(p, axis=1)
                if VALUE_DOT_MODE == 0:
                    value_update = tl.dot(p.to(v.dtype), v)
                elif VALUE_DOT_MODE == 1:
                    value_update = tl.dot(p, v)
                else:
                    value_update = tl.dot(p, v.to(tl.float32), input_precision="ieee")
                acc = acc * alpha[:, None] + value_update
                m_i = m_new
                l_i = l_new
                old_start += BLOCK_N
            segment_idx += 1

        current_start = 0
        while current_start < CURRENT_LEN:
            current_offsets = current_start + offs_n
            key_valid = current_offsets < CURRENT_LEN

            k_ptrs = (
                current_k_ptr
                + (current_base + current_offsets[None, :]) * HEAD_DIM
                + offs_d[:, None]
            )
            k = tl.load(k_ptrs, mask=key_valid[None, :], other=0.0)

            if QK_DOT_MODE == 0:
                scores = tl.dot(q, k) * softmax_scale
            elif QK_DOT_MODE == 1:
                scores = tl.dot(q.to(tl.float32), k.to(tl.float32)) * softmax_scale
            else:
                scores = (
                    tl.dot(
                        q.to(tl.float32),
                        k.to(tl.float32),
                        input_precision="ieee",
                    )
                    * softmax_scale
                )
            scores = tl.where(key_valid[None, :], scores, -float("inf"))

            block_has_valid = tl.sum(tl.where(key_valid, 1, 0), axis=0) > 0
            m_ij = tl.max(scores, axis=1)
            m_new = tl.where(block_has_valid, tl.maximum(m_i, m_ij), m_i)
            safe_m = tl.where(block_has_valid, m_new, 0.0)
            if USE_EXP2:
                alpha = tl.where(block_has_valid, tl.exp2(m_i - safe_m), 1.0)
            else:
                alpha = tl.where(block_has_valid, tl.exp(m_i - safe_m), 1.0)
            safe_scores = tl.where(key_valid[None, :], scores, safe_m[:, None])
            if USE_EXP2:
                p = tl.exp2(safe_scores - safe_m[:, None])
            else:
                p = tl.exp(safe_scores - safe_m[:, None])
            p = tl.where(key_valid[None, :] & block_has_valid, p, 0.0)

            v_ptrs = (
                current_v_ptr
                + (current_base + current_offsets[:, None]) * HEAD_DIM
                + offs_d[None, :]
            )
            v = tl.load(v_ptrs, mask=key_valid[:, None], other=0.0)

            l_new = l_i * alpha + tl.sum(p, axis=1)
            if VALUE_DOT_MODE == 0:
                value_update = tl.dot(p.to(v.dtype), v)
            elif VALUE_DOT_MODE == 1:
                value_update = tl.dot(p, v)
            else:
                value_update = tl.dot(p, v.to(tl.float32), input_precision="ieee")
            acc = acc * alpha[:, None] + value_update
            m_i = m_new
            l_i = l_new
            current_start += BLOCK_N

        output = acc / l_i[:, None]
        if OUTPUT_FLAT:
            output_ptrs = (
                output_ptr
                + pid_b * CURRENT_LEN * NUM_HEADS * HEAD_DIM
                + offs_m[:, None] * NUM_HEADS * HEAD_DIM
                + pid_h * HEAD_DIM
                + offs_d[None, :]
            )
        else:
            output_ptrs = (
                output_ptr
                + (current_base + offs_m[:, None]) * HEAD_DIM
                + offs_d[None, :]
            )
        tl.store(output_ptrs, output, mask=offs_m[:, None] < CURRENT_LEN)

else:
    _pack_flat_segments_kv_kernel = None
    _fused_real_pruning_segmented_attention_kernel = None


def _positive_int_env(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return int(default)
    try:
        value = int(raw_value)
    except ValueError:
        return int(default)
    return value if value > 0 else int(default)



def _fused_value_dot_mode_id() -> int:
    mode = os.environ.get("REAL_PRUNING_FUSED_VALUE_DOT_MODE", "bf16").strip().lower()
    if mode in {"bf16", "flash"}:
        return 0
    if mode in {"fp32", "ieee", "mixed", "fp32_probs"}:
        return 2
    raise ValueError(
        "REAL_PRUNING_FUSED_VALUE_DOT_MODE must be one of: bf16, fp32"
    )



def _fused_qk_dot_mode_id() -> int:
    mode = os.environ.get("REAL_PRUNING_FUSED_QK_DOT_MODE", "bf16").strip().lower()
    if mode in {"bf16", "tensorcore"}:
        return 0
    if mode in {"tf32", "fp32_tf32"}:
        return 1
    if mode in {"fp32", "ieee"}:
        return 2
    raise ValueError(
        "REAL_PRUNING_FUSED_QK_DOT_MODE must be one of: bf16, tf32, fp32"
    )



def _flat_segmented_attention_reference(
    *,
    q: torch.Tensor,
    persisted_k: Optional[torch.Tensor],
    persisted_v: Optional[torch.Tensor],
    segment_starts: torch.Tensor,
    segment_ends: torch.Tensor,
    segment_counts: torch.Tensor,
    current_k: torch.Tensor,
    current_v: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Reference implementation for flat-slab source-segment K/V layout tests."""

    if persisted_k is None or persisted_v is None:
        batch_size = current_k.shape[0]
        head_dim = current_k.shape[3]
        persisted_k = current_k.new_empty((batch_size, 0, head_dim))
        persisted_v = current_v.new_empty((batch_size, 0, head_dim))

    batch_size, num_heads, current_len, head_dim = q.shape
    output = torch.empty_like(current_v)
    starts_cpu = segment_starts.detach().cpu()
    ends_cpu = segment_ends.detach().cpu()
    counts_cpu = segment_counts.detach().cpu()
    for batch_idx in range(batch_size):
        for head_idx in range(num_heads):
            keys = []
            values = []
            segment_count = int(counts_cpu[head_idx].item())
            for segment_idx in range(segment_count):
                start = int(starts_cpu[head_idx, segment_idx].item())
                end = int(ends_cpu[head_idx, segment_idx].item())
                if start >= end:
                    continue
                keys.append(persisted_k[batch_idx, start:end, :])
                values.append(persisted_v[batch_idx, start:end, :])
            keys.append(current_k[batch_idx, head_idx, :, :])
            values.append(current_v[batch_idx, head_idx, :, :])
            key = torch.cat(keys, dim=0)
            value = torch.cat(values, dim=0)
            head_output = F.scaled_dot_product_attention(
                query=q[batch_idx : batch_idx + 1, head_idx : head_idx + 1, :, :],
                key=key.view(1, 1, key.shape[0], head_dim),
                value=value.view(1, 1, value.shape[0], head_dim),
                scale=softmax_scale,
                dropout_p=0.0,
            )
            output[batch_idx, head_idx, :, :] = head_output[0, 0, :, :]
    return output.transpose(1, 2).reshape(batch_size, current_len, num_heads * head_dim)



def pack_flat_segments_kv(
    *,
    dst_k: torch.Tensor,
    dst_v: torch.Tensor,
    src_k: torch.Tensor,
    src_v: torch.Tensor,
    dst_starts: torch.Tensor,
    src_starts: torch.Tensor,
    lengths: torch.Tensor,
    max_length: Optional[int] = None,
    block_t: int = 32,
):
    """Copy matching flat K/V segments with one launch per segment tile."""

    for name, tensor in (
        ("dst_k", dst_k),
        ("dst_v", dst_v),
        ("src_k", src_k),
        ("src_v", src_v),
    ):
        if tensor.ndim != 3:
            raise ValueError(f"pack_flat_segments_kv expects {name} shaped [B, S, D]")
    if dst_k.shape != dst_v.shape:
        raise ValueError("pack_flat_segments_kv requires matching K/V destination shapes")
    if src_k.shape != src_v.shape:
        raise ValueError("pack_flat_segments_kv requires matching K/V source shapes")
    if dst_k.shape[0] != src_k.shape[0] or dst_k.shape[2] != src_k.shape[2]:
        raise ValueError("pack_flat_segments_kv requires matching batch and head_dim")
    if not all(tensor.is_cuda for tensor in (dst_k, dst_v, src_k, src_v)):
        raise RuntimeError("pack_flat_segments_kv requires CUDA tensors")
    if triton is None or _pack_flat_segments_kv_kernel is None:
        raise RuntimeError("pack_flat_segments_kv requires Triton")
    if dst_starts.numel() == 0:
        return dst_k, dst_v
    if (
        dst_starts.device != dst_k.device
        or src_starts.device != dst_k.device
        or lengths.device != dst_k.device
    ):
        raise ValueError("pack segment metadata tensors must be on the destination device")
    if dst_starts.numel() != src_starts.numel() or dst_starts.numel() != lengths.numel():
        raise ValueError("pack segment metadata tensors must have matching lengths")
    if not dst_k.is_contiguous() or not dst_v.is_contiguous():
        raise ValueError("pack_flat_segments_kv requires contiguous destination tensors")
    if not src_k.is_contiguous():
        src_k = src_k.contiguous()
    if not src_v.is_contiguous():
        src_v = src_v.contiguous()

    head_dim = int(dst_k.shape[2])
    if head_dim > 128:
        raise RuntimeError("pack_flat_segments_kv currently supports head_dim <= 128")
    block_d = triton.next_power_of_2(head_dim)
    max_len = int(max_length) if max_length is not None else int(lengths.max().item())
    if max_len <= 0:
        return dst_k, dst_v
    grid = (
        int(dst_k.shape[0]),
        int(lengths.numel()),
        triton.cdiv(max_len, int(block_t)),
    )
    _pack_flat_segments_kv_kernel[grid](
        src_k,
        src_v,
        dst_k,
        dst_v,
        src_starts,
        dst_starts,
        lengths,
        src_k.stride(0),
        dst_k.stride(0),
        HEAD_DIM=head_dim,
        BLOCK_T=int(block_t),
        BLOCK_D=block_d,
        num_warps=4,
    )
    return dst_k, dst_v



def fused_real_pruning_flat_segmented_attention(
    *,
    q: torch.Tensor,
    persisted_k: Optional[torch.Tensor],
    persisted_v: Optional[torch.Tensor],
    segment_starts: torch.Tensor,
    segment_ends: torch.Tensor,
    segment_counts: torch.Tensor,
    current_k: torch.Tensor,
    current_v: torch.Tensor,
    softmax_scale: float,
    allow_cpu_fallback: bool = False,
    softmax_mode: str = "exp",
) -> torch.Tensor:
    """Compute attention over a flat K/V segment slab without compacting old K/V.

    `persisted_k/v` are `[B, capacity_slots, D]`. Segment starts/ends are flat
    token-slot offsets into that slab, grouped per attention head.
    """

    if q.ndim != 4 or current_k.ndim != 4 or current_v.ndim != 4:
        raise ValueError("q, current_k, and current_v must have shape [B, H, L, D]")
    if q.shape != current_k.shape or q.shape != current_v.shape:
        raise ValueError(
            "q, current_k, and current_v must have identical [B, H, L, D] shapes"
        )
    if q.device != current_k.device or q.device != current_v.device:
        raise ValueError("q, current_k, and current_v must be on the same device")
    if persisted_k is None or persisted_v is None:
        batch_size = current_k.shape[0]
        head_dim = current_k.shape[3]
        persisted_k = current_k.new_empty((batch_size, 0, head_dim))
        persisted_v = current_v.new_empty((batch_size, 0, head_dim))
    if persisted_k.ndim != 3 or persisted_v.ndim != 3:
        raise ValueError("persisted_k and persisted_v must have shape [B, S, D]")
    if persisted_k.shape != persisted_v.shape:
        raise ValueError("persisted_k and persisted_v must have identical shapes")
    if persisted_k.shape[0] != q.shape[0]:
        raise ValueError("persisted K/V batch dimension must match q")
    if persisted_k.shape[2] != q.shape[3]:
        raise ValueError("persisted K/V head_dim must match q")
    if persisted_k.device != q.device or persisted_v.device != q.device:
        raise ValueError("persisted K/V must be on the same device as q")
    if persisted_k.dtype != current_k.dtype:
        raise ValueError("persisted_k dtype must match current_k")
    if persisted_v.dtype != current_v.dtype:
        raise ValueError("persisted_v dtype must match current_v")

    batch_size, num_heads, current_len, head_dim = q.shape
    if segment_starts.shape != segment_ends.shape:
        raise ValueError("segment_starts and segment_ends must have identical shapes")
    if segment_starts.ndim != 2 or segment_starts.shape[0] != num_heads:
        raise ValueError("segment_starts must have shape [H, S]")
    if tuple(segment_counts.shape) != (num_heads,):
        raise ValueError("segment_counts must have shape [H]")
    if (
        segment_starts.device != q.device
        or segment_ends.device != q.device
        or segment_counts.device != q.device
    ):
        raise ValueError("segment metadata must be on the same device as q")
    if softmax_mode not in {"exp", "exp2"}:
        raise ValueError(f"unsupported softmax_mode {softmax_mode!r}")

    if q.device.type != "cuda":
        if allow_cpu_fallback:
            return _flat_segmented_attention_reference(
                q=q,
                persisted_k=persisted_k,
                persisted_v=persisted_v,
                segment_starts=segment_starts,
                segment_ends=segment_ends,
                segment_counts=segment_counts,
                current_k=current_k,
                current_v=current_v,
                softmax_scale=softmax_scale,
            )
        raise RuntimeError("real_scale_fused_segmented requires CUDA tensors")

    if triton is None or _fused_real_pruning_segmented_attention_kernel is None:
        raise RuntimeError("real_scale_fused_segmented requires Triton to be installed")
    if head_dim != 128:
        raise RuntimeError("real_scale_fused_segmented v1 only supports head_dim=128")
    if current_len <= 0:
        raise RuntimeError("real_scale_fused_segmented requires current_len > 0")
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise RuntimeError(
            "real_scale_fused_segmented supports only float16, bfloat16, or float32 q tensors"
        )
    if current_k.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise RuntimeError(
            "real_scale_fused_segmented supports only float16, bfloat16, or float32 k tensors"
        )
    if current_v.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise RuntimeError(
            "real_scale_fused_segmented supports only float16, bfloat16, or float32 v tensors"
        )

    q = q.contiguous()
    persisted_k = persisted_k.contiguous()
    persisted_v = persisted_v.contiguous()
    segment_starts = segment_starts.to(dtype=torch.int32).contiguous()
    segment_ends = segment_ends.to(dtype=torch.int32).contiguous()
    segment_counts = segment_counts.to(dtype=torch.int32).contiguous()
    current_k = current_k.contiguous()
    current_v = current_v.contiguous()

    old_capacity = int(persisted_k.shape[1])
    max_segments = int(segment_starts.shape[1])
    output = q.new_empty((batch_size, current_len, num_heads * head_dim))
    block_m = _positive_int_env("REAL_PRUNING_FUSED_BLOCK_M", 64)
    block_n = _positive_int_env("REAL_PRUNING_FUSED_BLOCK_N", 64)
    num_warps = _positive_int_env("REAL_PRUNING_FUSED_NUM_WARPS", 4)
    num_stages = _positive_int_env("REAL_PRUNING_FUSED_NUM_STAGES", 3)
    qk_dot_mode = _fused_qk_dot_mode_id()
    value_dot_mode = _fused_value_dot_mode_id()
    kernel_softmax_scale = float(softmax_scale)
    if softmax_mode == "exp2":
        kernel_softmax_scale *= 1.4426950408889634
    grid = (math.ceil(current_len / block_m), num_heads, batch_size)
    _fused_real_pruning_segmented_attention_kernel[grid](
        q,
        persisted_k,
        persisted_v,
        segment_starts,
        segment_ends,
        segment_counts,
        current_k,
        current_v,
        output,
        kernel_softmax_scale,
        NUM_HEADS=num_heads,
        CURRENT_LEN=current_len,
        OLD_CAPACITY=old_capacity,
        HEAD_DIM=head_dim,
        MAX_SEGMENTS=max_segments,
        FLAT_LAYOUT=True,
        OUTPUT_FLAT=True,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        USE_EXP2=(softmax_mode == "exp2"),
        QK_DOT_MODE=qk_dot_mode,
        VALUE_DOT_MODE=value_dot_mode,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return output
