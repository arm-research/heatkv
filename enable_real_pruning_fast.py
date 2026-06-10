# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT

from __future__ import annotations

import os
import time
import types
from typing import Mapping, Optional

import torch
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention as slow_attn

from enable_pruning import (
    ScaleSpecificPruningController,
    _assign_layer_indices,
    _validate_and_normalize_removed_heads_by_scale_specific,
    _validate_and_normalize_scale_specific_removal_orders,
)
from infinity.models.basic import SelfAttention, apply_rotary_emb
from real_pruning_fused_attention import (
    fused_real_pruning_flat_segmented_attention,
    pack_flat_segments_kv,
)
from tools.utils import (
    get_scale_schedule,
    strict_removed_heads_by_scale_specific_for_budget,
)


_PHASE_PROFILE_ENABLED = False
_PHASE_CPU_NS: dict[str, int] = {}
_PHASE_CUDA_MS: dict[str, float] = {}
_PHASE_CALLS: dict[str, int] = {}
_SEGMENTED_FLAT_STATS: dict[str, int | float] = {}
_PROFILE_PHASES = {"materialize", "attention", "compact"}


def reset_real_pruning_fast_phase_profile() -> None:
    _PHASE_CPU_NS.clear()
    _PHASE_CUDA_MS.clear()
    _PHASE_CALLS.clear()
    _SEGMENTED_FLAT_STATS.clear()


def set_real_pruning_fast_phase_profile_enabled(enabled: bool = True) -> None:
    global _PHASE_PROFILE_ENABLED
    _PHASE_PROFILE_ENABLED = bool(enabled)
    reset_real_pruning_fast_phase_profile()


def snapshot_real_pruning_fast_phase_profile(*, reset: bool = False) -> dict:
    result = {
        "cpu_ms": {name: ns / 1_000_000.0 for name, ns in _PHASE_CPU_NS.items()},
        "cuda_ms": dict(_PHASE_CUDA_MS),
        "calls": dict(_PHASE_CALLS),
        "segmented_flat_stats": dict(_SEGMENTED_FLAT_STATS),
    }
    if reset:
        reset_real_pruning_fast_phase_profile()
    return result


def _phase_profile_start(name: str, *, layer_idx=None, scale_ind=None):
    if not _PHASE_PROFILE_ENABLED or name not in _PROFILE_PHASES:
        return None
    start_event = end_event = None
    if torch.cuda.is_available():
        try:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        except Exception:
            start_event = end_event = None
    return (name, time.perf_counter_ns(), start_event, end_event)


def _phase_profile_end(token):
    if token is None:
        return
    name, start_ns, start_event, end_event = token
    _PHASE_CPU_NS[name] = _PHASE_CPU_NS.get(name, 0) + (time.perf_counter_ns() - start_ns)
    _PHASE_CALLS[name] = _PHASE_CALLS.get(name, 0) + 1
    if start_event is not None and end_event is not None:
        try:
            end_event.record()
            end_event.synchronize()
            _PHASE_CUDA_MS[name] = _PHASE_CUDA_MS.get(name, 0.0) + float(start_event.elapsed_time(end_event))
        except Exception:
            pass


def _phase_detail_subprofile_start(name: str, *, layer_idx=None, scale_ind=None):
    return None


def _segmented_flat_stats_enabled() -> bool:
    return _PHASE_PROFILE_ENABLED


def _segmented_flat_stat_add(name: str, value: int | float = 1):
    if not _segmented_flat_stats_enabled():
        return
    _SEGMENTED_FLAT_STATS[name] = _SEGMENTED_FLAT_STATS.get(name, 0) + value


def _segmented_flat_stat_max(name: str, value: int | float):
    if not _segmented_flat_stats_enabled():
        return
    value = int(value) if isinstance(value, int) else float(value)
    current = _SEGMENTED_FLAT_STATS.get(name)
    if current is None or value > current:
        _SEGMENTED_FLAT_STATS[name] = value


def _tensor_copy_bytes_for_slots(
    *,
    slots: int,
    batch_size: int,
    head_dim: int,
    k_dtype: torch.dtype,
    v_dtype: torch.dtype,
) -> int:
    slots = int(slots)
    if slots <= 0:
        return 0
    k_bytes = torch.empty((), dtype=k_dtype).element_size()
    v_bytes = torch.empty((), dtype=v_dtype).element_size()
    return int(slots) * int(batch_size) * int(head_dim) * (k_bytes + v_bytes)


def _segmented_flat_record_kv_copy(
    prefix: str,
    *,
    slots: int,
    k_tensor: torch.Tensor,
    v_tensor: torch.Tensor,
):
    if not _segmented_flat_stats_enabled():
        return
    slots = int(slots)
    _segmented_flat_stat_add(f"{prefix}_slots", slots)
    _segmented_flat_stat_add(
        f"{prefix}_bytes",
        _tensor_copy_bytes_for_slots(
            slots=slots,
            batch_size=k_tensor.shape[0],
            head_dim=k_tensor.shape[-1],
            k_dtype=k_tensor.dtype,
            v_dtype=v_tensor.dtype,
        ),
    )


def _segmented_flat_record_metadata(prefix: str, tensors: tuple[torch.Tensor, ...]):
    if not _segmented_flat_stats_enabled():
        return
    _segmented_flat_stat_add(f"{prefix}_calls")
    _segmented_flat_stat_add(
        f"{prefix}_bytes",
        sum(int(tensor.numel()) * int(tensor.element_size()) for tensor in tensors),
    )


def _fused_softmax_mode() -> str:
    mode = os.environ.get("REAL_PRUNING_FUSED_SOFTMAX_MODE", "exp")
    if mode not in {"exp", "exp2"}:
        return "exp"
    return mode

def _validate_merge_mode(merge_mode: str) -> str:
    if merge_mode != "exact":
        raise ValueError(
            "real pruning fast v1 only supports merge_mode='exact'"
        )
    return merge_mode

def _empty_fast_cache_state(
    *,
    batch_size: int,
    num_heads: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
):
    return (
        torch.empty((batch_size, num_heads, 0, head_dim), device=device, dtype=dtype),
        torch.empty((batch_size, num_heads, 0, head_dim), device=device, dtype=dtype),
        torch.zeros((num_heads,), device=device, dtype=torch.long),
        torch.empty((num_heads, 0), device=device, dtype=torch.long),
        torch.empty((num_heads, 0), device=device, dtype=torch.long),
    )

def _has_any_removed_heads(
    removed_heads_by_source_scale: Mapping[int, set[int]],
) -> bool:
    return any(bool(heads) for heads in removed_heads_by_source_scale.values())

def _static_logical_cache_capacity(scale_ranges: Mapping[int, tuple[int, int]]) -> int:
    if not scale_ranges:
        return 0
    final_scale_idx = max(int(scale_idx) for scale_idx in scale_ranges)
    return int(scale_ranges[final_scale_idx][0])

def _segmented_flat_cache_capacity_for_layer(
    controller,
    *,
    layer_idx: int,
    num_heads: int,
) -> int:
    """Max flat token slots needed for one layer's segmented cache."""

    max_slots = 0
    scale_ranges = controller.scale_ranges
    num_scales = len(controller.scale_schedule)
    for target_scale_idx in range(1, num_scales):
        removed_by_source = controller.removed_heads_by_source_for(
            layer_idx,
            target_scale_idx,
        )
        total_slots = 0
        for source_scale_idx, scale_range in scale_ranges.items():
            source_scale_idx = int(source_scale_idx)
            if source_scale_idx >= target_scale_idx:
                continue
            scale_start, scale_end = scale_range
            token_count = int(scale_end) - int(scale_start)
            if token_count <= 0:
                continue
            removed_heads = removed_by_source.get(source_scale_idx, set())
            total_slots += (int(num_heads) - len(removed_heads)) * token_count
        max_slots = max(max_slots, total_slots)
    return int(max_slots)

def _segmented_flat_static_schedule_enabled() -> bool:
    return os.environ.get(
        "REAL_PRUNING_SEGMENTED_STATIC_SCHEDULE",
        "1",
    ) not in {"0", "false", "False", "no", "NO"}

def _segmented_flat_static_schedule_allow_overalloc() -> bool:
    # Lower-latency default: allow a fixed non-moving slab layout even when it
    # needs more compact-cache slots than the dynamic allocator. Set either
    # REAL_PRUNING_SEGMENTED_STATIC_SCHEDULE=0 or this flag to 0 to opt out to
    # the lower-memory dynamic segmented cache.
    return os.environ.get(
        "REAL_PRUNING_SEGMENTED_STATIC_SCHEDULE_ALLOW_OVERALLOC",
        "1",
    ) not in {"0", "false", "False", "no", "NO"}

def _segmented_flat_triton_static_copy_enabled() -> bool:
    # Lower-latency default for the segmented fused path. Set this to 0 to use
    # the older torch.index_select/copy static-copy implementation.
    return os.environ.get(
        "REAL_PRUNING_SEGMENTED_TRITON_STATIC_COPY",
        "1",
    ) in {"1", "true", "True", "yes", "YES"}

def _segmented_flat_build_static_slot_schedule(
    controller,
    *,
    layer_idx: int,
    num_heads: int,
):
    """Precompute non-overlapping flat-slab slots for source-scale/head segments."""

    capacity_bound = _segmented_flat_cache_capacity_for_layer(
        controller,
        layer_idx=layer_idx,
        num_heads=num_heads,
    )
    scale_ranges = controller.scale_ranges
    num_scales = len(controller.scale_schedule)
    intervals = []
    for source_scale_idx, scale_range in sorted(scale_ranges.items()):
        source_scale_idx = int(source_scale_idx)
        start_target = source_scale_idx + 1
        if start_target >= num_scales:
            continue
        scale_start, scale_end = scale_range
        length = int(scale_end) - int(scale_start)
        if length <= 0:
            continue
        for head_idx in range(int(num_heads)):
            end_target = num_scales
            for target_scale_idx in range(start_target, num_scales):
                removed_by_source = controller.removed_heads_by_source_for(
                    layer_idx,
                    target_scale_idx,
                )
                if head_idx in removed_by_source.get(source_scale_idx, set()):
                    end_target = target_scale_idx
                    break
            if end_target <= start_target:
                continue
            intervals.append(
                {
                    "source_scale": source_scale_idx,
                    "head": int(head_idx),
                    "start_target": int(start_target),
                    "end_target": int(end_target),
                    "length": int(length),
                    "scale_start": int(scale_start),
                    "scale_end": int(scale_end),
                }
            )

    free_blocks = []
    active = []
    high_water = 0
    assigned = []
    for interval in sorted(
        intervals,
        key=lambda item: (
            item["start_target"],
            item["end_target"],
            -item["length"],
            item["source_scale"],
            item["head"],
        ),
    ):
        still_active = []
        for active_interval in active:
            if int(active_interval["end_target"]) <= int(interval["start_target"]):
                free_blocks.append(
                    (
                        int(active_interval["offset"]),
                        int(active_interval["length"]),
                    )
                )
            else:
                still_active.append(active_interval)
        active = still_active
        free_blocks = _segmented_flat_merge_free_blocks(free_blocks)

        offset = None
        length = int(interval["length"])
        best_block = None
        for block_idx, (block_offset, block_length) in enumerate(free_blocks):
            block_offset = int(block_offset)
            block_length = int(block_length)
            if block_length < length:
                continue
            leftover = block_length - length
            if best_block is None or (leftover, block_offset) < (
                best_block[0],
                best_block[2],
            ):
                best_block = (leftover, block_idx, block_offset, block_length)
        if best_block is not None:
            _, block_idx, block_offset, block_length = best_block
            offset = block_offset
            if block_length == length:
                del free_blocks[block_idx]
            else:
                free_blocks[block_idx] = (
                    block_offset + length,
                    block_length - length,
                )
        if offset is None:
            offset = high_water
            high_water += length
        allocated = dict(interval)
        allocated["offset"] = int(offset)
        assigned.append(allocated)
        active.append(allocated)

    if high_water > capacity_bound and not _segmented_flat_static_schedule_allow_overalloc():
        return {
            "enabled": False,
            "capacity": int(capacity_bound),
            "capacity_bound": int(capacity_bound),
            "allocated_capacity": int(high_water),
        }

    overalloc_slots = max(0, int(high_water) - int(capacity_bound))
    segments_by_key = {
        (int(segment["source_scale"]), int(segment["head"])): segment
        for segment in assigned
    }
    segments_by_target = {}
    metadata_cpu_by_target = {}
    for target_scale_idx in range(1, num_scales):
        live_segments = [
            segment
            for segment in assigned
            if int(segment["start_target"])
            <= int(target_scale_idx)
            < int(segment["end_target"])
        ]
        segments_by_target[target_scale_idx] = live_segments
        per_head = [[] for _ in range(int(num_heads))]
        for segment in live_segments:
            per_head[int(segment["head"])].append(segment)
        max_segments = max((len(items) for items in per_head), default=0)
        starts_cpu = torch.zeros((num_heads, max_segments), dtype=torch.int32)
        ends_cpu = torch.zeros((num_heads, max_segments), dtype=torch.int32)
        counts_cpu = torch.zeros((num_heads,), dtype=torch.int32)
        for head_idx, items in enumerate(per_head):
            items = sorted(items, key=lambda item: int(item["source_scale"]))
            counts_cpu[head_idx] = len(items)
            for segment_idx, segment in enumerate(items):
                offset = int(segment["offset"])
                length = int(segment["length"])
                starts_cpu[head_idx, segment_idx] = offset
                ends_cpu[head_idx, segment_idx] = offset + length
        metadata_cpu_by_target[target_scale_idx] = (
            starts_cpu,
            ends_cpu,
            counts_cpu,
        )

    def build_copy_runs(items: list[dict], *, sequence_len: int):
        runs = []
        run_start = None
        run_end = None
        index_parts = []
        copy_items = []

        def flush():
            nonlocal run_start, run_end, index_parts, copy_items
            if run_start is None or run_end is None or not index_parts:
                return
            indices_cpu = torch.cat(index_parts, dim=0)
            contiguous_src_start = None
            contiguous_src_end = None
            if int(indices_cpu.numel()) > 0:
                is_contiguous = True
                if int(indices_cpu.numel()) > 1:
                    is_contiguous = bool(
                        torch.all(indices_cpu[1:] == indices_cpu[:-1] + 1).item()
                    )
                if is_contiguous:
                    contiguous_src_start = int(indices_cpu[0].item())
                    contiguous_src_end = contiguous_src_start + int(indices_cpu.numel())
            runs.append(
                {
                    "dst_start": int(run_start),
                    "dst_end": int(run_end),
                    "sequence_len": int(sequence_len),
                    "indices_cpu": indices_cpu,
                    "indices_device_cache": {},
                    "contiguous_src_start": contiguous_src_start,
                    "contiguous_src_end": contiguous_src_end,
                    "copy_items": copy_items,
                }
            )
            run_start = None
            run_end = None
            index_parts = []
            copy_items = []

        for item in sorted(items, key=lambda value: int(value["dst_offset"])):
            dst_offset = int(item["dst_offset"])
            length = int(item["length"])
            if length <= 0:
                continue
            if run_end is not None and dst_offset != run_end:
                flush()
            if run_start is None:
                run_start = dst_offset
                run_end = dst_offset
            head_idx = int(item["head"])
            src_start = int(item["src_start"])
            copy_items.append(
                {
                    "dst_start": int(dst_offset),
                    "src_start": int(head_idx) * int(sequence_len) + int(src_start),
                    "length": int(length),
                }
            )
            index_parts.append(
                torch.arange(
                    head_idx * int(sequence_len) + src_start,
                    head_idx * int(sequence_len) + src_start + length,
                    dtype=torch.long,
                )
            )
            run_end += length
        flush()
        return runs

    old_pack_runs_by_target = {}
    old_pack_cache_lens_by_target = {}
    append_runs_by_key = {}
    for target_scale_idx, live_segments in segments_by_target.items():
        target_scale_idx = int(target_scale_idx)
        target_range = scale_ranges.get(target_scale_idx)
        if target_range is None:
            continue
        old_cache_len = int(target_range[0])
        old_pack_cache_lens_by_target[target_scale_idx] = old_cache_len
        old_items = []
        for segment in live_segments:
            scale_start = int(segment["scale_start"])
            scale_end = min(int(segment["scale_end"]), old_cache_len)
            if scale_start < scale_end:
                old_items.append(
                    {
                        "dst_offset": int(segment["offset"]),
                        "head": int(segment["head"]),
                        "src_start": scale_start,
                        "length": int(scale_end - scale_start),
                    }
                )
        old_pack_runs_by_target[target_scale_idx] = build_copy_runs(
            old_items,
            sequence_len=old_cache_len,
        )

    for current_scale_idx, scale_range in scale_ranges.items():
        current_scale_idx = int(current_scale_idx)
        next_target_scale_idx = current_scale_idx + 1
        if next_target_scale_idx >= num_scales:
            continue
        current_len = int(scale_range[1]) - int(scale_range[0])
        if current_len <= 0:
            continue
        append_items = []
        for head_idx in range(int(num_heads)):
            segment = segments_by_key.get((current_scale_idx, head_idx))
            if segment is None:
                continue
            if not (
                int(segment["start_target"])
                <= next_target_scale_idx
                < int(segment["end_target"])
            ):
                continue
            append_items.append(
                {
                    "dst_offset": int(segment["offset"]),
                    "head": int(head_idx),
                    "src_start": 0,
                    "length": current_len,
                }
            )
        append_runs_by_key[(current_scale_idx, next_target_scale_idx)] = build_copy_runs(
            append_items,
            sequence_len=current_len,
        )

    return {
        "enabled": True,
        "capacity": int(high_water),
        "capacity_bound": int(capacity_bound),
        "allocated_capacity": int(high_water),
        "overalloc_slots": int(overalloc_slots),
        "segments": assigned,
        "segments_by_key": segments_by_key,
        "segments_by_target": segments_by_target,
        "metadata_cpu_by_target": metadata_cpu_by_target,
        "metadata_device_cache": {},
        "old_pack_runs_by_target": old_pack_runs_by_target,
        "old_pack_cache_lens_by_target": old_pack_cache_lens_by_target,
        "append_runs_by_key": append_runs_by_key,
    }

def _segmented_flat_static_schedule(self):
    if not _segmented_flat_static_schedule_enabled():
        return None
    schedule = getattr(self, "_real_pruning_segmented_static_schedule", None)
    if not schedule or not schedule.get("enabled", False):
        return None
    return schedule

def _segmented_flat_static_attention_metadata(
    self,
    *,
    target_scale_idx: int,
    device: torch.device,
):
    schedule = _segmented_flat_static_schedule(self)
    if schedule is None:
        return None
    target_scale_idx = int(target_scale_idx)
    cache = schedule.setdefault("metadata_device_cache", {})
    key = (str(device), target_scale_idx)
    cached = cache.get(key)
    if cached is not None:
        _segmented_flat_stat_add("static_metadata_cache_hits")
        return cached
    cpu_metadata = schedule["metadata_cpu_by_target"].get(target_scale_idx)
    if cpu_metadata is None:
        return None
    metadata = tuple(tensor.to(device=device) for tensor in cpu_metadata)
    _segmented_flat_record_metadata("static_metadata_device_copy", metadata)
    cache[key] = metadata
    return metadata

def _segmented_flat_prefill_static_metadata_device_cache(
    schedule: dict | None,
    *,
    device: torch.device,
):
    """Move immutable static attention metadata to the target device once."""

    if schedule is None or not schedule.get("enabled", False):
        return
    device = torch.device(device)
    cache = schedule.setdefault("metadata_device_cache", {})
    for target_scale_idx, cpu_metadata in schedule.get("metadata_cpu_by_target", {}).items():
        key = (str(device), int(target_scale_idx))
        cached = cache.get(key)
        if cached is not None and all(tensor.device == device for tensor in cached):
            continue
        cache[key] = tuple(tensor.to(device=device) for tensor in cpu_metadata)

def _segmented_flat_static_copy_metadata_for_runs(
    runs: list[dict],
    *,
    device: torch.device,
):
    if not runs:
        return None
    cache_owner = runs[0]
    prepared = cache_owner.get("triton_copy_items_cpu_cache")
    if prepared is None:
        all_items = []
        for run in runs:
            all_items.extend(run.get("copy_items", []))
        if not all_items:
            cache_owner["triton_copy_items_cpu_cache"] = False
            return None
        dst_start_values = [int(item["dst_start"]) for item in all_items]
        src_start_values = [int(item["src_start"]) for item in all_items]
        length_values = [int(item["length"]) for item in all_items]
        prepared = {
            "dst_starts": dst_start_values,
            "src_starts": src_start_values,
            "lengths": length_values,
            "max_length": max(length_values),
            "slots": sum(length_values),
            "item_count": len(all_items),
        }
        cache_owner["triton_copy_items_cpu_cache"] = prepared
    if prepared is False:
        return None

    device = torch.device(device)
    cache = cache_owner.setdefault("triton_copy_items_device_cache", {})
    device_key = str(device)
    cached = cache.get(device_key)
    created = False
    if cached is None or cached[0].device != device:
        dst_starts = torch.tensor(
            prepared["dst_starts"],
            device=device,
            dtype=torch.int64,
        )
        src_starts = torch.tensor(
            prepared["src_starts"],
            device=device,
            dtype=torch.int64,
        )
        lengths = torch.tensor(
            prepared["lengths"],
            device=device,
            dtype=torch.int64,
        )
        cached = (dst_starts, src_starts, lengths)
        cache[device_key] = cached
        created = True
    return {
        "tensors": cached,
        "max_length": int(prepared["max_length"]),
        "slots": int(prepared["slots"]),
        "item_count": int(prepared["item_count"]),
        "created": created,
    }

def _segmented_flat_prefill_static_copy_device_cache(
    schedule: dict | None,
    *,
    device: torch.device,
):
    """Move immutable static copy descriptors to the target device once."""

    if schedule is None or not schedule.get("enabled", False):
        return
    if not _segmented_flat_triton_static_copy_enabled():
        return
    device = torch.device(device)
    if device.type != "cuda":
        return
    run_lists = list(schedule.get("old_pack_runs_by_target", {}).values())
    run_lists.extend(schedule.get("append_runs_by_key", {}).values())
    for runs in run_lists:
        _segmented_flat_static_copy_metadata_for_runs(runs, device=device)

def _segmented_flat_static_copy_flat_items(
    *,
    dst_k: torch.Tensor,
    dst_v: torch.Tensor,
    src_k: torch.Tensor,
    src_v: torch.Tensor,
    sequence_len: int,
    items: list[dict],
):
    if not items:
        return
    batch_size, num_heads, _, head_dim = src_k.shape
    flat_k = src_k.reshape(batch_size, num_heads * int(sequence_len), head_dim)
    flat_v = src_v.reshape(batch_size, num_heads * int(sequence_len), head_dim)

    run_start = None
    run_end = None
    index_parts = []

    def flush():
        nonlocal run_start, run_end, index_parts
        if run_start is None or run_end is None or not index_parts:
            return
        indices = torch.cat(index_parts, dim=0)
        slots = int(run_end) - int(run_start)
        _segmented_flat_stat_add("static_pack_runs")
        _segmented_flat_stat_add("static_pack_index_elements", int(indices.numel()))
        _segmented_flat_record_kv_copy(
            "static_pack_copy",
            slots=slots,
            k_tensor=src_k,
            v_tensor=src_v,
        )
        torch.index_select(
            flat_k,
            1,
            indices,
            out=dst_k[:, run_start:run_end, :],
        )
        torch.index_select(
            flat_v,
            1,
            indices,
            out=dst_v[:, run_start:run_end, :],
        )
        run_start = None
        run_end = None
        index_parts = []

    for item in sorted(items, key=lambda value: int(value["dst_offset"])):
        dst_offset = int(item["dst_offset"])
        length = int(item["length"])
        if length <= 0:
            continue
        if run_end is not None and dst_offset != run_end:
            flush()
        if run_start is None:
            run_start = dst_offset
            run_end = dst_offset
        index_parts.append(
            _segmented_flat_index_range(
                head_idx=int(item["head"]),
                sequence_len=int(sequence_len),
                start=int(item["src_start"]),
                end=int(item["src_start"]) + length,
                device=src_k.device,
            )
        )
        run_end += length
    flush()

def _segmented_flat_static_copy_precomputed_runs(
    *,
    dst_k: torch.Tensor,
    dst_v: torch.Tensor,
    src_k: torch.Tensor,
    src_v: torch.Tensor,
    runs: list[dict],
):
    if not runs:
        return
    batch_size, num_heads, sequence_len, head_dim = src_k.shape
    flat_k = src_k.reshape(batch_size, num_heads * int(sequence_len), head_dim)
    flat_v = src_v.reshape(batch_size, num_heads * int(sequence_len), head_dim)
    if _segmented_flat_triton_static_copy_enabled() and src_k.is_cuda:
        metadata = _segmented_flat_static_copy_metadata_for_runs(
            runs,
            device=src_k.device,
        )
        if metadata is not None:
            if metadata["created"]:
                _segmented_flat_stat_add("triton_static_copy_item_device_copies")
                _segmented_flat_stat_add(
                    "triton_static_copy_item_count",
                    metadata["item_count"],
                )
            dst_starts, src_starts, lengths = metadata["tensors"]
            slots = int(metadata["slots"])
            _segmented_flat_stat_add("triton_static_copy_calls")
            _segmented_flat_record_kv_copy(
                "triton_static_copy",
                slots=slots,
                k_tensor=src_k,
                v_tensor=src_v,
            )
            pack_flat_segments_kv(
                dst_k=dst_k,
                dst_v=dst_v,
                src_k=flat_k,
                src_v=flat_v,
                dst_starts=dst_starts,
                src_starts=src_starts,
                lengths=lengths,
                max_length=int(metadata["max_length"]),
            )
            return
    device_key = str(src_k.device)
    for run in runs:
        expected_sequence_len = int(run.get("sequence_len", sequence_len))
        if expected_sequence_len != int(sequence_len):
            raise RuntimeError("static precomputed copy run sequence length mismatch")
        run_start = int(run["dst_start"])
        run_end = int(run["dst_end"])
        if run_end <= run_start:
            continue
        contiguous_src_start = run.get("contiguous_src_start")
        contiguous_src_end = run.get("contiguous_src_end")
        if contiguous_src_start is not None and contiguous_src_end is not None:
            contiguous_src_start = int(contiguous_src_start)
            contiguous_src_end = int(contiguous_src_end)
            if contiguous_src_end - contiguous_src_start != run_end - run_start:
                raise RuntimeError("static precomputed contiguous copy length mismatch")
            slots = run_end - run_start
            _segmented_flat_stat_add("static_precomputed_contiguous_copy_runs")
            _segmented_flat_record_kv_copy(
                "static_precomputed_contiguous_copy",
                slots=slots,
                k_tensor=src_k,
                v_tensor=src_v,
            )
            dst_k[:, run_start:run_end, :].copy_(
                flat_k[:, contiguous_src_start:contiguous_src_end, :]
            )
            dst_v[:, run_start:run_end, :].copy_(
                flat_v[:, contiguous_src_start:contiguous_src_end, :]
            )
            continue
        cache = run.setdefault("indices_device_cache", {})
        indices = cache.get(device_key)
        if indices is None or indices.device != src_k.device:
            indices = run["indices_cpu"].to(device=src_k.device)
            cache[device_key] = indices
            _segmented_flat_stat_add("static_precomputed_index_device_copy_bytes")
            _segmented_flat_stat_add(
                "static_precomputed_index_device_copy_elements",
                int(indices.numel()),
            )
        slots = run_end - run_start
        _segmented_flat_stat_add("static_precomputed_copy_runs")
        _segmented_flat_stat_add("static_pack_index_elements", int(indices.numel()))
        _segmented_flat_record_kv_copy(
            "static_pack_copy",
            slots=slots,
            k_tensor=src_k,
            v_tensor=src_v,
        )
        torch.index_select(
            flat_k,
            1,
            indices,
            out=dst_k[:, run_start:run_end, :],
        )
        torch.index_select(
            flat_v,
            1,
            indices,
            out=dst_v[:, run_start:run_end, :],
        )

def _segmented_flat_static_pack_state(
    self,
    *,
    target_scale_idx: int,
    persisted_k: Optional[torch.Tensor],
    persisted_v: Optional[torch.Tensor],
    current_k: Optional[torch.Tensor],
    current_v: Optional[torch.Tensor],
    old_cache_len: int,
    current_scale_idx: Optional[int],
):
    schedule = _segmented_flat_static_schedule(self)
    if schedule is None:
        return False
    _segmented_flat_stat_add("static_pack_state_calls")
    target_scale_idx = int(target_scale_idx)
    current_scale_idx = None if current_scale_idx is None else int(current_scale_idx)
    if current_scale_idx is None and current_k is None and current_v is None:
        expected_old_cache_len = schedule.get("old_pack_cache_lens_by_target", {}).get(
            target_scale_idx
        )
        runs = schedule.get("old_pack_runs_by_target", {}).get(target_scale_idx)
        if (
            runs is not None
            and expected_old_cache_len is not None
            and int(expected_old_cache_len) == int(old_cache_len)
        ):
            if persisted_k is None or persisted_v is None:
                raise RuntimeError("segmented static pack missing persisted cache")
            _segmented_flat_stat_add("static_precomputed_pack_state_hits")
            _segmented_flat_static_copy_precomputed_runs(
                dst_k=self.cached_k,
                dst_v=self.cached_v,
                src_k=persisted_k,
                src_v=persisted_v,
                runs=runs,
            )
            return True

    old_items = []
    current_items = []
    current_len = 0 if current_k is None else int(current_k.shape[2])
    for segment in schedule["segments_by_target"].get(target_scale_idx, []):
        source_scale_idx = int(segment["source_scale"])
        scale_start = int(segment["scale_start"])
        scale_end = int(segment["scale_end"])
        if current_scale_idx is not None and source_scale_idx == current_scale_idx:
            if current_k is None or current_v is None:
                continue
            src_start = max(scale_start, int(old_cache_len)) - int(old_cache_len)
            src_end = min(scale_end, int(old_cache_len) + current_len) - int(old_cache_len)
            if src_start < src_end:
                current_items.append(
                    {
                        "dst_offset": int(segment["offset"]),
                        "head": int(segment["head"]),
                        "src_start": int(src_start),
                        "length": int(src_end - src_start),
                    }
                )
            continue
        if persisted_k is None or persisted_v is None:
            continue
        src_start = max(scale_start, 0)
        src_end = min(scale_end, int(old_cache_len))
        if src_start < src_end:
            old_items.append(
                {
                    "dst_offset": int(segment["offset"]),
                    "head": int(segment["head"]),
                    "src_start": int(src_start),
                    "length": int(src_end - src_start),
                }
            )
    if old_items:
        if persisted_k is None or persisted_v is None:
            raise RuntimeError("segmented static pack missing persisted cache")
        _segmented_flat_static_copy_flat_items(
            dst_k=self.cached_k,
            dst_v=self.cached_v,
            src_k=persisted_k,
            src_v=persisted_v,
            sequence_len=old_cache_len,
            items=old_items,
        )
    if current_items:
        if current_k is None or current_v is None:
            raise RuntimeError("segmented static pack missing current cache")
        _segmented_flat_static_copy_flat_items(
            dst_k=self.cached_k,
            dst_v=self.cached_v,
            src_k=current_k,
            src_v=current_v,
            sequence_len=current_k.shape[2],
            items=current_items,
        )
    return True

def _segmented_flat_static_append_current(
    self,
    *,
    current_k: torch.Tensor,
    current_v: torch.Tensor,
    current_scale_idx: int,
    next_target_scale_idx: int,
) -> bool:
    schedule = _segmented_flat_static_schedule(self)
    if schedule is None:
        return False
    _segmented_flat_stat_add("static_append_current_calls")
    current_scale_idx = int(current_scale_idx)
    next_target_scale_idx = int(next_target_scale_idx)
    runs = schedule.get("append_runs_by_key", {}).get(
        (current_scale_idx, next_target_scale_idx)
    )
    if runs is not None:
        _segmented_flat_stat_add("static_precomputed_append_hits")
        _segmented_flat_static_copy_precomputed_runs(
            dst_k=self.cached_k,
            dst_v=self.cached_v,
            src_k=current_k,
            src_v=current_v,
            runs=runs,
        )
        return True

    items = []
    for head_idx in range(int(current_k.shape[1])):
        segment = schedule["segments_by_key"].get((current_scale_idx, head_idx))
        if segment is None:
            continue
        if not (
            int(segment["start_target"])
            <= int(next_target_scale_idx)
            < int(segment["end_target"])
        ):
            continue
        items.append(
            {
                "dst_offset": int(segment["offset"]),
                "head": int(head_idx),
                "src_start": 0,
                "length": int(current_k.shape[2]),
            }
        )
    _segmented_flat_static_copy_flat_items(
        dst_k=self.cached_k,
        dst_v=self.cached_v,
        src_k=current_k,
        src_v=current_v,
        sequence_len=current_k.shape[2],
        items=items,
    )
    return True

def _segmented_flat_empty_metadata(num_heads: int, device: torch.device):
    return (
        torch.zeros((num_heads,), device=device, dtype=torch.long),
        torch.empty((num_heads, 0), device=device, dtype=torch.long),
        torch.empty((num_heads, 0), device=device, dtype=torch.long),
    )

def _segmented_flat_merge_free_blocks(free_blocks):
    if not free_blocks:
        return []
    merged = []
    for offset, length in sorted((int(o), int(l)) for o, l in free_blocks if int(l) > 0):
        if not merged:
            merged.append([offset, length])
            continue
        last_offset, last_length = merged[-1]
        last_end = last_offset + last_length
        if offset <= last_end:
            merged[-1][1] = max(last_end, offset + length) - last_offset
        else:
            merged.append([offset, length])
    return [(offset, length) for offset, length in merged]

def _segmented_flat_reset_runtime_state(self):
    self._real_pruning_segmented_cache_active = False
    self._real_pruning_segmented_segments = []
    self._real_pruning_segmented_free_blocks = []
    self._real_pruning_segmented_capacity = 0
    self._real_pruning_segmented_metadata_cache = {}

def _dense_prefix_prealloc_enabled() -> bool:
    return os.environ.get(
        "REAL_PRUNING_DENSE_PREFIX_PREALLOC",
        "0",
    ) not in {"0", "false", "False", "no", "NO"}

def _fused_segmented_baseline_prefix_enabled() -> bool:
    return os.environ.get(
        "REAL_PRUNING_FUSED_SEGMENTED_BASELINE_PREFIX",
        "1",
    ) not in {"0", "false", "False", "no", "NO"}

def _dense_prefix_reset_runtime_state(self):
    self._real_pruning_dense_prefix_k = None
    self._real_pruning_dense_prefix_v = None
    self._real_pruning_dense_prefix_k_blhd = None
    self._real_pruning_dense_prefix_v_blhd = None
    self._real_pruning_dense_prefix_capacity = 0

def _dense_prefix_cache_capacity_for_layer(controller, *, layer_idx: int) -> int:
    scale_ranges = controller.scale_ranges
    num_scales = len(controller.scale_schedule)
    for target_scale_idx in range(1, num_scales):
        removed_by_source = controller.removed_heads_by_source_for(
            layer_idx,
            target_scale_idx,
        )
        if not _has_any_removed_heads(removed_by_source):
            continue
        previous_scale_idx = int(target_scale_idx) - 1
        previous_range = scale_ranges.get(previous_scale_idx)
        if previous_range is None:
            return 0
        return int(previous_range[0])
    return _static_logical_cache_capacity(scale_ranges)

def _segmented_flat_ensure_storage(
    self,
    *,
    batch_size: int,
    head_dim: int,
    device: torch.device,
    k_dtype: torch.dtype,
    v_dtype: torch.dtype,
):
    capacity = int(getattr(self, "_real_pruning_segmented_capacity", 0) or 0)
    if capacity <= 0:
        schedule = _segmented_flat_static_schedule(self)
        if schedule is not None:
            capacity = int(schedule["capacity"])
        else:
            capacity = _segmented_flat_cache_capacity_for_layer(
                self._real_pruning_scale_specific_controller,
                layer_idx=self.layer_idx,
                num_heads=self.num_heads,
            )
        self._real_pruning_segmented_capacity = capacity
    if capacity <= 0:
        self.cached_k = torch.empty((batch_size, 0, head_dim), device=device, dtype=k_dtype)
        self.cached_v = torch.empty((batch_size, 0, head_dim), device=device, dtype=v_dtype)
        self._real_pruning_segmented_free_blocks = []
        return

    needs_allocate = (
        self.cached_k is None
        or self.cached_v is None
        or self.cached_k.ndim != 3
        or self.cached_v.ndim != 3
        or tuple(self.cached_k.shape) != (batch_size, capacity, head_dim)
        or tuple(self.cached_v.shape) != (batch_size, capacity, head_dim)
        or self.cached_k.device != device
        or self.cached_v.device != device
        or self.cached_k.dtype != k_dtype
        or self.cached_v.dtype != v_dtype
    )
    if needs_allocate:
        self.cached_k = torch.empty((batch_size, capacity, head_dim), device=device, dtype=k_dtype)
        self.cached_v = torch.empty((batch_size, capacity, head_dim), device=device, dtype=v_dtype)
        self._real_pruning_segmented_segments = []
        self._real_pruning_segmented_free_blocks = [(0, capacity)]
        self._real_pruning_segmented_metadata_cache = {}
        _segmented_flat_stat_add("storage_allocations")
        _segmented_flat_stat_add("storage_capacity_slots", capacity)
        _segmented_flat_record_kv_copy(
            "storage_capacity",
            slots=capacity,
            k_tensor=self.cached_k,
            v_tensor=self.cached_v,
        )

def _segmented_flat_free_block(self, offset: int, length: int):
    if length <= 0:
        return
    _segmented_flat_stat_add("free_block_calls")
    _segmented_flat_stat_add("free_block_slots", int(length))
    free_blocks = list(getattr(self, "_real_pruning_segmented_free_blocks", []))
    free_blocks.append((int(offset), int(length)))
    self._real_pruning_segmented_free_blocks = _segmented_flat_merge_free_blocks(free_blocks)

def _segmented_flat_defragment(self):
    _segmented_flat_stat_add("defrag_calls")
    segments = sorted(
        getattr(self, "_real_pruning_segmented_segments", []),
        key=lambda item: (int(item["source_scale"]), int(item["head"]), int(item["offset"])),
    )
    if not segments:
        capacity = int(getattr(self, "_real_pruning_segmented_capacity", 0) or 0)
        self._real_pruning_segmented_free_blocks = [(0, capacity)] if capacity else []
        return

    next_k = torch.empty_like(self.cached_k)
    next_v = torch.empty_like(self.cached_v)
    next_offset = 0
    moved_slots = 0
    for segment in segments:
        old_offset = int(segment["offset"])
        length = int(segment["length"])
        moved_slots += length
        next_k[:, next_offset : next_offset + length, :] = self.cached_k[
            :,
            old_offset : old_offset + length,
            :,
        ]
        next_v[:, next_offset : next_offset + length, :] = self.cached_v[
            :,
            old_offset : old_offset + length,
            :,
        ]
        segment["offset"] = next_offset
        next_offset += length
    self.cached_k = next_k
    self.cached_v = next_v
    _segmented_flat_record_kv_copy(
        "defrag_copy",
        slots=moved_slots,
        k_tensor=self.cached_k,
        v_tensor=self.cached_v,
    )
    capacity = int(getattr(self, "_real_pruning_segmented_capacity", 0) or 0)
    self._real_pruning_segmented_segments = segments
    self._real_pruning_segmented_free_blocks = (
        [(next_offset, capacity - next_offset)] if next_offset < capacity else []
    )
    self._real_pruning_segmented_metadata_cache = {}

def _segmented_flat_alloc_block(self, length: int) -> int:
    length = int(length)
    if length <= 0:
        return 0
    free_blocks = list(getattr(self, "_real_pruning_segmented_free_blocks", []))
    for index, (offset, block_length) in enumerate(free_blocks):
        offset = int(offset)
        block_length = int(block_length)
        if block_length < length:
            continue
        alloc_offset = offset
        if block_length == length:
            del free_blocks[index]
        else:
            free_blocks[index] = (offset + length, block_length - length)
        self._real_pruning_segmented_free_blocks = free_blocks
        return alloc_offset

    _segmented_flat_stat_add("alloc_defrag_attempts")
    _segmented_flat_defragment(self)
    free_blocks = list(getattr(self, "_real_pruning_segmented_free_blocks", []))
    for index, (offset, block_length) in enumerate(free_blocks):
        offset = int(offset)
        block_length = int(block_length)
        if block_length < length:
            continue
        alloc_offset = offset
        if block_length == length:
            del free_blocks[index]
        else:
            free_blocks[index] = (offset + length, block_length - length)
        self._real_pruning_segmented_free_blocks = free_blocks
        return alloc_offset
    raise RuntimeError("segmented flat cache capacity exhausted")

def _segmented_flat_try_alloc_block_no_defrag(self, length: int) -> Optional[int]:
    length = int(length)
    if length <= 0:
        return 0
    free_blocks = list(getattr(self, "_real_pruning_segmented_free_blocks", []))
    for index, (offset, block_length) in enumerate(free_blocks):
        offset = int(offset)
        block_length = int(block_length)
        if block_length < length:
            continue
        alloc_offset = offset
        if block_length == length:
            del free_blocks[index]
        else:
            free_blocks[index] = (offset + length, block_length - length)
        self._real_pruning_segmented_free_blocks = free_blocks
        return alloc_offset
    return None

def _segmented_flat_try_alloc_head_blocks_no_defrag(
    self,
    *,
    head_count: int,
    tokens_per_head: int,
) -> Optional[list[tuple[int, int]]]:
    head_count = int(head_count)
    tokens_per_head = int(tokens_per_head)
    if head_count <= 0:
        return []
    if tokens_per_head <= 0:
        return [(0, head_count)]

    free_blocks = [
        (int(offset), int(length))
        for offset, length in getattr(self, "_real_pruning_segmented_free_blocks", [])
        if int(length) > 0
    ]
    allocations: list[tuple[int, int]] = []
    next_free_blocks: list[tuple[int, int]] = []
    remaining_heads = head_count
    for offset, block_length in free_blocks:
        if remaining_heads > 0:
            alloc_heads = min(remaining_heads, block_length // tokens_per_head)
            if alloc_heads > 0:
                allocations.append((offset, alloc_heads))
                alloc_length = alloc_heads * tokens_per_head
                offset += alloc_length
                block_length -= alloc_length
                remaining_heads -= alloc_heads
        if block_length > 0:
            next_free_blocks.append((offset, block_length))

    if remaining_heads > 0:
        return None

    self._real_pruning_segmented_free_blocks = _segmented_flat_merge_free_blocks(
        next_free_blocks
    )
    return allocations

def _segmented_flat_append_segment(
    self,
    *,
    source_scale_idx: int,
    head_idx: int,
    values_k: torch.Tensor,
    values_v: torch.Tensor,
):
    length = int(values_k.shape[1])
    if length <= 0:
        return
    _segmented_flat_stat_add("per_head_append_calls")
    _segmented_flat_record_kv_copy(
        "per_head_append_copy",
        slots=length,
        k_tensor=values_k,
        v_tensor=values_v,
    )
    offset = _segmented_flat_alloc_block(self, length)
    self.cached_k[:, offset : offset + length, :] = values_k
    self.cached_v[:, offset : offset + length, :] = values_v
    self._real_pruning_segmented_segments.append(
        {
            "source_scale": int(source_scale_idx),
            "head": int(head_idx),
            "offset": int(offset),
            "length": int(length),
        }
    )
    self._real_pruning_segmented_metadata_cache = {}

def _segmented_flat_append_current_grouped(
    self,
    *,
    source_scale_idx: int,
    kept_heads: list[int],
    current_k: torch.Tensor,
    current_v: torch.Tensor,
) -> bool:
    if not kept_heads:
        return True
    _segmented_flat_stat_add("grouped_append_attempts")
    current_len = int(current_k.shape[2])
    if current_len <= 0:
        return True
    total_length = len(kept_heads) * current_len
    offset = _segmented_flat_try_alloc_block_no_defrag(self, total_length)
    if offset is not None:
        allocations = [(offset, len(kept_heads))]
    else:
        _segmented_flat_stat_add("grouped_append_single_block_misses")
        allocations = _segmented_flat_try_alloc_head_blocks_no_defrag(
            self,
            head_count=len(kept_heads),
            tokens_per_head=current_len,
        )
        if allocations is None:
            _segmented_flat_stat_add("grouped_append_fallbacks")
            return False
        _segmented_flat_stat_add("grouped_append_multiblock_successes")

    _segmented_flat_stat_add("grouped_append_successes")
    _segmented_flat_stat_add("grouped_append_heads", len(kept_heads))
    _segmented_flat_stat_add("grouped_append_blocks", len(allocations))
    _segmented_flat_record_kv_copy(
        "grouped_append_copy",
        slots=total_length,
        k_tensor=current_k,
        v_tensor=current_v,
    )
    batch_size, num_heads, _, head_dim = current_k.shape
    flat_k = current_k.reshape(batch_size, num_heads * current_len, head_dim)
    flat_v = current_v.reshape(batch_size, num_heads * current_len, head_dim)
    head_offset = 0
    index_elements = 0
    for offset, allocation_heads in allocations:
        chunk_heads = kept_heads[head_offset : head_offset + allocation_heads]
        chunk_length = int(allocation_heads) * current_len
        index_parts = [
            _segmented_flat_index_range(
                head_idx=head_idx,
                sequence_len=current_len,
                start=0,
                end=current_len,
                device=current_k.device,
            )
            for head_idx in chunk_heads
        ]
        current_indices = torch.cat(index_parts, dim=0)
        index_elements += int(current_indices.numel())
        torch.index_select(
            flat_k,
            1,
            current_indices,
            out=self.cached_k[:, offset : offset + chunk_length, :],
        )
        torch.index_select(
            flat_v,
            1,
            current_indices,
            out=self.cached_v[:, offset : offset + chunk_length, :],
        )
        for index, head_idx in enumerate(chunk_heads):
            self._real_pruning_segmented_segments.append(
                {
                    "source_scale": int(source_scale_idx),
                    "head": int(head_idx),
                    "offset": int(offset + index * current_len),
                    "length": int(current_len),
                }
            )
        head_offset += int(allocation_heads)
    _segmented_flat_stat_add("grouped_append_index_elements", index_elements)
    self._real_pruning_segmented_metadata_cache = {}
    return True

def _segmented_flat_index_range(
    *,
    head_idx: int,
    sequence_len: int,
    start: int,
    end: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.arange(
        int(head_idx) * int(sequence_len) + int(start),
        int(head_idx) * int(sequence_len) + int(end),
        device=device,
        dtype=torch.long,
    )

def _segmented_flat_activation_plan(
    *,
    num_heads: int,
    old_cache_len: int,
    current_len: int,
    current_scale_idx: Optional[int],
    logical_len: int,
    removed_heads_by_source_scale: Mapping[int, set[int]],
    scale_ranges: Mapping[int, tuple[int, int]],
    device: torch.device,
):
    segments = []
    old_index_parts = []
    current_index_parts = []
    old_slot_count = 0
    current_slot_count = 0
    write_offset = 0
    old_cache_len = int(old_cache_len)
    current_len = int(current_len)
    current_scale_idx = None if current_scale_idx is None else int(current_scale_idx)
    for source_scale_idx, scale_start, scale_end in _segmented_flat_scale_segments(
        logical_len,
        scale_ranges,
    ):
        removed_heads = removed_heads_by_source_scale.get(int(source_scale_idx), set())
        old_start = int(scale_start)
        old_end = min(int(scale_end), old_cache_len)
        current_start = max(int(scale_start), old_cache_len) - old_cache_len
        current_end = min(int(scale_end), old_cache_len + current_len) - old_cache_len
        for head_idx in range(int(num_heads)):
            if head_idx in removed_heads:
                continue
            if old_start < old_end:
                length = old_end - old_start
                segments.append(
                    {
                        "source_scale": int(source_scale_idx),
                        "head": int(head_idx),
                        "offset": int(write_offset),
                        "length": int(length),
                    }
                )
                old_index_parts.append(
                    _segmented_flat_index_range(
                        head_idx=head_idx,
                        sequence_len=old_cache_len,
                        start=old_start,
                        end=old_end,
                        device=device,
                    )
                )
                write_offset += length
                old_slot_count += length
            if (
                current_scale_idx is not None
                and int(source_scale_idx) == current_scale_idx
                and current_start < current_end
            ):
                length = current_end - current_start
                segments.append(
                    {
                        "source_scale": int(source_scale_idx),
                        "head": int(head_idx),
                        "offset": int(write_offset),
                        "length": int(length),
                    }
                )
                current_index_parts.append(
                    _segmented_flat_index_range(
                        head_idx=head_idx,
                        sequence_len=current_len,
                        start=current_start,
                        end=current_end,
                        device=device,
                    )
                )
                write_offset += length
                current_slot_count += length
    old_indices = (
        torch.cat(old_index_parts, dim=0)
        if old_index_parts
        else torch.empty((0,), device=device, dtype=torch.long)
    )
    current_indices = (
        torch.cat(current_index_parts, dim=0)
        if current_index_parts
        else torch.empty((0,), device=device, dtype=torch.long)
    )
    return segments, old_indices, current_indices, old_slot_count, current_slot_count

def _segmented_flat_pack_activation(
    self,
    *,
    persisted_k: Optional[torch.Tensor],
    persisted_v: Optional[torch.Tensor],
    current_k: Optional[torch.Tensor],
    current_v: Optional[torch.Tensor],
    old_indices: torch.Tensor,
    current_indices: torch.Tensor,
    old_slot_count: int,
    current_slot_count: int,
):
    _segmented_flat_stat_add("activation_pack_calls")
    write_offset = 0
    old_slot_count = int(old_slot_count)
    current_slot_count = int(current_slot_count)
    if old_slot_count > 0:
        if persisted_k is None or persisted_v is None:
            raise RuntimeError("segmented flat activation missing persisted cache")
        batch_size, num_heads, old_cache_len, head_dim = persisted_k.shape
        flat_k = persisted_k.reshape(batch_size, num_heads * old_cache_len, head_dim)
        flat_v = persisted_v.reshape(batch_size, num_heads * old_cache_len, head_dim)
        _segmented_flat_record_kv_copy(
            "activation_old_copy",
            slots=old_slot_count,
            k_tensor=persisted_k,
            v_tensor=persisted_v,
        )
        _segmented_flat_stat_add("activation_old_index_elements", int(old_indices.numel()))
        torch.index_select(
            flat_k,
            1,
            old_indices,
            out=self.cached_k[:, write_offset : write_offset + old_slot_count, :],
        )
        torch.index_select(
            flat_v,
            1,
            old_indices,
            out=self.cached_v[:, write_offset : write_offset + old_slot_count, :],
        )
        write_offset += old_slot_count
    if current_slot_count > 0:
        if current_k is None or current_v is None:
            raise RuntimeError("segmented flat activation missing current cache")
        batch_size, num_heads, current_len, head_dim = current_k.shape
        flat_k = current_k.reshape(batch_size, num_heads * current_len, head_dim)
        flat_v = current_v.reshape(batch_size, num_heads * current_len, head_dim)
        _segmented_flat_record_kv_copy(
            "activation_current_copy",
            slots=current_slot_count,
            k_tensor=current_k,
            v_tensor=current_v,
        )
        _segmented_flat_stat_add(
            "activation_current_index_elements",
            int(current_indices.numel()),
        )
        torch.index_select(
            flat_k,
            1,
            current_indices,
            out=self.cached_k[:, write_offset : write_offset + current_slot_count, :],
        )
        torch.index_select(
            flat_v,
            1,
            current_indices,
            out=self.cached_v[:, write_offset : write_offset + current_slot_count, :],
        )

def _segmented_flat_build_attention_metadata(
    self,
    *,
    device: torch.device,
    target_scale_idx: Optional[int] = None,
):
    if target_scale_idx is not None:
        static_metadata = _segmented_flat_static_attention_metadata(
            self,
            target_scale_idx=target_scale_idx,
            device=device,
        )
        if static_metadata is not None:
            return static_metadata

    _segmented_flat_stat_add("dynamic_metadata_build_calls")
    segments = getattr(self, "_real_pruning_segmented_segments", [])
    per_head = [[] for _ in range(self.num_heads)]
    for segment in segments:
        per_head[int(segment["head"])].append(segment)
    max_segments = max((len(items) for items in per_head), default=0)
    starts_cpu = torch.zeros((self.num_heads, max_segments), dtype=torch.int32)
    ends_cpu = torch.zeros((self.num_heads, max_segments), dtype=torch.int32)
    counts_cpu = torch.zeros((self.num_heads,), dtype=torch.int32)
    for head_idx, items in enumerate(per_head):
        items = sorted(items, key=lambda item: int(item["source_scale"]))
        counts_cpu[head_idx] = len(items)
        for segment_idx, segment in enumerate(items):
            offset = int(segment["offset"])
            length = int(segment["length"])
            starts_cpu[head_idx, segment_idx] = offset
            ends_cpu[head_idx, segment_idx] = offset + length
    metadata = (
        starts_cpu.to(device=device),
        ends_cpu.to(device=device),
        counts_cpu.to(device=device),
    )
    _segmented_flat_stat_add("dynamic_metadata_segments", len(segments))
    _segmented_flat_stat_add("dynamic_metadata_max_segments_sum", max_segments)
    _segmented_flat_record_metadata("dynamic_metadata_device_copy", metadata)
    return metadata

def _segmented_flat_scale_segments(logical_len: int, scale_ranges: Mapping[int, tuple[int, int]]):
    logical_len = int(logical_len)
    for source_scale_idx, scale_range in sorted(scale_ranges.items()):
        scale_start, scale_end = scale_range
        scale_start = int(scale_start)
        scale_end = min(int(scale_end), logical_len)
        if scale_start < scale_end:
            yield int(source_scale_idx), scale_start, scale_end

def _segmented_flat_activate_from_dense_old(
    self,
    *,
    persisted_k: torch.Tensor,
    persisted_v: torch.Tensor,
    old_cache_len: int,
    target_scale_idx: int,
    removed_heads_by_source_scale: Mapping[int, set[int]],
):
    batch_size, num_heads, _, head_dim = persisted_k.shape
    _dense_prefix_reset_runtime_state(self)
    _segmented_flat_ensure_storage(
        self,
        batch_size=batch_size,
        head_dim=head_dim,
        device=persisted_k.device,
        k_dtype=persisted_k.dtype,
        v_dtype=persisted_v.dtype,
    )
    self._real_pruning_segmented_cache_active = True
    if _segmented_flat_static_pack_state(
        self,
        target_scale_idx=target_scale_idx,
        persisted_k=persisted_k,
        persisted_v=persisted_v,
        current_k=None,
        current_v=None,
        old_cache_len=old_cache_len,
        current_scale_idx=None,
    ):
        self._real_pruning_segmented_segments = []
        self._real_pruning_segmented_free_blocks = []
        self._real_pruning_segmented_metadata_cache = {}
        return

    self._real_pruning_segmented_segments = []
    self._real_pruning_segmented_free_blocks = [
        (0, int(getattr(self, "_real_pruning_segmented_capacity", 0) or 0))
    ]
    self._real_pruning_segmented_metadata_cache = {}
    (
        segments,
        old_indices,
        current_indices,
        old_slot_count,
        current_slot_count,
    ) = _segmented_flat_activation_plan(
        num_heads=num_heads,
        old_cache_len=old_cache_len,
        current_len=0,
        current_scale_idx=None,
        logical_len=old_cache_len,
        removed_heads_by_source_scale=removed_heads_by_source_scale,
        scale_ranges=self._real_pruning_scale_specific_controller.scale_ranges,
        device=persisted_k.device,
    )
    _segmented_flat_pack_activation(
        self,
        persisted_k=persisted_k,
        persisted_v=persisted_v,
        current_k=None,
        current_v=None,
        old_indices=old_indices,
        current_indices=current_indices,
        old_slot_count=old_slot_count,
        current_slot_count=current_slot_count,
    )
    self._real_pruning_segmented_segments = segments
    used_slots = int(old_slot_count) + int(current_slot_count)
    capacity = int(getattr(self, "_real_pruning_segmented_capacity", 0) or 0)
    self._real_pruning_segmented_free_blocks = (
        [(used_slots, capacity - used_slots)] if used_slots < capacity else []
    )

def _segmented_flat_activate_from_dense_and_current(
    self,
    *,
    persisted_k: Optional[torch.Tensor],
    persisted_v: Optional[torch.Tensor],
    current_k: torch.Tensor,
    current_v: torch.Tensor,
    current_scale_idx: int,
    old_cache_len: int,
    logical_len: int,
    target_scale_idx: int,
    removed_heads_by_source_scale: Mapping[int, set[int]],
):
    batch_size, num_heads, current_len, head_dim = current_k.shape
    _dense_prefix_reset_runtime_state(self)
    _segmented_flat_ensure_storage(
        self,
        batch_size=batch_size,
        head_dim=head_dim,
        device=current_k.device,
        k_dtype=current_k.dtype,
        v_dtype=current_v.dtype,
    )
    self._real_pruning_segmented_cache_active = True
    if _segmented_flat_static_pack_state(
        self,
        target_scale_idx=target_scale_idx,
        persisted_k=persisted_k,
        persisted_v=persisted_v,
        current_k=current_k,
        current_v=current_v,
        old_cache_len=old_cache_len,
        current_scale_idx=current_scale_idx,
    ):
        self._real_pruning_segmented_segments = []
        self._real_pruning_segmented_free_blocks = []
        self._real_pruning_segmented_metadata_cache = {}
        return

    self._real_pruning_segmented_segments = []
    self._real_pruning_segmented_free_blocks = [
        (0, int(getattr(self, "_real_pruning_segmented_capacity", 0) or 0))
    ]
    self._real_pruning_segmented_metadata_cache = {}
    (
        segments,
        old_indices,
        current_indices,
        old_slot_count,
        current_slot_count,
    ) = _segmented_flat_activation_plan(
        num_heads=num_heads,
        old_cache_len=old_cache_len,
        current_len=current_len,
        current_scale_idx=current_scale_idx,
        logical_len=logical_len,
        removed_heads_by_source_scale=removed_heads_by_source_scale,
        scale_ranges=self._real_pruning_scale_specific_controller.scale_ranges,
        device=current_k.device,
    )
    _segmented_flat_pack_activation(
        self,
        persisted_k=persisted_k,
        persisted_v=persisted_v,
        current_k=current_k,
        current_v=current_v,
        old_indices=old_indices,
        current_indices=current_indices,
        old_slot_count=old_slot_count,
        current_slot_count=current_slot_count,
    )
    self._real_pruning_segmented_segments = segments
    used_slots = int(old_slot_count) + int(current_slot_count)
    capacity = int(getattr(self, "_real_pruning_segmented_capacity", 0) or 0)
    self._real_pruning_segmented_free_blocks = (
        [(used_slots, capacity - used_slots)] if used_slots < capacity else []
    )

def _segmented_flat_update_for_next_step(
    self,
    *,
    current_k: torch.Tensor,
    current_v: torch.Tensor,
    current_scale_idx: int,
    next_target_scale_idx: int,
    next_removed_heads_by_source_scale: Mapping[int, set[int]],
):
    if _segmented_flat_static_append_current(
        self,
        current_k=current_k,
        current_v=current_v,
        current_scale_idx=current_scale_idx,
        next_target_scale_idx=next_target_scale_idx,
    ):
        return

    kept_segments = []
    for segment in getattr(self, "_real_pruning_segmented_segments", []):
        source_scale_idx = int(segment["source_scale"])
        head_idx = int(segment["head"])
        if head_idx in next_removed_heads_by_source_scale.get(source_scale_idx, set()):
            _segmented_flat_free_block(
                self,
                int(segment["offset"]),
                int(segment["length"]),
            )
            continue
        kept_segments.append(segment)
    self._real_pruning_segmented_segments = kept_segments

    removed_current_heads = next_removed_heads_by_source_scale.get(
        int(current_scale_idx),
        set(),
    )
    kept_current_heads = [
        int(head_idx)
        for head_idx in range(current_k.shape[1])
        if head_idx not in removed_current_heads
    ]
    if _segmented_flat_append_current_grouped(
        self,
        source_scale_idx=current_scale_idx,
        kept_heads=kept_current_heads,
        current_k=current_k,
        current_v=current_v,
    ):
        return
    for head_idx in kept_current_heads:
        _segmented_flat_append_segment(
            self,
            source_scale_idx=current_scale_idx,
            head_idx=head_idx,
            values_k=current_k[:, head_idx, :, :],
            values_v=current_v[:, head_idx, :, :],
        )

def _dense_prefix_cache_for_next_step(
    self,
    *,
    persisted_k: Optional[torch.Tensor],
    persisted_v: Optional[torch.Tensor],
    current_k: torch.Tensor,
    current_v: torch.Tensor,
    old_cache_len: int,
    logical_len: int,
):
    batch_size, num_heads, current_len, head_dim = current_k.shape
    old_cache_len = int(old_cache_len)
    logical_len = int(logical_len)

    max_cache_len = 0
    if _dense_prefix_prealloc_enabled():
        max_cache_len = _dense_prefix_cache_capacity_for_layer(
            self._real_pruning_scale_specific_controller,
            layer_idx=self.layer_idx,
        )

    if max_cache_len >= logical_len and max_cache_len > 0:
        prefix_k = getattr(self, "_real_pruning_dense_prefix_k", None)
        prefix_v = getattr(self, "_real_pruning_dense_prefix_v", None)
        needs_allocate = (
            prefix_k is None
            or prefix_v is None
            or tuple(prefix_k.shape) != (batch_size, num_heads, max_cache_len, head_dim)
            or tuple(prefix_v.shape) != (batch_size, num_heads, max_cache_len, head_dim)
            or prefix_k.device != current_k.device
            or prefix_v.device != current_v.device
            or prefix_k.dtype != current_k.dtype
            or prefix_v.dtype != current_v.dtype
        )
        if needs_allocate:
            prefix_k = current_k.new_empty((batch_size, num_heads, max_cache_len, head_dim))
            prefix_v = current_v.new_empty((batch_size, num_heads, max_cache_len, head_dim))
            self._real_pruning_dense_prefix_k = prefix_k
            self._real_pruning_dense_prefix_v = prefix_v
            self._real_pruning_dense_prefix_capacity = int(max_cache_len)
            _segmented_flat_stat_add("dense_prefix_prealloc_allocations")
            _segmented_flat_record_kv_copy(
                "dense_prefix_prealloc_capacity",
                slots=int(num_heads) * int(max_cache_len),
                k_tensor=prefix_k,
                v_tensor=prefix_v,
            )
            if old_cache_len > 0:
                if persisted_k is None or persisted_v is None:
                    raise RuntimeError("dense prefix prealloc missing persisted cache")
                prefix_k[:, :, :old_cache_len, :] = persisted_k[:, :, :old_cache_len, :]
                prefix_v[:, :, :old_cache_len, :] = persisted_v[:, :, :old_cache_len, :]
                _segmented_flat_record_kv_copy(
                    "dense_prefix_prealloc_old_copy",
                    slots=int(num_heads) * int(old_cache_len),
                    k_tensor=prefix_k,
                    v_tensor=prefix_v,
                )

        prefix_k[:, :, old_cache_len:logical_len, :] = current_k
        prefix_v[:, :, old_cache_len:logical_len, :] = current_v
        _segmented_flat_stat_add("dense_prefix_prealloc_append_calls")
        _segmented_flat_record_kv_copy(
            "dense_prefix_prealloc_append_copy",
            slots=int(num_heads) * int(current_len),
            k_tensor=current_k,
            v_tensor=current_v,
        )
        next_k = prefix_k[:, :, :logical_len, :]
        next_v = prefix_v[:, :, :logical_len, :]
    elif persisted_k is None or persisted_v is None or persisted_k.shape[2] == 0:
        _dense_prefix_reset_runtime_state(self)
        next_k = current_k.contiguous()
        next_v = current_v.contiguous()
        _segmented_flat_stat_add("dense_prefix_init_calls")
        _segmented_flat_record_kv_copy(
            "dense_prefix_init_copy",
            slots=int(current_k.shape[1]) * int(current_k.shape[2]),
            k_tensor=current_k,
            v_tensor=current_v,
        )
    else:
        _dense_prefix_reset_runtime_state(self)
        next_k = torch.cat((persisted_k, current_k), dim=2)
        next_v = torch.cat((persisted_v, current_v), dim=2)
        _segmented_flat_stat_add("dense_prefix_cat_calls")
        _segmented_flat_record_kv_copy(
            "dense_prefix_cat_output",
            slots=int(next_k.shape[1]) * int(next_k.shape[2]),
            k_tensor=next_k,
            v_tensor=next_v,
        )
    next_lengths = torch.full(
        (num_heads,),
        int(logical_len),
        device=current_k.device,
        dtype=torch.long,
    )
    empty_metadata = torch.empty((num_heads, 0), device=current_k.device, dtype=torch.long)
    return next_k, next_v, next_lengths, empty_metadata, empty_metadata


def _real_pruning_scale_specific_fast_forward(
    self,
    x,
    attn_bias_or_two_vector=None,
    attn_fn=None,
    scale_schedule=None,
    rope2d_freqs_grid=None,
    scale_ind=0,
):
    if not self.caching:
        raise NotImplementedError("real pruning fused segmented only supports kv-cached inference")
    if attn_bias_or_two_vector is not None:
        raise NotImplementedError("real pruning fused segmented does not support training attention masks")
    if attn_fn is not None:
        raise NotImplementedError("real pruning fused segmented does not support custom attention functions")

    batch_size, seq_len, channels = x.shape
    final_scale = self._num_scales == scale_ind + 1
    old_cache_len = int(getattr(self, "_real_pruning_logical_len", 0))
    logical_len = old_cache_len + seq_len

    if (
        _fused_segmented_baseline_prefix_enabled()
        and not bool(getattr(self, "_real_pruning_segmented_cache_active", False))
    ):
        current_removed_heads_by_source_scale = (
            self._real_pruning_scale_specific_controller.removed_heads_by_source_for(
                self.layer_idx,
                scale_ind,
            )
        )
        if not _has_any_removed_heads(current_removed_heads_by_source_scale):
            output = self._pre_real_prune_scale_specific_fast_forward(
                x,
                attn_bias_or_two_vector=attn_bias_or_two_vector,
                attn_fn=attn_fn,
                scale_schedule=scale_schedule,
                rope2d_freqs_grid=rope2d_freqs_grid,
                scale_ind=scale_ind,
            )
            self._real_pruning_logical_len = logical_len
            self._real_pruning_fast_lengths = None
            self._real_pruning_fast_source_scale_ids = None
            self._real_pruning_fast_logical_positions = None
            _dense_prefix_reset_runtime_state(self)

            if final_scale:
                _segmented_flat_reset_runtime_state(self)
                (
                    self.cached_k,
                    self.cached_v,
                    self._real_pruning_fast_lengths,
                    self._real_pruning_fast_source_scale_ids,
                    self._real_pruning_fast_logical_positions,
                ) = _empty_fast_cache_state(
                    batch_size=batch_size,
                    num_heads=self.num_heads,
                    head_dim=self.head_dim,
                    device=x.device,
                    dtype=x.dtype,
                )
                return output

            next_removed_heads_by_source_scale = (
                self._real_pruning_scale_specific_controller.removed_heads_by_source_for(
                    self.layer_idx,
                    scale_ind + 1,
                )
            )
            if _has_any_removed_heads(next_removed_heads_by_source_scale):
                token = _phase_profile_start("compact", layer_idx=self.layer_idx, scale_ind=scale_ind)
                try:
                    _segmented_flat_activate_from_dense_old(
                        self,
                        persisted_k=self.cached_k,
                        persisted_v=self.cached_v,
                        old_cache_len=logical_len,
                        target_scale_idx=scale_ind + 1,
                        removed_heads_by_source_scale=next_removed_heads_by_source_scale,
                    )
                    (
                        self._real_pruning_fast_lengths,
                        self._real_pruning_fast_source_scale_ids,
                        self._real_pruning_fast_logical_positions,
                    ) = _segmented_flat_empty_metadata(self.num_heads, x.device)
                finally:
                    _phase_profile_end(token)
            return output

    qkv = F.linear(
        input=x,
        weight=self.mat_qkv.weight,
        bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias)),
    ).view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
    q, k, v = qkv.unbind(dim=2)
    q = q.permute(0, 2, 1, 3).contiguous()
    k = k.permute(0, 2, 1, 3).contiguous()
    v = v.permute(0, 2, 1, 3).contiguous()
    del qkv

    if self.cos_attn:
        scale_mul = (
            self.scale_mul_1H11.reshape(1, self.num_heads, 1, 1)
            .clamp_max(self.max_scale_mul)
            .exp()
        )
        q = F.normalize(q, dim=-1, eps=1e-12).mul(scale_mul).contiguous()
        k = F.normalize(k, dim=-1, eps=1e-12).contiguous()
        v = v.contiguous()

    if rope2d_freqs_grid is not None:
        if scale_schedule is None:
            raise ValueError("scale_schedule is required when rope2d_freqs_grid is provided")
        q, k = apply_rotary_emb(
            q,
            k,
            scale_schedule,
            rope2d_freqs_grid,
            self.pad_to_multiplier,
            self.rope2d_normalized_by_hw,
            scale_ind,
        )

    current_removed_heads_by_source_scale = (
        self._real_pruning_scale_specific_controller.removed_heads_by_source_for(
            self.layer_idx,
            scale_ind,
        )
    )
    segmented_active = bool(getattr(self, "_real_pruning_segmented_cache_active", False))
    dense_prefix_cache_candidate = None

    if old_cache_len == 0:
        token = _phase_profile_start("attention", layer_idx=self.layer_idx, scale_ind=scale_ind)
        try:
            output = slow_attn(
                query=q,
                key=k,
                value=v,
                scale=self.scale,
                dropout_p=0.0,
            ).transpose(1, 2).reshape(batch_size, seq_len, channels)
        finally:
            _phase_profile_end(token)
    elif not segmented_active and not _has_any_removed_heads(current_removed_heads_by_source_scale):
        token = _phase_profile_start("materialize", layer_idx=self.layer_idx, scale_ind=scale_ind)
        try:
            attn_k = torch.cat((self.cached_k, k), dim=2)
            attn_v = torch.cat((self.cached_v, v), dim=2)
            dense_prefix_cache_candidate = (attn_k, attn_v)
            _segmented_flat_stat_add("dense_attention_cat_calls")
            _segmented_flat_record_kv_copy(
                "dense_attention_cat_output",
                slots=int(attn_k.shape[1]) * int(attn_k.shape[2]),
                k_tensor=attn_k,
                v_tensor=attn_v,
            )
        finally:
            _phase_profile_end(token)

        token = _phase_profile_start("attention", layer_idx=self.layer_idx, scale_ind=scale_ind)
        try:
            output = slow_attn(
                query=q,
                key=attn_k,
                value=attn_v,
                scale=self.scale,
                dropout_p=0.0,
            ).transpose(1, 2).reshape(batch_size, seq_len, channels)
        finally:
            _phase_profile_end(token)
    else:
        token = _phase_profile_start("materialize", layer_idx=self.layer_idx, scale_ind=scale_ind)
        try:
            if not segmented_active:
                _segmented_flat_activate_from_dense_old(
                    self,
                    persisted_k=self.cached_k,
                    persisted_v=self.cached_v,
                    old_cache_len=old_cache_len,
                    target_scale_idx=scale_ind,
                    removed_heads_by_source_scale=current_removed_heads_by_source_scale,
                )
            segment_starts, segment_ends, segment_counts = _segmented_flat_build_attention_metadata(
                self,
                device=k.device,
                target_scale_idx=scale_ind,
            )
        finally:
            _phase_profile_end(token)

        token = _phase_profile_start("attention", layer_idx=self.layer_idx, scale_ind=scale_ind)
        try:
            output = fused_real_pruning_flat_segmented_attention(
                q=q,
                persisted_k=self.cached_k,
                persisted_v=self.cached_v,
                segment_starts=segment_starts,
                segment_ends=segment_ends,
                segment_counts=segment_counts,
                current_k=k,
                current_v=v,
                softmax_scale=self.scale,
                softmax_mode=_fused_softmax_mode(),
            )
        finally:
            _phase_profile_end(token)

    del q
    if final_scale:
        cache_batch_size = int(k.shape[0])
        cache_head_dim = int(k.shape[-1])
        cache_device = k.device
        cache_dtype = k.dtype
        del k, v
    else:
        cache_batch_size = int(k.shape[0])
        cache_head_dim = int(k.shape[-1])
        cache_device = k.device
        cache_dtype = k.dtype

    output = self.proj_drop(self.proj(output))

    token = _phase_profile_start("compact", layer_idx=self.layer_idx, scale_ind=scale_ind)
    try:
        if final_scale:
            _dense_prefix_reset_runtime_state(self)
            _segmented_flat_reset_runtime_state(self)
            cached_k, cached_v, cached_lengths, cached_source_scale_ids, cached_logical_positions = _empty_fast_cache_state(
                batch_size=cache_batch_size,
                num_heads=self.num_heads,
                head_dim=cache_head_dim,
                device=cache_device,
                dtype=cache_dtype,
            )
        else:
            next_removed_heads_by_source_scale = (
                self._real_pruning_scale_specific_controller.removed_heads_by_source_for(
                    self.layer_idx,
                    scale_ind + 1,
                )
            )
            segmented_active = bool(getattr(self, "_real_pruning_segmented_cache_active", False))
            if not segmented_active and not _has_any_removed_heads(next_removed_heads_by_source_scale):
                if dense_prefix_cache_candidate is not None:
                    cached_k, cached_v = dense_prefix_cache_candidate
                    _dense_prefix_reset_runtime_state(self)
                    cached_lengths = torch.full(
                        (self.num_heads,),
                        int(logical_len),
                        device=cache_device,
                        dtype=torch.long,
                    )
                    cached_source_scale_ids = torch.empty((self.num_heads, 0), device=cache_device, dtype=torch.long)
                    cached_logical_positions = torch.empty((self.num_heads, 0), device=cache_device, dtype=torch.long)
                else:
                    cached_k, cached_v, cached_lengths, cached_source_scale_ids, cached_logical_positions = _dense_prefix_cache_for_next_step(
                        self,
                        persisted_k=self.cached_k,
                        persisted_v=self.cached_v,
                        current_k=k,
                        current_v=v,
                        old_cache_len=old_cache_len,
                        logical_len=logical_len,
                    )
            else:
                if segmented_active:
                    _segmented_flat_update_for_next_step(
                        self,
                        current_k=k,
                        current_v=v,
                        current_scale_idx=scale_ind,
                        next_target_scale_idx=scale_ind + 1,
                        next_removed_heads_by_source_scale=next_removed_heads_by_source_scale,
                    )
                else:
                    _segmented_flat_activate_from_dense_and_current(
                        self,
                        persisted_k=self.cached_k,
                        persisted_v=self.cached_v,
                        current_k=k,
                        current_v=v,
                        current_scale_idx=scale_ind,
                        old_cache_len=old_cache_len,
                        logical_len=logical_len,
                        target_scale_idx=scale_ind + 1,
                        removed_heads_by_source_scale=next_removed_heads_by_source_scale,
                    )
                cached_k = self.cached_k
                cached_v = self.cached_v
                cached_lengths, cached_source_scale_ids, cached_logical_positions = _segmented_flat_empty_metadata(
                    self.num_heads,
                    cache_device,
                )
    finally:
        _phase_profile_end(token)

    self.cached_k = cached_k
    self.cached_v = cached_v
    self._real_pruning_fast_lengths = cached_lengths
    self._real_pruning_fast_source_scale_ids = cached_source_scale_ids
    self._real_pruning_fast_logical_positions = cached_logical_positions
    self._real_pruning_logical_len = logical_len
    return output


def _real_pruning_scale_specific_fast_kv_caching(self, enable: bool):
    self._pre_real_prune_scale_specific_fast_kv_caching(enable)
    self._real_pruning_scale_specific_controller.reset()
    self._real_pruning_fast_lengths = None
    self._real_pruning_fast_source_scale_ids = None
    self._real_pruning_fast_logical_positions = None
    self._real_pruning_logical_len = 0
    self._real_pruning_static_source_segments_cache = {}
    _dense_prefix_reset_runtime_state(self)
    _segmented_flat_reset_runtime_state(self)
    if enable:
        schedule = _segmented_flat_static_schedule(self)
        if schedule is not None:
            try:
                module_device = next(self.parameters()).device
            except StopIteration:
                module_device = torch.device("cpu")
            if torch.device(module_device).type == "cuda":
                _segmented_flat_prefill_static_metadata_device_cache(schedule, device=module_device)
                _segmented_flat_prefill_static_copy_device_cache(schedule, device=module_device)


def enable_real_pruning_scale_specific_fast(
    model,
    budget,
    scale_specific_removal_orders,
    *,
    protected_prefix_scales=4,
    source_scale_start=None,
    skip_last_scale=True,
    merge_mode="exact",
    attention_impl="fused_segmented",
    requested_generation_batch_size=None,
):
    merge_mode = _validate_merge_mode(merge_mode)
    if attention_impl != "fused_segmented":
        raise ValueError(f"unsupported real pruning fast attention_impl: {attention_impl!r}")
    modules = [module for module in model.modules() if isinstance(module, SelfAttention)]
    if not modules:
        return model

    _assign_layer_indices(modules)

    num_heads = modules[0].num_heads
    if any(module.num_heads != num_heads for module in modules):
        raise ValueError("real pruning fast requires all SelfAttention modules to use the same num_heads")

    protected_prefix_scales = int(protected_prefix_scales)
    if protected_prefix_scales < 0:
        raise ValueError("protected_prefix_scales must be non-negative")
    if source_scale_start is None:
        source_scale_start = protected_prefix_scales
    source_scale_start = int(source_scale_start)

    scale_schedule = get_scale_schedule()
    if len(scale_schedule) <= protected_prefix_scales:
        raise ValueError("scale-specific real pruning fast requires at least one scale after the protected prefix")

    normalized_orders = _validate_and_normalize_scale_specific_removal_orders(
        scale_specific_removal_orders,
        modules,
        source_scale_start=source_scale_start,
    )
    strict_view = strict_removed_heads_by_scale_specific_for_budget(
        budget,
        normalized_orders,
        num_layers=len(modules),
        num_heads=num_heads,
        source_scale_start=source_scale_start,
        protected_prefix_scales=protected_prefix_scales,
        skip_last_scale=skip_last_scale,
    )
    normalized_removed_heads_by_scale = _validate_and_normalize_removed_heads_by_scale_specific(
        strict_view["removed_heads_by_scale"],
        modules,
        source_scale_start=source_scale_start,
    )
    controller = ScaleSpecificPruningController(
        num_layers=len(modules),
        num_heads=num_heads,
        static_removed_heads_by_scale=normalized_removed_heads_by_scale,
        source_scale_start=source_scale_start,
    )

    for module in modules:
        module._num_scales = len(scale_schedule)
        module._real_pruning_scale_specific_controller = controller
        module._real_prune_scale_specific_budget = float(budget)
        module._real_prune_merge_mode = merge_mode
        module._real_pruning_fast_lengths = None
        module._real_pruning_fast_source_scale_ids = None
        module._real_pruning_fast_logical_positions = None
        module._real_pruning_logical_len = 0
        module._real_pruning_static_source_segments_cache = {}
        module._real_pruning_requested_generation_batch_size = (
            None if requested_generation_batch_size is None else int(requested_generation_batch_size)
        )
        if _segmented_flat_static_schedule_enabled():
            static_schedule = _segmented_flat_build_static_slot_schedule(
                controller,
                layer_idx=module.layer_idx,
                num_heads=module.num_heads,
            )
            try:
                module_device = next(module.parameters()).device
            except StopIteration:
                module_device = torch.device("cpu")
            _segmented_flat_prefill_static_metadata_device_cache(static_schedule, device=module_device)
            _segmented_flat_prefill_static_copy_device_cache(static_schedule, device=module_device)
            module._real_pruning_segmented_static_schedule = static_schedule
        else:
            module._real_pruning_segmented_static_schedule = None
        _segmented_flat_reset_runtime_state(module)
        module._real_pruning_fast_attention_impl = attention_impl
        if not hasattr(module, "_pre_real_prune_scale_specific_fast_forward"):
            module._pre_real_prune_scale_specific_fast_forward = module.forward
        module.forward = types.MethodType(_real_pruning_scale_specific_fast_forward, module)
        if not hasattr(module, "_pre_real_prune_scale_specific_fast_kv_caching"):
            module._pre_real_prune_scale_specific_fast_kv_caching = module.kv_caching
        module.kv_caching = types.MethodType(_real_pruning_scale_specific_fast_kv_caching, module)

    return model


def enable_real_pruning_scale_specific_fused_segmented(
    model,
    budget,
    scale_specific_removal_orders,
    *,
    protected_prefix_scales=4,
    source_scale_start=None,
    skip_last_scale=True,
    merge_mode="exact",
    requested_generation_batch_size=None,
):
    return enable_real_pruning_scale_specific_fast(
        model,
        budget,
        scale_specific_removal_orders,
        protected_prefix_scales=protected_prefix_scales,
        source_scale_start=source_scale_start,
        skip_last_scale=skip_last_scale,
        merge_mode=merge_mode,
        attention_impl="fused_segmented",
        requested_generation_batch_size=requested_generation_batch_size,
    )
