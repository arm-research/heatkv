# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT

import torch
import torch.nn.functional as F

from tools.utils import get_scale_schedule


def _assign_layer_indices(modules):
    for layer_idx, module in enumerate(modules):
        module.layer_idx = layer_idx


def _validate_and_normalize_removal_order(removal_order, modules):
    if not modules:
        raise ValueError("model does not contain any SelfAttention modules")

    num_layers = len(modules)
    expected_slots = sum(module.num_heads for module in modules)
    if len(removal_order) != expected_slots:
        raise ValueError(
            f"expected {expected_slots} removal-order entries, got {len(removal_order)}"
        )

    normalized_order = []
    seen = set()
    for idx, pair in enumerate(removal_order):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(f"removal_order[{idx}] must be a (head, layer) tuple")

        head, layer = int(pair[0]), int(pair[1])
        if not 0 <= layer < num_layers:
            raise ValueError(f"layer index out of range at position {idx}: {layer}")
        if not 0 <= head < modules[layer].num_heads:
            raise ValueError(f"head index out of range at position {idx}: {head}")

        key = (head, layer)
        if key in seen:
            raise ValueError(f"duplicate (head, layer) entry at position {idx}: {key}")

        seen.add(key)
        normalized_order.append(key)

    return normalized_order


def _empty_removed_heads(num_layers):
    return {layer_idx: set() for layer_idx in range(num_layers)}


def _clone_removed_heads(removed_heads_by_layer):
    return {
        layer_idx: set(heads)
        for layer_idx, heads in removed_heads_by_layer.items()
    }


def _build_removed_allow_mask(
    *,
    removed_heads,
    num_heads,
    batch_size,
    query_len,
    key_len,
    old_cache_len,
    retained_tokens,
    device,
):
    if not removed_heads:
        return None

    retained_tokens = min(retained_tokens, old_cache_len)
    if retained_tokens >= old_cache_len:
        return None

    removed_indices = torch.tensor(
        sorted(removed_heads),
        device=device,
        dtype=torch.long,
    )
    allow_mask = torch.ones(
        (batch_size, num_heads, query_len, key_len),
        dtype=torch.bool,
        device=device,
    )
    allow_mask[:, removed_indices, :, retained_tokens:old_cache_len] = False
    return allow_mask


def _validate_and_normalize_scale_specific_removal_orders(
    scale_specific_removal_orders,
    modules,
    *,
    source_scale_start=4,
    source_scale_end=11,
):
    expected_orders = source_scale_end - source_scale_start + 1
    if len(scale_specific_removal_orders) != expected_orders:
        raise ValueError(
            "scale_specific_removal_orders must contain exactly "
            f"{expected_orders} orders for source scales {source_scale_start}-{source_scale_end}"
        )

    normalized_orders = []
    for offset, removal_order in enumerate(scale_specific_removal_orders):
        scale_idx = source_scale_start + offset
        normalized_orders.append(
            _validate_and_normalize_removal_order(removal_order, modules)
        )

    return tuple(normalized_orders)


def _validate_and_normalize_removed_heads_by_scale_specific(
    removed_heads_by_scale,
    modules,
    *,
    source_scale_start=4,
    source_scale_end=11,
):
    if not modules:
        raise ValueError("model does not contain any SelfAttention modules")

    num_layers = len(modules)
    normalized = {}

    for target_scale_idx, source_map in removed_heads_by_scale.items():
        target_scale_idx = int(target_scale_idx)
        if not isinstance(source_map, dict):
            raise ValueError(
                f"removed_heads_by_scale[{target_scale_idx}] must be a dict"
            )

        normalized_source_map = {}
        for source_scale_idx in range(int(source_scale_start), int(source_scale_end) + 1):
            normalized_source_map[source_scale_idx] = _empty_removed_heads(num_layers)

        for source_scale_idx, layer_map in source_map.items():
            source_scale_idx = int(source_scale_idx)
            if not int(source_scale_start) <= source_scale_idx <= int(source_scale_end):
                raise ValueError(
                    f"removed_heads_by_scale[{target_scale_idx}] source scale out of range: "
                    f"{source_scale_idx}"
                )
            if not isinstance(layer_map, dict):
                raise ValueError(
                    f"removed_heads_by_scale[{target_scale_idx}][{source_scale_idx}] "
                    "must be a dict"
                )

            normalized_layer_map = _empty_removed_heads(num_layers)
            for layer_idx, heads in layer_map.items():
                layer_idx = int(layer_idx)
                if not 0 <= layer_idx < num_layers:
                    raise ValueError(
                        f"removed_heads_by_scale[{target_scale_idx}][{source_scale_idx}] "
                        f"layer index out of range: {layer_idx}"
                    )
                seen = set()
                for head_idx in heads:
                    head_idx = int(head_idx)
                    if not 0 <= head_idx < modules[layer_idx].num_heads:
                        raise ValueError(
                            f"removed_heads_by_scale[{target_scale_idx}][{source_scale_idx}] "
                            f"head index out of range for layer {layer_idx}: {head_idx}"
                        )
                    if head_idx in seen:
                        raise ValueError(
                            f"removed_heads_by_scale[{target_scale_idx}][{source_scale_idx}] "
                            f"duplicate head {head_idx} for layer {layer_idx}"
                        )
                    seen.add(head_idx)
                    normalized_layer_map[layer_idx].add(head_idx)
            normalized_source_map[source_scale_idx] = normalized_layer_map

        normalized[target_scale_idx] = normalized_source_map

    return normalized


def _scale_token_ranges(scale_schedule):
    ranges = {}
    cursor = 0
    for scale_idx, (_, height, width) in enumerate(scale_schedule):
        next_cursor = cursor + int(height) * int(width)
        ranges[scale_idx] = (cursor, next_cursor)
        cursor = next_cursor
    return ranges


def _attention_probs_from_allow_mask(scores, allow_mask):
    scores = scores.float()
    if allow_mask is None:
        return torch.softmax(scores, dim=-1)
    return torch.softmax(scores.masked_fill(~allow_mask, float("-inf")), dim=-1)


class PruningController:
    def __init__(
        self,
        *,
        num_layers,
        num_heads,
        retained_tokens,
        prune_counts,
        static_removed_heads_by_scale=None,
    ):
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.retained_tokens = int(retained_tokens)
        self.prune_counts = dict(prune_counts)
        self._static_removed_heads_by_scale = static_removed_heads_by_scale
        self._live_mode = static_removed_heads_by_scale is None
        self.reset()

    @property
    def live_mode(self):
        return self._live_mode

    def reset(self):
        self._recording_scale = None
        self._reported_layers = set()
        self._scale_scores = torch.zeros(
            (self.num_layers, self.num_heads),
            dtype=torch.float64,
        )
        self._removed_heads_by_scale = {0: _empty_removed_heads(self.num_layers)}

    def _removed_heads_state(self, scale_idx):
        if self._static_removed_heads_by_scale is not None:
            return self._static_removed_heads_by_scale.get(
                scale_idx,
                self._static_removed_heads_by_scale[max(self._static_removed_heads_by_scale)],
            )

        if scale_idx in self._removed_heads_by_scale:
            return self._removed_heads_by_scale[scale_idx]

        available_scales = [idx for idx in self._removed_heads_by_scale if idx <= scale_idx]
        if not available_scales:
            return _empty_removed_heads(self.num_layers)
        return self._removed_heads_by_scale[max(available_scales)]

    def removed_heads_for(self, layer_idx, scale_idx):
        return set(self._removed_heads_state(scale_idx).get(layer_idx, set()))

    def _finalize_live_scale(self, scale_idx):
        current_removed = _clone_removed_heads(self._removed_heads_state(scale_idx))
        next_scale_idx = scale_idx + 1
        target_removed = self.prune_counts.get(
            next_scale_idx,
            self.prune_counts.get(scale_idx, 0),
        )
        current_removed_count = sum(len(heads) for heads in current_removed.values())
        delta = max(0, target_removed - current_removed_count)

        if delta:
            candidates = []
            for layer_idx in range(self.num_layers):
                removed_heads = current_removed[layer_idx]
                for head_idx in range(self.num_heads):
                    if head_idx in removed_heads:
                        continue
                    candidates.append(
                        (
                            float(self._scale_scores[layer_idx, head_idx].item()),
                            layer_idx,
                            head_idx,
                        )
                    )
            candidates.sort(key=lambda item: (item[0], item[1], item[2]))
            for _, layer_idx, head_idx in candidates[:delta]:
                current_removed[layer_idx].add(head_idx)

        self._removed_heads_by_scale[next_scale_idx] = _clone_removed_heads(current_removed)
        self._scale_scores.zero_()
        self._reported_layers.clear()
        self._recording_scale = None

    def record_attention(
        self,
        layer_idx,
        scale_idx,
        *,
        q,
        k,
        attn_scale,
        old_cache_len,
    ):
        if not self.live_mode:
            return

        if self._recording_scale is None:
            self._recording_scale = scale_idx
        elif scale_idx != self._recording_scale:
            raise RuntimeError(
                f"live pruning expected scale {self._recording_scale}, got {scale_idx}"
            )

        if layer_idx in self._reported_layers:
            return

        allow_mask = _build_removed_allow_mask(
            removed_heads=self.removed_heads_for(layer_idx, scale_idx),
            num_heads=self.num_heads,
            batch_size=q.shape[0],
            query_len=q.shape[-2],
            key_len=k.shape[-2],
            old_cache_len=old_cache_len,
            retained_tokens=self.retained_tokens,
            device=k.device,
        )

        with torch.no_grad():
            scores = torch.matmul(q, k.transpose(-2, -1)) * attn_scale
            attn_probs = _attention_probs_from_allow_mask(scores, allow_mask)
            retained_tokens = min(self.retained_tokens, old_cache_len)
            if retained_tokens < old_cache_len:
                prunable_mass = attn_probs[..., retained_tokens:old_cache_len].sum(dim=(0, 2, 3))
                self._scale_scores[layer_idx].add_(prunable_mass.detach().cpu().to(torch.float64))

        self._reported_layers.add(layer_idx)
        if len(self._reported_layers) == self.num_layers:
            self._finalize_live_scale(scale_idx)


class ScaleSpecificPruningController:
    def __init__(
        self,
        *,
        num_layers,
        num_heads,
        static_removed_heads_by_scale=None,
        prune_counts=None,
        scale_specific_removal_orders=None,
        source_scale_start=4,
        source_scale_end=11,
    ):
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.source_scale_start = int(source_scale_start)
        self.source_scale_end = int(source_scale_end)
        self.scale_schedule = get_scale_schedule()
        self.scale_ranges = _scale_token_ranges(self.scale_schedule)
        if static_removed_heads_by_scale is not None:
            self._static_removed_heads_by_scale = static_removed_heads_by_scale
        else:
            self.prune_counts = dict(prune_counts)
            self._static_removed_heads_by_scale = {}

            total_slots = self.num_layers * self.num_heads
            for target_scale_idx in range(len(self.scale_schedule)):
                removed_count = max(
                    0,
                    min(total_slots, int(self.prune_counts.get(target_scale_idx, 0))),
                )
                removed_by_source = {}

                max_source_scale = min(self.source_scale_end, target_scale_idx - 1)
                for source_scale_idx in range(self.source_scale_start, max_source_scale + 1):
                    removed = _empty_removed_heads(self.num_layers)
                    order = scale_specific_removal_orders[source_scale_idx - self.source_scale_start]
                    for head_idx, layer_idx in order[:removed_count]:
                        removed[layer_idx].add(head_idx)
                    removed_by_source[source_scale_idx] = removed

                self._static_removed_heads_by_scale[target_scale_idx] = removed_by_source

    def reset(self):
        return None

    def removed_heads_for(self, layer_idx, target_scale_idx, source_scale_idx):
        target_removed = self._removed_heads_state(target_scale_idx)
        source_removed = target_removed.get(source_scale_idx)
        if source_removed is None:
            return set()
        return set(source_removed.get(layer_idx, set()))

    def removed_heads_by_source_for(self, layer_idx, target_scale_idx):
        target_removed = self._removed_heads_state(target_scale_idx)
        return {
            source_scale_idx: set(source_removed.get(layer_idx, set()))
            for source_scale_idx, source_removed in target_removed.items()
        }

    def _removed_heads_state(self, target_scale_idx):
        if target_scale_idx in self._static_removed_heads_by_scale:
            return self._static_removed_heads_by_scale[target_scale_idx]

        available_scales = [
            scale_idx
            for scale_idx in self._static_removed_heads_by_scale
            if scale_idx <= target_scale_idx
        ]
        if not available_scales:
            return {
                source_scale_idx: _empty_removed_heads(self.num_layers)
                for source_scale_idx in range(self.source_scale_start, self.source_scale_end + 1)
            }
        return self._static_removed_heads_by_scale[max(available_scales)]
