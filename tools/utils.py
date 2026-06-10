# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: MIT

import json
import math
# import types
# import numpy as np
# from pathlib import Path
from typing import Any

# import pandas as pd
# from matplotlib.colors import ListedColormap
# from matplotlib import pyplot as plt
# from PIL import Image
# # import torch
from infinity.utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates


def get_dataset(dataset_name: str):
    from datasets import load_dataset

    if dataset_name == "takara":
        return load_dataset("takara-ai/image_captions", split="train", streaming=True)
    if dataset_name in {"coco2017", "test"}:
        return load_dataset("phiyodr/coco2017", split="validation")
    if dataset_name in ('GenEval'):
        return geneval_dataset(dataset_name)
    if dataset_name == 'HPSv21':
        return hpsv21_dataset()
    raise ValueError(f"Dataset not implemented: {dataset_name}")

def geneval_dataset(dataset):
    with open(f"datasets/{dataset}.json", 'r') as f:
        geneval = json.load(f)
    for item in geneval:
        for _ in range(4):
            yield item
            
def hpsv21_dataset():
    prompts = hpsv2.benchmark_prompts('all')
    for _,v in prompts.items():
        for entry in v:
            yield {"prompt": entry}


def get_prompt(entry: dict[str, Any], dataset_name: str) -> str:
    if dataset_name in {"coco2017", "test"}:
        captions = entry.get("captions", [])
        prompt = captions[0] if captions else ""
    else:
        prompt = entry.get("caption", "")
    return prompt.strip() if isinstance(prompt, str) else ""


def _resolve_template(template):
    if isinstance(template, int):
        return float(h_div_w_templates[template])
    if isinstance(template, str):
        return float(template)
    return float(template)


def _scale_schedule(template, pn):
    template_key = _resolve_template(template)
    return [(1, h, w) for (_, h, w) in dynamic_resolution_h_w[template_key][pn]["scales"]]


def _validate_removal_order(removal_order, *, num_layers, num_heads):
    total_slots = int(num_layers) * int(num_heads)
    if len(removal_order) != total_slots:
        raise ValueError(
            f"expected {total_slots} removal-order entries, got {len(removal_order)}"
        )

    normalized_order = []
    seen = set()
    for idx, pair in enumerate(removal_order):
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError(
                f"removal_order[{idx}] must be a (head, layer) tuple"
            )
        head, layer = int(pair[0]), int(pair[1])
        if not 0 <= head < num_heads:
            raise ValueError(f"head index out of range at position {idx}: {head}")
        if not 0 <= layer < num_layers:
            raise ValueError(f"layer index out of range at position {idx}: {layer}")
        key = (head, layer)
        if key in seen:
            raise ValueError(f"duplicate (head, layer) entry at position {idx}: {key}")
        seen.add(key)
        normalized_order.append(key)
    return normalized_order


def _validate_scale_specific_removal_orders(
    scale_specific_removal_orders,
    *,
    num_layers,
    num_heads,
    source_scale_start=4,
    source_scale_end=11,
):
    expected_orders = int(source_scale_end) - int(source_scale_start) + 1
    if len(scale_specific_removal_orders) != expected_orders:
        raise ValueError(
            "scale_specific_removal_orders must contain exactly "
            f"{expected_orders} orders for source scales {source_scale_start}-{source_scale_end}"
        )

    normalized = []
    for offset, removal_order in enumerate(scale_specific_removal_orders):
        normalized.append(
            tuple(
                _validate_removal_order(
                    removal_order,
                    num_layers=num_layers,
                    num_heads=num_heads,
                )
            )
        )
    return tuple(normalized)


def _scale_specific_order_ranks(scale_specific_removal_orders):
    return {
        source_scale_idx: {
            slot: rank
            for rank, slot in enumerate(removal_order)
        }
        for source_scale_idx, removal_order in scale_specific_removal_orders.items()
    }


def _scale_specific_atom_key(atom, order_ranks):
    source_scale_idx, head_idx, layer_idx = atom
    return (-layer_idx, -source_scale_idx, order_ranks[source_scale_idx][(head_idx, layer_idx)])


def _scale_specific_target_removed_atoms(
    target_scale_idx,
    removed_count,
    *,
    scale_specific_removal_orders,
    source_scale_start=4,
    source_scale_end=11,
):
    max_source_scale = min(int(source_scale_end), int(target_scale_idx))
    if max_source_scale < int(source_scale_start):
        return set()

    removed_atoms = set()
    for source_scale_idx in range(int(source_scale_start), max_source_scale + 1):
        for head_idx, layer_idx in scale_specific_removal_orders[source_scale_idx][:removed_count]:
            removed_atoms.add((source_scale_idx, head_idx, layer_idx))
    return removed_atoms


def _empty_removed_heads_by_source_scale(num_layers, source_scale_indices):
    return {
        source_scale_idx: {layer_idx: tuple() for layer_idx in range(num_layers)}
        for source_scale_idx in source_scale_indices
    }


def _removed_atoms_by_source_and_layer(removed_atoms, *, num_layers, source_scale_indices):
    removed = {
        source_scale_idx: {layer_idx: [] for layer_idx in range(num_layers)}
        for source_scale_idx in source_scale_indices
    }
    for source_scale_idx, head_idx, layer_idx in removed_atoms:
        removed[source_scale_idx][layer_idx].append(head_idx)
    return {
        source_scale_idx: {
            layer_idx: tuple(heads)
            for layer_idx, heads in layer_map.items()
        }
        for source_scale_idx, layer_map in removed.items()
    }


def _strict_scale_specific_cache_total_after_layer(
    *,
    target_scale_idx,
    written_layer_idx,
    scale_tokens,
    num_layers,
    num_heads,
    protected_prefix_scales,
    source_scale_start,
    source_scale_end,
    active_removed,
):
    num_layers = int(num_layers)
    num_heads = int(num_heads)
    written_layer_idx = int(written_layer_idx)
    target_scale_idx = int(target_scale_idx)

    active_removed_counts = {}
    for source_scale_idx, _, layer_idx in active_removed:
        key = (int(source_scale_idx), int(layer_idx))
        active_removed_counts[key] = active_removed_counts.get(key, 0) + 1

    protected_cached_scales = min(int(protected_prefix_scales), target_scale_idx)
    protected_prefix_tokens = sum(scale_tokens[:protected_cached_scales])
    total = num_layers * num_heads * protected_prefix_tokens

    max_source_scale = min(int(source_scale_end), target_scale_idx)
    for source_scale_idx in range(int(source_scale_start), max_source_scale + 1):
        for layer_idx in range(num_layers):
            if source_scale_idx == target_scale_idx and layer_idx > written_layer_idx:
                continue
            removed_here = active_removed_counts.get((source_scale_idx, layer_idx), 0)
            total += (num_heads - removed_here) * scale_tokens[source_scale_idx]

    return total


def _strict_scale_specific_activation_candidates(
    target_removed_atoms,
    active_removed,
    layer_idx,
    scale_idx,
    *,
    order_ranks,
):
    candidates = [
        atom
        for atom in target_removed_atoms
        if atom not in active_removed and not (scale_idx == atom[0] and layer_idx < atom[2])
    ]
    candidates.sort(key=lambda atom: _scale_specific_atom_key(atom, order_ranks))
    return candidates


def pruned_heads_for_budget(
    budget,
    *,
    retained_tokens=51,
    num_layers=32,
    num_heads=16,
    skip_last_scale=True,
):
    """
    Compute how many (layer, head) cache slots must be pruned after each scale
    so the peak incoming KV-cache footprint stays within the requested budget
    ratio.

    A pruned slot retains at most ``retained_tokens`` cached positions from the
    already materialized cache. The newly generated tokens for the current
    scale are not counted in this budget calculation.

    The returned counts are keyed by the full ``get_scale_schedule()`` indices.
    """
    budget = float(budget)
    if not 0.0 <= budget <= 1.0:
        raise ValueError(f"budget must be in [0, 1], got {budget}")

    scale_schedule = get_scale_schedule()
    if not scale_schedule:
        return {}

    total_slots = int(num_layers) * int(num_heads)
    if total_slots <= 0:
        raise ValueError("num_layers * num_heads must be positive")
    if retained_tokens < 0:
        raise ValueError("retained_tokens must be non-negative")

    scale_tokens = [h * w for _, h, w in scale_schedule]
    incoming_cache_tokens = []
    running_cache_tokens = 0
    for idx, tokens in enumerate(scale_tokens):
        incoming_cache_tokens.append(running_cache_tokens)
        if idx < len(scale_tokens) - 1 or not skip_last_scale:
            running_cache_tokens += tokens

    full_peak_tokens = max(incoming_cache_tokens, default=0)
    if full_peak_tokens <= 0:
        return {idx: 0 for idx in range(len(scale_schedule))}

    min_budget = min(retained_tokens, full_peak_tokens) / full_peak_tokens
    if budget + 1e-12 < min_budget:
        raise ValueError(
            f"budget {budget} is below the minimum feasible ratio "
            f"{min_budget:.12f}"
        )

    budgeted_peak_tokens = budget * full_peak_tokens
    pruned_counts = {}

    for idx, incoming_tokens in enumerate(incoming_cache_tokens):
        retained_here = min(retained_tokens, incoming_tokens)
        if incoming_tokens <= budgeted_peak_tokens:
            required_pruned = 0
        elif incoming_tokens == retained_here:
            raise ValueError(
                f"budget {budget} is infeasible at scale index {idx}"
            )
        else:
            numerator = total_slots * (incoming_tokens - budgeted_peak_tokens)
            denominator = incoming_tokens - retained_here
            required_pruned = math.ceil(numerator / denominator)

        pruned_counts[idx] = max(0, min(total_slots, required_pruned))

    return pruned_counts


def strict_pruning_schedule_for_budget_scale_specific(
    budget,
    scale_specific_removal_orders,
    *,
    num_layers=32,
    num_heads=16,
    source_scale_start=4,
    source_scale_end=11,
    protected_prefix_scales=4,
    skip_last_scale=True,
):
    """
    Compute a strict static pruning schedule for fine-grained scale-specific
    virtual pruning.

    Each removable unit is a ``(source_scale, head, layer)`` atom. The number
    of removed heads per source scale at target scale ``t`` is fixed by
    ``pruned_heads_for_budget(...)``, and the removed atoms are exactly the
    first ``x`` entries of each source scale's provided order. The scheduler
    only chooses how early those already-decided removals must become active
    within scale ``t`` so every layer-cache checkpoint stays under budget,
    using this priority:
    - last layer to first
    - larger source scales first within a layer
    - source-scale order rank within a ``(layer, source_scale)`` bucket
    """
    budget = float(budget)
    if not 0.0 <= budget <= 1.0:
        raise ValueError(f"budget must be in [0, 1], got {budget}")

    num_layers = int(num_layers)
    num_heads = int(num_heads)
    if num_layers <= 0 or num_heads <= 0:
        raise ValueError("num_layers * num_heads must be positive")

    normalized_orders_seq = _validate_scale_specific_removal_orders(
        scale_specific_removal_orders,
        num_layers=num_layers,
        num_heads=num_heads,
        source_scale_start=source_scale_start,
        source_scale_end=source_scale_end,
    )
    normalized_orders = {
        int(source_scale_start) + offset: removal_order
        for offset, removal_order in enumerate(normalized_orders_seq)
    }
    order_ranks = _scale_specific_order_ranks(normalized_orders)

    scale_schedule = get_scale_schedule()
    if not scale_schedule:
        return {
            "final_removed_counts": {},
            "final_removed_by_scale": {},
            "early_activation_by_scale": {},
        }

    scale_tokens = [h * w for _, h, w in scale_schedule]
    budgeted_scale_count = len(scale_tokens) - 1 if skip_last_scale else len(scale_tokens)
    source_scale_indices = tuple(
        range(int(source_scale_start), min(int(source_scale_end), len(scale_tokens) - 1) + 1)
    )
    if budgeted_scale_count <= 0:
        zero_by_scale = {idx: 0 for idx in range(len(scale_tokens))}
        empty_by_scale = {
            idx: _empty_removed_heads_by_source_scale(num_layers, source_scale_indices)
            for idx in range(len(scale_tokens))
        }
        return {
            "final_removed_counts": zero_by_scale,
            "final_removed_by_scale": empty_by_scale,
            "early_activation_by_scale": {idx: {} for idx in range(len(scale_tokens))},
        }

    full_peak_tokens = sum(scale_tokens[:budgeted_scale_count])
    protected_prefix_tokens = sum(scale_tokens[:min(int(protected_prefix_scales), len(scale_tokens))])
    min_budget = protected_prefix_tokens / full_peak_tokens if full_peak_tokens > 0 else 0.0
    if budget + 1e-12 < min_budget:
        raise ValueError(
            f"budget {budget} is below the minimum feasible ratio "
            f"{min_budget:.12f}"
        )

    threshold_total_tokens = budget * num_layers * num_heads * full_peak_tokens
    prune_counts = pruned_heads_for_budget(
        budget,
        retained_tokens=protected_prefix_tokens,
        num_layers=num_layers,
        num_heads=num_heads,
        skip_last_scale=skip_last_scale,
    )

    final_removed_counts = {}
    final_removed_by_scale = {}
    early_activation_by_scale = {}

    active_removed = set()

    for target_scale_idx in range(len(scale_tokens)):
        next_scale_idx = min(target_scale_idx + 1, len(scale_tokens) - 1)
        target_removed_count = int(
            prune_counts.get(
                next_scale_idx,
                prune_counts.get(target_scale_idx, 0),
            )
        )
        if target_scale_idx >= budgeted_scale_count:
            final_removed_counts[target_scale_idx] = len(active_removed)
            final_removed_by_scale[target_scale_idx] = _removed_atoms_by_source_and_layer(
                active_removed,
                num_layers=num_layers,
                source_scale_indices=source_scale_indices,
            )
            early_activation_by_scale[target_scale_idx] = {}
            continue

        target_removed_atoms = _scale_specific_target_removed_atoms( # Correct, after this scale all these atoms will be removed
            target_scale_idx,
            target_removed_count,
            scale_specific_removal_orders=normalized_orders,
            source_scale_start=source_scale_start,
            source_scale_end=source_scale_end,
        )

        current_active_removed = set(active_removed)
        early_map = {}

        for layer_idx in range(num_layers):
            passed_target_removed = {
                atom for atom in target_removed_atoms
                if atom[2] <= layer_idx
            }
            current_active_removed.update(passed_target_removed)

            total = _strict_scale_specific_cache_total_after_layer( # Correct
                target_scale_idx=target_scale_idx,
                written_layer_idx=layer_idx,
                scale_tokens=scale_tokens,
                num_layers=num_layers,
                num_heads=num_heads,
                protected_prefix_scales=protected_prefix_scales,
                source_scale_start=source_scale_start,
                source_scale_end=source_scale_end,
                active_removed=current_active_removed,
            )
            if total <= threshold_total_tokens:
                continue

            activated_now = []
            for atom in _strict_scale_specific_activation_candidates(
                target_removed_atoms,
                current_active_removed,
                layer_idx=layer_idx,
                scale_idx=target_scale_idx,
                order_ranks=order_ranks,
            ):
                if atom[2] <= layer_idx:
                    continue
                current_active_removed.add(atom)
                activated_now.append(atom)
                total = _strict_scale_specific_cache_total_after_layer( # correct
                    target_scale_idx=target_scale_idx,
                    written_layer_idx=layer_idx,
                    scale_tokens=scale_tokens,
                    num_layers=num_layers,
                    num_heads=num_heads,
                    protected_prefix_scales=protected_prefix_scales,
                    source_scale_start=source_scale_start,
                    source_scale_end=source_scale_end,
                    active_removed=current_active_removed,
                )
                if total <= threshold_total_tokens:
                    break

            if total > threshold_total_tokens:
                raise ValueError(
                    f"budget {budget} is infeasible at scale index {target_scale_idx}, layer {layer_idx}"
                )
            if activated_now:
                early_map[layer_idx] = activated_now

        final_total = _strict_scale_specific_cache_total_after_layer(
            target_scale_idx=target_scale_idx,
            written_layer_idx=num_layers - 1,
            scale_tokens=scale_tokens,
            num_layers=num_layers,
            num_heads=num_heads,
            protected_prefix_scales=protected_prefix_scales,
            source_scale_start=source_scale_start,
            source_scale_end=source_scale_end,
            active_removed=current_active_removed,
        )
        if final_total > threshold_total_tokens:
            raise ValueError(
                f"budget {budget} is infeasible at end of scale index {target_scale_idx}"
            )
        if current_active_removed != target_removed_atoms:
            raise RuntimeError(
                "strict scale-specific schedule failed to activate "
                f"end-of-scale removals at scale index {target_scale_idx}"
            )

        active_removed = set(target_removed_atoms)
        final_removed_counts[target_scale_idx] = len(active_removed)
        final_removed_by_scale[target_scale_idx] = _removed_atoms_by_source_and_layer(
            active_removed,
            num_layers=num_layers,
            source_scale_indices=source_scale_indices,
        )
        early_activation_by_scale[target_scale_idx] = early_map

    return {
        "final_removed_counts": final_removed_counts,
        "final_removed_by_scale": final_removed_by_scale,
        "early_activation_by_scale": early_activation_by_scale,
    }


def strict_removed_heads_by_scale_specific_for_budget(
    budget,
    scale_specific_removal_orders,
    *,
    num_layers=32,
    num_heads=16,
    source_scale_start=4,
    source_scale_end=11,
    protected_prefix_scales=4,
    skip_last_scale=True,
    strict_schedule=None,
):
    normalized_orders_seq = _validate_scale_specific_removal_orders(
        scale_specific_removal_orders,
        num_layers=num_layers,
        num_heads=num_heads,
        source_scale_start=source_scale_start,
        source_scale_end=source_scale_end,
    )
    scale_schedule = get_scale_schedule()
    schedule = strict_pruning_schedule_for_budget_scale_specific(
        budget,
        normalized_orders_seq,
        num_layers=num_layers,
        num_heads=num_heads,
        source_scale_start=source_scale_start,
        source_scale_end=source_scale_end,
        protected_prefix_scales=protected_prefix_scales,
        skip_last_scale=skip_last_scale,
    ) if strict_schedule is None else strict_schedule

    source_scale_indices = tuple(
        range(int(source_scale_start), min(int(source_scale_end), len(scale_schedule) - 1) + 1)
    )
    removed_heads_by_scale = {}
    removed_counts_by_scale = {}
    remaining_atoms_by_scale = {}

    prev_final_removed = set()
    total_atoms = 0
    for target_scale_idx in range(len(scale_schedule)):
        max_source_scale = min(int(source_scale_end), target_scale_idx, len(scale_schedule) - 1)
        eligible_source_scale_indices = tuple(
            range(int(source_scale_start), max_source_scale + 1)
        ) if max_source_scale >= int(source_scale_start) else tuple()
        total_atoms = len(eligible_source_scale_indices) * int(num_layers) * int(num_heads)

        scale_removed = set(prev_final_removed)
        early_map = schedule["early_activation_by_scale"].get(target_scale_idx, {})
        for atoms in early_map.values():
            scale_removed.update(atoms)

        removed_heads_by_scale[target_scale_idx] = _removed_atoms_by_source_and_layer(
            scale_removed,
            num_layers=num_layers,
            source_scale_indices=source_scale_indices,
        )
        removed_counts_by_scale[target_scale_idx] = len(scale_removed)
        remaining_atoms_by_scale[target_scale_idx] = total_atoms - len(scale_removed)

        final_removed = set()
        for source_scale_idx, layer_map in schedule["final_removed_by_scale"].get(target_scale_idx, {}).items():
            for layer_idx, heads in layer_map.items():
                for head_idx in heads:
                    final_removed.add((int(source_scale_idx), int(head_idx), int(layer_idx)))
        prev_final_removed = final_removed

    return {
        "removed_heads_by_scale": removed_heads_by_scale,
        "removed_counts_by_scale": removed_counts_by_scale,
        "remaining_atoms_by_scale": remaining_atoms_by_scale,
        "strict_schedule": schedule,
    }


def get_scale_schedule():
    return _scale_schedule(1.0, "1M")


def orders(start_order=3, prompts=10, model_type=2):
    try:
        with open('outputs/scas_2b/scas_orders.json') as file:
            return json.load(file)['orders']
    except:
        pass
    assert start_order == 3 and prompts == 10
    with open(f'orders/{model_type}b.json', 'r') as file:
            return json.load(file)
