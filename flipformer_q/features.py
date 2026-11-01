from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from .domain import inverse_permutation, suffix_sorted_len


def state_single_features(state: Sequence[int]) -> np.ndarray:
    n = len(state)
    inv = inverse_permutation(state)
    suf = suffix_sorted_len(state)
    values = np.asarray(state, dtype=np.int64)
    positions = np.arange(n, dtype=np.int64)
    feats = np.zeros((n, 8), dtype=np.float32)
    feats[:, 0] = (positions + 1) / n
    feats[:, 1] = values / n
    feats[:, 2] = (inv[values - 1] + 1) / n
    feats[:, 3] = np.abs(values - (positions + 1)) / n
    breaks = np.abs(np.diff(values)) != 1
    feats[:-1, 4] = breaks
    feats[1:, 5] = breaks
    if suf:
        feats[n - suf :, 6] = 1.0
    feats[:, 7] = values == positions + 1
    return feats


def state_pair_features(state: Sequence[int]) -> np.ndarray:
    n = len(state)
    arr = np.asarray(state, dtype=np.int64)
    denom = max(1, n - 1)
    positions = np.arange(n, dtype=np.int64)
    index_delta = positions[None, :] - positions[:, None]
    value_delta = arr[None, :] - arr[:, None]
    pair = np.empty((n, n, 8), dtype=np.float32)
    pair[..., 0] = index_delta / denom
    pair[..., 1] = value_delta / denom
    pair[..., 2] = arr[:, None] < arr[None, :]
    pair[..., 3] = (index_delta * value_delta) < 0
    pair[..., 4] = np.eye(n, dtype=bool)
    pair[..., 5] = np.abs(value_delta) == 1
    pair[..., 6] = np.abs(value_delta) / denom
    pair[..., 7] = np.abs(index_delta) / denom
    return pair


def action_features(state: Sequence[int], action: int) -> np.ndarray:
    n = len(state)
    if not 2 <= action <= n:
        raise ValueError(f"action must be in [2, {n}], got {action}")
    prefix = np.asarray(state[:action], dtype=np.float32)
    suffix = np.asarray(state[action:], dtype=np.float32)
    out = np.zeros(10, dtype=np.float32)
    out[0] = action / n
    out[1] = prefix.mean() / n
    out[2] = prefix.std() / max(1.0, n)
    out[3] = prefix[0] / n
    out[4] = prefix[-1] / n
    out[5] = suffix[0] / n if suffix.size else 0.0
    out[6] = action >= n // 2
    out[7] = action == n
    out[8] = np.abs(np.diff(prefix)).mean() / max(1.0, n) if prefix.size > 1 else 0.0
    out[9] = np.abs(prefix[::-1] - np.sort(prefix)).mean() / max(1.0, n)
    return out


def all_action_features(state: Sequence[int]) -> np.ndarray:
    """Build all prefix-flip action features together, avoiding one Python pass per action."""
    n = len(state)
    values = np.asarray(state, dtype=np.float32)
    actions = np.arange(2, n + 1, dtype=np.int64)
    positions = np.arange(n, dtype=np.int64)
    prefix_mask = positions[None, :] < actions[:, None]
    prefix_values = np.where(prefix_mask, values[None, :], 0.0)
    lengths = actions.astype(np.float32)
    means = prefix_values.sum(axis=1) / lengths
    variances = np.maximum((prefix_values**2).sum(axis=1) / lengths - means**2, 0.0)

    out = np.zeros((n - 1, 10), dtype=np.float32)
    out[:, 0] = actions / n
    out[:, 1] = means / n
    out[:, 2] = np.sqrt(variances) / max(1.0, n)
    out[:, 3] = values[0] / n
    out[:, 4] = values[actions - 1] / n
    suffix_exists = actions < n
    out[suffix_exists, 5] = values[actions[suffix_exists]] / n
    out[:, 6] = actions >= n // 2
    out[:, 7] = actions == n

    adjacent_diff = np.abs(np.diff(values))
    diff_mask = positions[:-1][None, :] < (actions[:, None] - 1)
    out[:, 8] = np.where(diff_mask, adjacent_diff[None, :], 0.0).sum(axis=1) / np.maximum(actions - 1, 1) / n

    reversed_indices = np.maximum(actions[:, None] - 1 - positions[None, :], 0)
    reversed_prefix = np.where(prefix_mask, values[reversed_indices], 0.0)
    sorted_prefix = np.sort(np.where(prefix_mask, values[None, :], np.inf), axis=1)
    out[:, 9] = np.where(prefix_mask, np.abs(reversed_prefix - sorted_prefix), 0.0).sum(axis=1) / lengths / n
    return out


def quantum_summary_features(state: Sequence[int], action: int) -> np.ndarray:
    s = state_single_features(state)
    a = action_features(state, action)
    q = np.zeros(8, dtype=np.float32)
    q[:4] = s.mean(axis=0)[:4]
    q[4:] = a[:4]
    q = np.clip(q, 0.0, 1.0)
    return q
