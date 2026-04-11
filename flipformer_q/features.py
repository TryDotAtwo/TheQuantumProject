from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from .domain import inverse_permutation, suffix_sorted_len


def state_single_features(state: Sequence[int]) -> np.ndarray:
    n = len(state)
    inv = inverse_permutation(state)
    suf = suffix_sorted_len(state)
    feats = np.zeros((n, 8), dtype=np.float32)
    for i, v in enumerate(state):
        feats[i, 0] = (i + 1) / n
        feats[i, 1] = v / n
        feats[i, 2] = (inv[v - 1] + 1) / n
        feats[i, 3] = abs(v - (i + 1)) / n
        feats[i, 4] = 1.0 if i < n - 1 and abs(state[i] - state[i + 1]) != 1 else 0.0
        feats[i, 5] = 1.0 if i > 0 and abs(state[i] - state[i - 1]) != 1 else 0.0
        feats[i, 6] = 1.0 if i >= n - suf else 0.0
        feats[i, 7] = 1.0 if v == i + 1 else 0.0
    return feats


def state_pair_features(state: Sequence[int]) -> np.ndarray:
    n = len(state)
    pair = np.zeros((n, n, 8), dtype=np.float32)
    arr = np.asarray(state, dtype=np.int64)
    for i in range(n):
        for j in range(n):
            pair[i, j, 0] = (j - i) / max(1, n - 1)
            pair[i, j, 1] = (arr[j] - arr[i]) / max(1, n - 1)
            pair[i, j, 2] = 1.0 if arr[i] < arr[j] else 0.0
            pair[i, j, 3] = 1.0 if (i - j) * (arr[i] - arr[j]) < 0 else 0.0
            pair[i, j, 4] = 1.0 if i == j else 0.0
            pair[i, j, 5] = 1.0 if abs(arr[i] - arr[j]) == 1 else 0.0
            pair[i, j, 6] = abs(arr[i] - arr[j]) / max(1, n - 1)
            pair[i, j, 7] = abs(i - j) / max(1, n - 1)
    return pair


def action_features(state: Sequence[int], action: int) -> np.ndarray:
    n = len(state)
    k = action
    out = np.zeros(10, dtype=np.float32)
    prefix = np.asarray(state[:k], dtype=np.float32)
    suffix = np.asarray(state[k:], dtype=np.float32) if k < n else np.zeros((0,), dtype=np.float32)
    out[0] = k / n
    out[1] = float(prefix.mean()) / n if len(prefix) else 0.0
    out[2] = float(prefix.std()) / max(1.0, n)
    out[3] = float(prefix[0]) / n if len(prefix) else 0.0
    out[4] = float(prefix[-1]) / n if len(prefix) else 0.0
    out[5] = float(suffix[0]) / n if len(suffix) else 0.0
    out[6] = 1.0 if k >= n // 2 else 0.0
    out[7] = 1.0 if k == n else 0.0
    out[8] = float(np.abs(np.diff(prefix)).mean()) / max(1.0, n) if len(prefix) > 1 else 0.0
    out[9] = float(np.abs(prefix[::-1] - np.sort(prefix)).mean()) / max(1.0, n) if len(prefix) else 0.0
    return out


def quantum_summary_features(state: Sequence[int], action: int) -> np.ndarray:
    s = state_single_features(state)
    a = action_features(state, action)
    n = len(state)
    q = np.zeros(8, dtype=np.float32)
    q[:4] = s.mean(axis=0)[:4]
    q[4:] = a[:4]
    q = np.clip(q, 0.0, 1.0)
    return q
