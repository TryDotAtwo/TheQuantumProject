from __future__ import annotations

import random
from typing import Iterable, List, Sequence, Tuple

import numpy as np


State = Tuple[int, ...]


def identity_state(n: int) -> State:
    return tuple(range(1, n + 1))


def all_actions(n: int) -> List[int]:
    return list(range(2, n + 1))


def apply_flip(state: Sequence[int], k: int) -> State:
    return tuple(list(state[:k])[::-1] + list(state[k:]))


def is_sorted(state: Sequence[int]) -> bool:
    return all(state[i] == i + 1 for i in range(len(state)))


def random_scramble_depth(rng: random.Random, n: int, lo: int, hi: int) -> int:
    lo = max(1, lo)
    hi = max(lo, hi)
    return rng.randint(lo, hi)


def random_state(n: int, rng: random.Random, scramble: int) -> State:
    s = list(identity_state(n))
    actions = all_actions(n)
    prev = None
    for _ in range(scramble):
        a = rng.choice(actions)
        # avoid immediate exact cancellation by repeating same flip twice
        if prev is not None and a == prev and len(actions) > 1:
            cands = [x for x in actions if x != prev]
            a = rng.choice(cands)
        s = list(apply_flip(s, a))
        prev = a
    return tuple(s)


def inverse_permutation(state: Sequence[int]) -> np.ndarray:
    inv = np.empty(len(state), dtype=np.int64)
    for pos, val in enumerate(state):
        inv[val - 1] = pos
    return inv


def gap_heuristic(state: Sequence[int]) -> int:
    ext = list(state) + [len(state) + 1]
    gaps = 0
    for i in range(len(ext) - 1):
        if abs(ext[i] - ext[i + 1]) != 1:
            gaps += 1
    return gaps


def adjacency_breaks(state: Sequence[int]) -> int:
    out = 0
    for i in range(len(state) - 1):
        if state[i + 1] != state[i] + 1:
            out += 1
    return out


def suffix_sorted_len(state: Sequence[int]) -> int:
    n = len(state)
    out = 0
    for i in range(n - 1, -1, -1):
        if state[i] == i + 1:
            out += 1
        else:
            break
    return out


def heuristic_value(state: Sequence[int]) -> float:
    n = len(state)
    gaps = gap_heuristic(state)
    br = adjacency_breaks(state)
    suf = suffix_sorted_len(state)
    inv = inverse_permutation(state)
    pos_err = float(np.abs(inv - np.arange(n)).mean())
    return 1.25 * gaps + 0.65 * br + 0.08 * pos_err - 0.10 * suf
