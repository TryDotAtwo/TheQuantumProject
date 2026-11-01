from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import torch

from .domain import State, all_actions, apply_flip, gap_heuristic, is_sorted
from .features import all_action_features, quantum_summary_features, state_pair_features, state_single_features


@dataclass
class SearchResult:
    solved: bool
    path: List[int]
    final_state: State
    nodes_expanded: int
    q_calls: int = 0


def _prepare_model_inputs_batch(states: Sequence[State], device: torch.device):
    singles = np.stack([state_single_features(state) for state in states], axis=0).astype(np.float32)
    pairs = np.stack([state_pair_features(state) for state in states], axis=0).astype(np.float32)
    actions = [
        all_action_features(state).astype(np.float32)
        for state in states
    ]
    action_t = np.stack(actions, axis=0).astype(np.float32)
    return (
        torch.as_tensor(singles, dtype=torch.float32, device=device),
        torch.as_tensor(pairs, dtype=torch.float32, device=device),
        torch.as_tensor(action_t, dtype=torch.float32, device=device),
    )


@torch.inference_mode()
def model_scores_batch_for_search(model, states: Sequence[State], device: torch.device):
    single, pair, action_t = _prepare_model_inputs_batch(states, device)
    out = model(single, pair, action_t, heads={"value", "policy", "q"})
    policies = torch.softmax(out["policy_logits"], dim=-1).detach().cpu().numpy()
    q_values = out["q_values"].detach().cpu().numpy()
    values = out["value"].detach().cpu().numpy().astype(np.float32)
    return policies, q_values, values


def _uncertain(policy: np.ndarray, margin: float) -> bool:
    if len(policy) < 2:
        return False
    top2 = np.sort(policy)[-2:]
    return float(top2[-1] - top2[-2]) < margin


def beam_search(
    model,
    start: State,
    beam_width: int,
    beam_depth: int,
    expand_topk: int,
    device: torch.device,
    rerank_topk: int = 0,
    quantum_emulator=None,
    quantum_weight: float = 0.0,
    uncertainty_margin: float = 0.1,
) -> SearchResult:
    if beam_width <= 0 or beam_depth <= 0 or expand_topk <= 0:
        raise ValueError("beam_width, beam_depth, and expand_topk must be positive")
    if is_sorted(start):
        return SearchResult(True, [], start, nodes_expanded=0, q_calls=0)

    beam = [(0.0, 0, start, [])]
    visited = {start: 0}
    nodes_expanded = 0
    q_calls = 0

    for depth in range(beam_depth):
        candidates = []
        active = []
        for _, g, state, path in beam:
            if is_sorted(state):
                return SearchResult(True, path, state, nodes_expanded, q_calls=q_calls)
            active.append((g, state, path))

        if not active:
            break

        policies, q_batch, values = model_scores_batch_for_search(model, [state for _, state, _ in active], device)

        for beam_idx, (g, state, path) in enumerate(active):
            policy = policies[beam_idx]
            q_values = q_batch[beam_idx]
            value = float(values[beam_idx])
            actions = all_actions(len(state))
            order = np.argsort(-policy)[:expand_topk].tolist()

            # optional quantum rerank only on uncertain states
            if quantum_emulator is not None and rerank_topk > 0 and _uncertain(policy, uncertainty_margin):
                top = np.argsort(-policy)[:rerank_topk].tolist()
                for idx in top:
                    qfeat = quantum_summary_features(state, actions[idx])
                    qres = quantum_emulator.compute(qfeat)
                    q_calls += 1
                    policy[idx] += quantum_weight * float(qres.features.mean())

            order = np.argsort(-policy)[:expand_topk].tolist()

            for idx in order:
                a = actions[idx]
                nxt = apply_flip(state, a)
                ng = g + 1
                if visited.get(nxt, 10 ** 9) <= ng:
                    continue
                visited[nxt] = ng
                # Training targets are nonnegative remaining-cost proxies. Keep
                # that sign convention consistent at inference time.
                action_remaining = max(0.0, float(q_values[idx]))
                state_remaining = max(0.0, value)
                f = g + 0.5 * state_remaining + 0.5 * (1.0 + action_remaining)
                if is_sorted(nxt):
                    solved_path = path + [a]
                    return SearchResult(True, solved_path, nxt, nodes_expanded + 1, q_calls=q_calls)
                candidates.append((f, ng, nxt, path + [a]))
                nodes_expanded += 1

        if not candidates:
            break
        candidates.sort(key=lambda x: x[0])
        beam = candidates[:beam_width]

    best = min(beam, key=lambda x: x[0]) if beam else (0, 0, start, [])
    return SearchResult(False, best[3], best[2], nodes_expanded, q_calls=q_calls)


def beam_teacher_targets(model, state: State, device: torch.device, beam_width: int, beam_depth: int):
    """Build one approximate search-policy label plus consistent gap-cost targets.

    A prior implementation launched a separate deep beam search for every legal
    flip. One shared search is enough to select a teacher action; per-action
    values stay in the same gap-heuristic scale as bootstrap labels.
    """
    actions = all_actions(len(state))
    vals = np.asarray([gap_heuristic(apply_flip(state, a)) for a in actions], dtype=np.float32)
    best = float(vals.min())
    opt = (vals <= best + 1e-6).astype(np.float32)
    probs = opt / max(float(opt.sum()), 1.0)

    result = beam_search(
        model=model,
        start=state,
        beam_width=beam_width,
        beam_depth=beam_depth,
        expand_topk=min(len(actions), 8),
        device=device,
        rerank_topk=0,
    )
    if result.solved and result.path:
        teacher_action = result.path[0]
        teacher_index = teacher_action - 2
        probs.fill(0.0)
        probs[teacher_index] = 1.0

    far = (vals >= vals.max() - 1e-6).astype(np.float32)
    far = far / max(far.sum(), 1.0)
    value = float(vals.min())
    q = vals
    return probs.astype(np.float32), np.asarray(q, dtype=np.float32), far.astype(np.float32), float(value)
