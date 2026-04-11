from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .domain import State, all_actions, apply_flip, heuristic_value, is_sorted
from .features import action_features, quantum_summary_features, state_pair_features, state_single_features


@dataclass
class SearchResult:
    solved: bool
    path: List[int]
    final_state: State
    nodes_expanded: int
    best_length: int
    q_calls: int = 0


def _prepare_model_inputs(state: State, device: torch.device):
    single = torch.tensor(state_single_features(state), dtype=torch.float32, device=device).unsqueeze(0)
    pair = torch.tensor(state_pair_features(state), dtype=torch.float32, device=device).unsqueeze(0)
    acts = np.stack([action_features(state, a) for a in all_actions(len(state))], axis=0)
    action_t = torch.tensor(acts, dtype=torch.float32, device=device).unsqueeze(0)
    return single, pair, action_t


def _prepare_model_inputs_batch(states: Sequence[State], device: torch.device):
    singles = np.stack([state_single_features(state) for state in states], axis=0).astype(np.float32)
    pairs = np.stack([state_pair_features(state) for state in states], axis=0).astype(np.float32)
    actions = [
        np.stack([action_features(state, a) for a in all_actions(len(state))], axis=0).astype(np.float32)
        for state in states
    ]
    action_t = np.stack(actions, axis=0).astype(np.float32)
    return (
        torch.as_tensor(singles, dtype=torch.float32, device=device),
        torch.as_tensor(pairs, dtype=torch.float32, device=device),
        torch.as_tensor(action_t, dtype=torch.float32, device=device),
    )


@torch.no_grad()
def model_scores(model, state: State, device: torch.device):
    single, pair, action_t = _prepare_model_inputs(state, device)
    out = model(single, pair, action_t)
    policy = torch.softmax(out["policy_logits"], dim=-1)[0].detach().cpu().numpy()
    q_values = out["q_values"][0].detach().cpu().numpy()
    value = float(out["value"][0].detach().cpu().item())
    far = torch.softmax(out["far_logits"], dim=-1)[0].detach().cpu().numpy()
    return policy, q_values, value, far


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
    if is_sorted(start):
        return SearchResult(True, [], start, nodes_expanded=0, best_length=0, q_calls=0)

    beam = [(heuristic_value(start), 0, start, [])]
    visited = {start: 0}
    best_length = 10 ** 9
    nodes_expanded = 0
    q_calls = 0

    for depth in range(beam_depth):
        candidates = []
        active = []
        for _, g, state, path in beam:
            if is_sorted(state):
                best_length = min(best_length, len(path))
                return SearchResult(True, path, state, nodes_expanded, best_length, q_calls=q_calls)
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
                h = max(0.0, -float(q_values[idx]))
                f = ng + 0.5 * h + 0.5 * max(0.0, -value)
                candidates.append((f, ng, nxt, path + [a]))
                nodes_expanded += 1

        if not candidates:
            break
        candidates.sort(key=lambda x: x[0])
        beam = candidates[:beam_width]

    best = min(beam, key=lambda x: x[0]) if beam else (0, 0, start, [])
    return SearchResult(False, best[3], best[2], nodes_expanded, len(best[3]), q_calls=q_calls)


def beam_teacher_targets(model, state: State, device: torch.device, beam_width: int, beam_depth: int):
    actions = all_actions(len(state))
    child_values = []
    child_lengths = []
    for a in actions:
        nxt = apply_flip(state, a)
        if is_sorted(nxt):
            child_values.append(0.0)
            child_lengths.append(1)
            continue
        res = beam_search(
            model=model,
            start=nxt,
            beam_width=max(8, beam_width // 2),
            beam_depth=max(8, beam_depth // 2),
            expand_topk=min(len(actions), 8),
            device=device,
            rerank_topk=0,
            quantum_emulator=None,
            quantum_weight=0.0,
        )
        child_values.append(float(len(res.path)))
        child_lengths.append(len(res.path) + 1)
    vals = np.asarray(child_values, dtype=np.float32)
    best = float(vals.min())
    opt = (vals <= best + 1e-6).astype(np.float32)
    if opt.sum() <= 0:
        probs = np.ones_like(vals) / len(vals)
    else:
        probs = opt / opt.sum()
    far = (vals >= vals.max() - 1e-6).astype(np.float32)
    far = far / max(far.sum(), 1.0)
    value = best
    q = vals
    return probs.astype(np.float32), np.asarray(q, dtype=np.float32), far.astype(np.float32), float(value)
