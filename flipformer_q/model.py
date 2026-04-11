from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PairBiasedAttention(nn.Module):
    def __init__(self, d_model: int, d_pair: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.d_pair = d_pair
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.pair_bias = nn.Linear(d_pair, n_heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pair: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q = self.q(x).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(x).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(x).view(b, n, self.n_heads, self.head_dim).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        bias = self.pair_bias(pair).permute(0, 3, 1, 2)
        logits = logits + bias
        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)
        y = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, n, d)
        return self.out(y)


class TrianglePairUpdate(nn.Module):
    def __init__(self, d_pair: int, dropout: float):
        super().__init__()
        self.ln = nn.LayerNorm(d_pair)
        self.left = nn.Linear(d_pair, d_pair)
        self.right = nn.Linear(d_pair, d_pair)
        self.out = nn.Linear(d_pair, d_pair)
        self.dropout = nn.Dropout(dropout)

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        x = self.ln(pair)
        a = self.left(x)
        b = self.right(x)
        # (B,N,N,D), aggregate over middle index k
        tri = torch.einsum("bikd,bkjd->bijd", a, b) / math.sqrt(max(1, x.size(-1)))
        return self.dropout(self.out(torch.tanh(tri)))


class SingleToPair(nn.Module):
    def __init__(self, d_model: int, d_pair: int, dropout: float):
        super().__init__()
        self.left = nn.Linear(d_model, d_pair)
        self.right = nn.Linear(d_model, d_pair)
        self.diff = nn.Linear(d_model, d_pair)
        self.out = nn.Linear(d_pair, d_pair)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        l = self.left(x)[:, :, None, :]
        r = self.right(x)[:, None, :, :]
        d = self.diff(x[:, :, None, :] - x[:, None, :, :])
        y = torch.tanh(l + r + d)
        return self.dropout(self.out(y))


class FeedForward(nn.Module):
    def __init__(self, d_in: int, mult: int, dropout: float):
        super().__init__()
        hidden = d_in * mult
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Linear(d_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_in),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FlipformerBlock(nn.Module):
    def __init__(self, d_model: int, d_pair: int, n_heads: int, mlp_mult: int, dropout: float):
        super().__init__()
        self.single_ln = nn.LayerNorm(d_model)
        self.attn = PairBiasedAttention(d_model, d_pair, n_heads, dropout)
        self.single_ff = FeedForward(d_model, mlp_mult, dropout)
        self.single_to_pair = SingleToPair(d_model, d_pair, dropout)
        self.pair_ln = nn.LayerNorm(d_pair)
        self.tri = TrianglePairUpdate(d_pair, dropout)
        self.pair_ff = FeedForward(d_pair, mlp_mult, dropout)

    def forward(self, x: torch.Tensor, pair: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x + self.attn(self.single_ln(x), pair)
        x = x + self.single_ff(x)
        pair = pair + self.single_to_pair(x)
        pair = pair + self.tri(self.pair_ln(pair))
        pair = pair + self.pair_ff(pair)
        return x, pair


class FlipAwareActionHead(nn.Module):
    def __init__(self, d_model: int, d_pair: int, action_dim: int, n_actions: int, head_mlp_mult: int, dropout: float):
        super().__init__()
        action_hidden = d_model * head_mlp_mult
        score_hidden = d_model * head_mlp_mult
        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, action_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(action_hidden, action_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(action_hidden, d_model),
        )
        self.score_mlp = nn.Sequential(
            nn.Linear(d_model * 3 + d_pair, score_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(score_hidden, score_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(score_hidden, 1),
        )
        self.n_actions = n_actions

    def forward(self, single: torch.Tensor, pair: torch.Tensor, action_feats: torch.Tensor) -> torch.Tensor:
        _, n, _ = single.shape
        action_count = action_feats.shape[1]
        if action_count != self.n_actions:
            raise ValueError(f"Expected {self.n_actions} actions, got {action_count}")

        pooled_single = single.mean(dim=1)[:, None, :].expand(-1, action_count, -1)
        pair_diag = pair.mean(dim=(1, 2))
        act_embed = self.action_mlp(action_feats)
        if act_embed.shape[-1] < pair_diag.shape[-1]:
            raise ValueError(f"Action embedding width {act_embed.shape[-1]} is smaller than pair width {pair_diag.shape[-1]}")

        action_pos = torch.arange(2, action_count + 2, device=single.device)
        prefix_sum = single.cumsum(dim=1)
        prefix_sum_at_action = prefix_sum[:, action_pos - 1, :]
        prefix = prefix_sum_at_action / action_pos.to(single.dtype).view(1, action_count, 1)

        total = prefix_sum[:, -1:, :]
        suffix_count = (n - action_pos).to(single.dtype).view(1, action_count, 1)
        suffix = (total - prefix_sum_at_action) / suffix_count.clamp_min(1.0)
        suffix = torch.where((action_pos < n).view(1, action_count, 1), suffix, torch.zeros_like(suffix))

        pair_context = pair_diag[:, None, :] + act_embed[:, :, : pair_diag.shape[-1]]
        h = torch.cat([pooled_single, prefix, suffix, pair_context], dim=-1)
        return self.score_mlp(h).squeeze(-1)


class FlipformerModel(nn.Module):
    def __init__(
        self,
        n: int,
        single_in: int,
        pair_in: int,
        action_in: int,
        d_model: int = 128,
        d_pair: int = 64,
        n_heads: int = 8,
        n_blocks: int = 3,
        recycle_steps: int = 3,
        mlp_mult: int = 8,
        head_mlp_mult: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n = n
        self.n_actions = n - 1
        self.recycle_steps = recycle_steps

        self.single_in = nn.Linear(single_in, d_model)
        self.pair_in = nn.Linear(pair_in, d_pair)
        self.blocks = nn.ModuleList(
            [FlipformerBlock(d_model, d_pair, n_heads, mlp_mult, dropout) for _ in range(n_blocks)]
        )
        self.recycle_single = nn.Linear(d_model, d_model)
        self.recycle_pair = nn.Linear(d_pair, d_pair)

        value_hidden = d_model * head_mlp_mult
        self.value_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, value_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(value_hidden, value_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(value_hidden, 1),
        )
        self.policy_head = FlipAwareActionHead(d_model, d_pair, action_in, self.n_actions, head_mlp_mult, dropout)
        self.q_head = FlipAwareActionHead(d_model, d_pair, action_in, self.n_actions, head_mlp_mult, dropout)
        self.far_head = FlipAwareActionHead(d_model, d_pair, action_in, self.n_actions, head_mlp_mult, dropout)

    def trunk(self, single: torch.Tensor, pair: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.single_in(single)
        p = self.pair_in(pair)
        for r in range(self.recycle_steps):
            if r > 0:
                s = s + self.recycle_single(s)
                p = p + self.recycle_pair(p)
            for block in self.blocks:
                s, p = block(s, p)
        return s, p

    def forward(
        self,
        single: torch.Tensor,
        pair: torch.Tensor,
        action_feats: torch.Tensor,
        heads: Optional[set[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        if heads is None:
            heads = {"value", "policy", "q", "far"}
        s, p = self.trunk(single, pair)
        pooled = s.mean(dim=1)
        out = {
            "single_latent": s,
            "pair_latent": p,
        }
        if "value" in heads:
            out["value"] = self.value_head(pooled).squeeze(-1)
        if "policy" in heads:
            out["policy_logits"] = self.policy_head(s, p, action_feats)
        if "q" in heads:
            out["q_values"] = self.q_head(s, p, action_feats)
        if "far" in heads:
            out["far_logits"] = self.far_head(s, p, action_feats)
        return out
