from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal


QuantumMode = Literal["none", "aer_cpu", "aer_gpu"]
MIN_SUPPORTED_N = 10
MAX_SUPPORTED_N = 60


@dataclass
class Config:
    n: int = 30
    train_samples: int = 100000
    benchmark_samples: int = 200
    shard_size: int = 5000
    seed: int = 123456
    scramble_min: int = 10
    scramble_max: int = 80

    d_model: int = 128
    d_pair: int = 64
    n_heads: int = 8
    n_blocks: int = 3
    recycle_steps: int = 2
    mlp_mult: int = 8
    head_mlp_mult: int = 6
    dropout: float = 0.1

    batch_size: int = 64
    micro_batch_size: int = 0
    epochs: int = 2
    lr: float = 3e-4
    weight_decay: float = 1e-4
    clip_grad_norm: float = 1.0
    log_every: int = 10
    amp: bool = True

    beam_width: int = 48
    beam_depth: int = 96
    expand_topk: int = 12
    rerank_topk: int = 4
    quantum_weight: float = 0.25
    uncertainty_margin: float = 0.10

    quantum_mode: QuantumMode = "none"
    quantum_qubits: int = 4
    quantum_layers: int = 2
    quantum_cache_size: int = 100000

    population_size: int = 8
    elite_size: int = 2
    generations: int = 5
    patience: int = 2

    selftrain_rounds: int = 3
    accept_margin: float = 1e-4

    num_workers: int = 4
    device: str = "auto"

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> "Config":
        if not MIN_SUPPORTED_N <= self.n <= MAX_SUPPORTED_N:
            raise ValueError(f"n must be in [{MIN_SUPPORTED_N}, {MAX_SUPPORTED_N}], got {self.n}")
        if self.train_samples <= 0 or self.benchmark_samples <= 0 or self.shard_size <= 0:
            raise ValueError("train_samples, benchmark_samples, and shard_size must be positive")
        if self.scramble_min < 1 or self.scramble_max < self.scramble_min:
            raise ValueError("scramble bounds must satisfy 1 <= scramble_min <= scramble_max")
        if self.d_model <= 0 or self.d_pair <= 0:
            raise ValueError("d_model and d_pair must be positive")
        if self.n_heads <= 0 or self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be positive and divisible by n_heads")
        if self.d_pair > self.d_model:
            raise ValueError("d_pair must be <= d_model")
        if self.n_blocks <= 0 or self.recycle_steps <= 0:
            raise ValueError("n_blocks and recycle_steps must be positive")
        if self.mlp_mult < 1 or self.head_mlp_mult < 1:
            raise ValueError("mlp multipliers must be positive")
        if not math.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.batch_size <= 0 or self.micro_batch_size < 0:
            raise ValueError("batch_size must be positive and micro_batch_size must be nonnegative")
        if self.epochs <= 0 or not math.isfinite(self.lr) or self.lr <= 0 or not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("epochs and lr must be positive; weight_decay must be nonnegative")
        if (
            not math.isfinite(self.clip_grad_norm)
            or self.clip_grad_norm < 0
            or self.log_every <= 0
            or self.num_workers < 0
        ):
            raise ValueError("clip_grad_norm and num_workers must be nonnegative; log_every must be positive")
        if self.beam_width <= 0 or self.beam_depth <= 0 or self.expand_topk <= 0:
            raise ValueError("beam_width, beam_depth, and expand_topk must be positive")
        if self.rerank_topk < 0 or self.quantum_cache_size < 0:
            raise ValueError("rerank_topk and quantum_cache_size must be nonnegative")
        if self.quantum_qubits <= 0 or self.quantum_layers < 0:
            raise ValueError("quantum_qubits must be positive and quantum_layers nonnegative")
        if not math.isfinite(self.quantum_weight) or self.quantum_weight < 0:
            raise ValueError("quantum_weight must be finite and nonnegative")
        if self.quantum_mode not in {"none", "aer_cpu", "aer_gpu"}:
            raise ValueError(f"Unsupported quantum_mode: {self.quantum_mode}")
        if self.population_size < 2 or not 1 <= self.elite_size < self.population_size:
            raise ValueError("evolution requires population_size >= 2 and 1 <= elite_size < population_size")
        if self.generations <= 0 or self.patience <= 0 or self.selftrain_rounds <= 0:
            raise ValueError("generations, patience, and selftrain_rounds must be positive")
        if not math.isfinite(self.accept_margin) or self.accept_margin < 0:
            raise ValueError("accept_margin must be finite and nonnegative")
        if not math.isfinite(self.uncertainty_margin) or not 0.0 <= self.uncertainty_margin <= 1.0:
            raise ValueError("uncertainty_margin must be in [0, 1]")
        if not self.device.strip():
            raise ValueError("device must be a nonempty device name")
        return self
