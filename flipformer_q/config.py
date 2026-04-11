from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal


QuantumMode = Literal["none", "aer_cpu", "aer_gpu"]
DatasetMode = Literal["bootstrap", "selftrain", "distill", "stage3"]
StageMode = Literal["seed", "selftrain", "distill", "quantum"]
SelectionMetric = Literal["score", "solved_rate", "mean_length"]

MIN_SUPPORTED_N = 10
MAX_SUPPORTED_N = 60


@dataclass
class Config:
    n: int = 30
    train_samples: int = 100000
    val_samples: int = 5000
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
    offspring_per_generation: int = 6
    generations: int = 5
    patience: int = 2

    selftrain_rounds: int = 3
    accept_margin: float = 1e-4
    stage_mode: StageMode = "seed"

    lambda_value: float = 1.0
    lambda_policy: float = 1.0
    lambda_q: float = 0.75
    lambda_far: float = 0.5
    lambda_rank: float = 0.25

    num_workers: int = 4
    device: str = "cuda"
    out_dir: str = "outputs"

    def output_path(self) -> Path:
        return Path(self.out_dir)

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> "Config":
        if not MIN_SUPPORTED_N <= self.n <= MAX_SUPPORTED_N:
            raise ValueError(f"n must be in [{MIN_SUPPORTED_N}, {MAX_SUPPORTED_N}], got {self.n}")
        if self.d_model <= 0 or self.d_pair <= 0:
            raise ValueError(f"d_model and d_pair must be positive, got d_model={self.d_model}, d_pair={self.d_pair}")
        if self.n_heads <= 0 or self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads, got d_model={self.d_model}, n_heads={self.n_heads}")
        if self.d_pair > self.d_model:
            raise ValueError(f"d_pair must be <= d_model, got d_pair={self.d_pair}, d_model={self.d_model}")
        if self.n_blocks <= 0 or self.recycle_steps <= 0:
            raise ValueError(f"n_blocks and recycle_steps must be positive, got n_blocks={self.n_blocks}, recycle_steps={self.recycle_steps}")
        if self.mlp_mult < 1 or self.head_mlp_mult < 1:
            raise ValueError(f"mlp_mult and head_mlp_mult must be >= 1, got mlp_mult={self.mlp_mult}, head_mlp_mult={self.head_mlp_mult}")
        if self.batch_size <= 0 or self.micro_batch_size < 0:
            raise ValueError(f"batch_size must be positive and micro_batch_size must be >= 0, got batch_size={self.batch_size}, micro_batch_size={self.micro_batch_size}")
        return self

    def with_stage(self, stage: StageMode) -> "Config":
        d = self.to_dict()
        d["stage_mode"] = stage
        cfg = Config(**d)
        if stage == "seed":
            cfg.lambda_value = 1.0
            cfg.lambda_policy = 0.0
            cfg.lambda_q = 1.0
            cfg.lambda_far = 0.0
            cfg.lambda_rank = 0.0
        elif stage == "selftrain":
            cfg.lambda_value = 1.0
            cfg.lambda_policy = 0.25
            cfg.lambda_q = 0.75
            cfg.lambda_far = 0.0
            cfg.lambda_rank = 0.10
        elif stage == "distill":
            cfg.lambda_value = 1.0
            cfg.lambda_policy = 0.75
            cfg.lambda_q = 1.0
            cfg.lambda_far = 0.35
            cfg.lambda_rank = 0.20
        elif stage == "quantum":
            cfg.lambda_value = 1.0
            cfg.lambda_policy = 1.0
            cfg.lambda_q = 1.0
            cfg.lambda_far = 0.50
            cfg.lambda_rank = 0.25
        return cfg
