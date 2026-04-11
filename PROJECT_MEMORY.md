# Project Memory

## 2026-04-11

- prompt: user asked whether project architecture is similar to AlphaFold 2.
- finding: architecture is AF2-like only at a high level: single stream, pair stream, pair-biased attention, triangular pair update, recycling, multi-head prediction; it does not implement AlphaFold 2's biological MSA/template/structure module pipeline.
- change: none; answer-only task.

- prompt: user requested adding `--num-workers` support across CLI, `Config`, and training `DataLoader`.
- change: `run_flipformer_quantum.py` declares `--num-workers` and forwards `args.num_workers` into `Config`.
- change: `flipformer_q/config.py` stores `num_workers: int = 4`.
- change: `flipformer_q/train.py` passes `prefetch_factor=2 if cfg.num_workers > 0 else None` to `DataLoader`.
- verification: `py run_flipformer_quantum.py iterative --help` shows `--num-workers NUM_WORKERS`.
- verification: `py -B -c ... ast.parse(...)` succeeded for changed Python files.
- note: `py_compile` failed because Python could not rename `.pyc` files under existing cache paths; `.pycache_check` cleanup also failed with Windows `AccessDenied`.
- rule: check project rules before edits; update project memory after task-relevant changes.

- prompt: user reported GPU not loaded and requested log plus hardware-load inspection.
- finding: `nvidia-smi` showed RTX 3070 GPU memory allocated by `python.exe` but SM/GPU utilization stayed at 0%; CPU had one main Python process plus 8 Python worker processes consuming CPU.
- finding: `outputs/iter_n40/dataset_seed_r0/manifest.json` had `device="cuda"` and `num_workers=8`; `run_seed_r0/train_metrics.csv` appended a second run but stopped around step 8 during inspection.
- diagnosis: CUDA is reached because GPU memory is allocated; GPU is starved by input/data pipeline work, especially compressed `.npz` loading plus per-sample `torch.tensor(...)` conversion in `ShardedNpzDataset.__getitem__`.
- next_fix_candidate: replace compressed/lazy per-sample NPZ access with uncompressed or memory-mapped shards and batch-level zero-copy/as_tensor conversion.

- prompt: user approved speedups for dataloader and action head after observing first train batch elapsed ~274s with GPU utilization 0%.
- change: `write_sharded_dataset` now writes shard directories containing uncompressed `.npy` arrays and records `storage="npy_dir"` in manifest.
- change: `ShardedNpzDataset` now supports new `.npy` shard directories through `np.load(..., mmap_mode="r")`, keeps legacy `.npz` support, implements batch `__getitems__`, and converts numpy batches with `torch.as_tensor`.
- change: `FlipAwareActionHead` now computes all action scores in one vectorized pass instead of a Python loop over actions.
- change: `FlipformerModel.forward` now accepts optional `heads`; training now skips zero-weight heads, so bootstrap training computes only `value` and `q` heads.
- change: AMP calls moved from deprecated `torch.cuda.amp.*` API to `torch.amp.*`.
- verification: AST parse passed for `data.py`, `train.py`, and `model.py`; legacy `.npz` dataloader batch smoke passed; model full-head and selective-head smoke passed; bootstrap loss smoke passed.
- note: running training process must be restarted before the speedups take effect because already-running Python code uses modules loaded before the patch.

- result: user reran iterative seed stage after speedups; `dataset_seed_r0/manifest.json` uses `storage="npy_dir"` with shard directories.
- result: first train batch changed from ~274s before speedups to ~3.5s; full seed epoch changed to ~11.57s for 79 batches; train throughput reached ~6.83 batches/s by final logged step.
- result: sampled RTX 3070 utilization during training reached ~24-28% instead of 0%; GPU memory stayed around ~2031MiB.
- finding: dataloader/action-head bottleneck is substantially reduced; next likely bottlenecks are small model size, pair-stream compute shape, and CPU-heavy dataset generation for bootstrap/selftrain data.

- prompt: user reported GPU utilization ~29-31% with VRAM ~7229MiB and asked whether GPU should be near 100%.
- finding: seed training files finished at `17:01:22`, but Python process kept running with no newer output files; likely active work was `evaluate_individual` / `beam_search`, which previously called model forward once per state inside Python loops.
- change: `beam_search` now batches model scoring for all active beam states per depth via `model_scores_batch_for_search`, uses `torch.inference_mode`, and requests only `value`, `policy`, and `q` heads for search.
- change: `run_stage` now deletes teacher/training model references and calls `torch.cuda.empty_cache()` between dataset generation, training, and evaluation; `evaluate_individual` also clears CUDA cache after evaluation.
- change: `evaluate_individual` now prints progress through `ProgressPrinter`, so long evaluation/search stages are visible in terminal.
- verification: AST parse passed for `search.py`, `pipeline.py`, and `evolution.py`; CPU beam-search smoke passed.
- note: already-running PID keeps old loaded Python modules; batched search and cache cleanup require process restart.

- prompt: user requested larger trainable perceptron layers and explicit support for `n` from 10 to 100.
- change: default model width increased to `d_model=64`, `d_pair=32`; default MLP multiplier increased to `mlp_mult=6`; new `head_mlp_mult=4` config/CLI parameter controls value/action head MLP width.
- change: `FeedForward`, `FlipAwareActionHead`, and `value_head` now use deeper two-hidden-layer MLPs instead of single-hidden-layer MLPs.
- change: `Config.validate()` now enforces `10 <= n <= 100`, positive dimensions, `d_model % n_heads == 0`, `d_pair <= d_model`, positive block/recycle counts, and positive MLP multipliers.
- change: CLI now exposes `--mlp-mult` and `--head-mlp-mult`; `build_model` validates config; `train_model` checks manifest `n` equals config `n`.
- verification: AST parse passed; forward shape smoke passed for `n=10` and `n=100`; CLI config parsing passed; CPU train smoke on `test_ds` with `n=10` passed; default model parameter count is `1,264,396`.
- note: `n=100` architecture shapes are valid, but safe GPU batch size remains hardware-dependent because pair stream memory scales as `O(batch*n^2*d_pair)` and triangle update compute scales heavily with `n`.

- prompt: user challenged `64/32/6/4` defaults as too small.
- change: default capacity increased to `d_model=128`, `d_pair=64`, `n_heads=8`, `n_blocks=3`, `mlp_mult=8`, `head_mlp_mult=6`.
- verification: AST parse passed; forward shape smoke passed for `n=10` and `n=100`; CLI default parsing passed; new default parameter count is `10,865,692`.

- prompt: user narrowed target scope to solving up to `n=60` for now.
- change: `MAX_SUPPORTED_N` changed from `100` to `60`; validation now accepts `10 <= n <= 60` and rejects `n=61`.
- verification: AST parse passed; forward shape smoke passed for `n=60`; config and CLI validation passed for `n=60`; config and CLI validation rejected `n=61`.
