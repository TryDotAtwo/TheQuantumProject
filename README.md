# Flipformer pancake-sorting experiments

This repository contains a research prototype for sorting permutations with **prefix reversals** (the pancake-sorting problem), for `10 <= n <= 60`.

The model uses single-position and pair-position features with policy, action-cost (`q`), and state-cost (`value`) heads. Search is approximate beam search. Bootstrap targets use a gap-based heuristic; self-training uses one model-guided beam search per generated example to choose a teacher action. Neither target is an exact distance oracle, and the search has no optimality or completeness guarantee.

The optional Qiskit Aer circuit supplies features for local action reranking. It is not a trainable variational quantum model, and this repository makes no claim that the quantum path improves solution quality or speed. A requested Aer backend fails explicitly if it cannot initialize; GPU requests do not fall back to CPU.

## Install

Core CPU training and classical search:

```bash
python -m pip install -r requirements.txt
```

Plots and quantum reranking are optional:

```bash
python -m pip install -r requirements-plots.txt
python -m pip install -r requirements-quantum.txt
```

`--device auto` selects CUDA when available and CPU otherwise. An explicit `--device cuda` or `--device cuda:N` errors if that device is unavailable.

## Workflow

Generate a local bootstrap dataset:

```bash
python run_flipformer_quantum.py generate --n 30 --train-samples 20000 --shard-size 5000 --scramble-min 10 --scramble-max 80 --dataset-dir outputs/bootstrap_n30
```

Train a first model:

```bash
python run_flipformer_quantum.py train --n 30 --manifest outputs/bootstrap_n30/manifest.json --run-dir outputs/run_n30 --epochs 2 --batch-size 64
```

`--init-ckpt` loads compatible model weights and starts a fresh optimizer. `--resume` restores a full training checkpoint, including optimizer, AMP scaler, epoch, and random-number-generator state. Resume checkpoints are written atomically at epoch boundaries to `model_last.pt`; `--epochs` means additional epochs. Older weights-only checkpoints can be warm-started but cannot provide a full resume.

Evaluate the model with the same seeded benchmark states across comparison modes:

```bash
python run_flipformer_quantum.py compare --n 30 --run-dir outputs/compare_n30 --cls-ckpt outputs/run_n30/model_last.pt --hybrid-ckpt outputs/run_n30/model_last.pt --quantum-mode aer_cpu
```

Hybrid evaluation requires `--quantum-mode aer_cpu` or `aer_gpu`. The result records the requested and actual backend. Trial output distinguishes solved solution length from the partial depth of unsolved searches. Mean solution length covers solved trials only and is `null` when none were solved.

Other entry points are `stage-run`, `iterative`, and `evolve`. Run `python run_flipformer_quantum.py <command> --help` for their options.

## Data format

Datasets use sharded `.npy` arrays and a JSON manifest whose shard paths are relative to the manifest. Legacy `.npz` shards are still readable. The bundled `test_ds` files are small fixtures for data-format inspection, not evidence of model quality or a full-scale run.

## Known limits

- Teacher labels are approximate and depend on the current model and beam settings.
- Evolution ranks candidates by solved rate, solution length on solved trials, gap heuristic on unsolved terminal states, and expanded-node cost. The gap term is a search proxy, not an exact remaining distance.
- Search quality and runtime depend on `n`, beam width/depth, hardware, and reranking settings.
- Benchmark scores are comparative heuristics for this prototype; they are not certificates of optimality.
- No full-scale sample-count, throughput, or quantum advantage claim is made by this repository.
