# Flipformer + local quantum emulator

Локальный исследовательский пакет для pancake sorting без внешних датасетов.

Архитектура:
- `Flipformer`: AF2-like `single stream + pair stream + recycling + flip-aware heads`
- `beam search`: и для teacher-label generation, и для inference
- `local quantum emulator`: Qiskit Aer CPU/GPU для reranking top-K actions
- `self-training`: bootstrap -> self-train -> evaluate -> accept/reject
- `evolution`: population of 8 models, crossover, mutation, lineage graph

## Ограничения v1

1. Для произвольного `n` система не строит exact labels повсюду.  
   Teacher labels для больших `n` строятся из beam-search под текущей моделью; следовательно это self-improving approximate solver, не exact oracle.

2. Полный `1,000,000` samples поддерживается через sharded `.npz`, но wall-clock зависит от beam settings и GPU.

3. Quantum block не обучается как variational circuit end-to-end.  
   Quantum block используется как local Aer-based feature extractor/reranker. Такая схема нужна, чтобы сначала стабильно отладить архитектуру.

4. На GPU квантовая эмуляция работает только при корректной установке Aer с GPU support.  
   В документации Qiskit Aer указано, что `AerSimulator` поддерживает GPU для `statevector`-метода, а `save_statevector()` позволяет сохранить состояние симуляции; Aer также предоставляет свои primitives для симуляции. citeturn539630search15turn539630search1turn539630search11

## Структура

- `flipformer_q/model.py` — Flipformer
- `flipformer_q/search.py` — beam search + teacher targets
- `flipformer_q/data.py` — local dataset shards
- `flipformer_q/quantum/aer_emulator.py` — Qiskit Aer emulator
- `flipformer_q/train.py` — supervised training
- `flipformer_q/evolution.py` — self-training + genetic loop
- `run_flipformer_quantum.py` — CLI

## Быстрый старт

### 1. Bootstrap dataset

```bash
python run_flipformer_quantum.py generate \
  --n 30 \
  --train-samples 20000 \
  --shard-size 5000 \
  --scramble-min 10 \
  --scramble-max 80 \
  --dataset-dir outputs/ds_bootstrap_n30
```

### 2. Train first model

```bash
python run_flipformer_quantum.py train \
  --n 30 \
  --manifest outputs/ds_bootstrap_n30/manifest.json \
  --run-dir outputs/run_bootstrap_n30 \
  --epochs 2 \
  --batch-size 64 \
  --device cuda
```

### 3. Generate self-train dataset with current model

```bash
python run_flipformer_quantum.py generate \
  --n 30 \
  --train-samples 50000 \
  --shard-size 5000 \
  --dataset-dir outputs/ds_self_n30 \
  --teacher-ckpt outputs/run_bootstrap_n30/model_last.pt \
  --device cuda
```

### 4. Continue training from previous checkpoint

```bash
python run_flipformer_quantum.py train \
  --n 30 \
  --manifest outputs/ds_self_n30/manifest.json \
  --run-dir outputs/run_self_n30 \
  --init-ckpt outputs/run_bootstrap_n30/model_last.pt \
  --epochs 2 \
  --batch-size 64 \
  --device cuda
```

### 5. Compare classical vs hybrid

```bash
python run_flipformer_quantum.py compare \
  --n 30 \
  --run-dir outputs/compare_n30 \
  --cls-ckpt outputs/run_self_n30/model_last.pt \
  --hybrid-ckpt outputs/run_self_n30/model_last.pt \
  --benchmark-samples 200 \
  --quantum-mode aer_gpu \
  --device cuda
```

### 6. Evolutionary self-training

```bash
python run_flipformer_quantum.py evolve \
  --n 30 \
  --run-dir outputs/evolve_n30 \
  --train-samples 100000 \
  --shard-size 10000 \
  --benchmark-samples 200 \
  --population-size 8 \
  --elite-size 2 \
  --generations 5 \
  --patience 2 \
  --device cuda \
  --quantum-mode none
```

## На что смотреть

- `outputs/.../train_metrics.csv`
- `outputs/.../eval/trials_*.csv`
- `outputs/.../compare/*png`
- `outputs/.../lineage_graph.json`

## Когда включать quantum mode

Сначала:
- `--quantum-mode none`

Потом:
- `--quantum-mode aer_cpu`  
или
- `--quantum-mode aer_gpu`

Qiskit Aer — high-performance simulator; IBM docs показывают использование Aer primitives и Aer simulator для exact/noisy local simulation, а Aer docs описывают `AerSimulator(method="statevector", device="GPU")` и `save_statevector()` для получения состояния. citeturn539630search11turn539630search15turn539630search1turn539630search9
