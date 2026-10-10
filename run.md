# Run Guide

Copy-paste commands for **running**, **testing**, and **showcasing** the project to the panel.
Run everything from the repository root (`marl-resource-allocation/`).

> Commands below were executed against this repo layout. Typical timings are on a laptop CPU.

---

## 0. One-time setup

```bash
pip install -r requirements.txt     # numpy, torch, pyyaml, pandas, matplotlib, gymnasium, pytest
pip install -e .                    # optional: makes `import marl` work from anywhere
```

---

## 1. Testing (~10 s)

```bash
pytest                              # runs tests/ only (configured in pytest.ini) -> 67 tests
pytest tests/test_rewards.py -v     # a single module, verbose
pytest -k "metadata or scoring"     # tests matching a keyword
```

What the suite covers: environment & invariants (no over-allocation, valid task lifecycle),
action masking, reward components, agents/actor/critic, GAE, baselines, checkpoint metadata,
config resolution, CPU-vs-memory utilization independence, score-tuple wiring, rejection caps.

---

## 2. Running the pipeline

### 2.1 Baselines (~10 s)

```bash
python scripts/evaluation/run_baselines.py --config configs/experiment.yaml
# -> results/processed/baseline_comparison.csv
# options: --baselines fcfs met  --episodes 20  --seed 42  --output <path>
```

Baselines: `fcfs`, `round_robin`, `met`, `random_feasible` (all run on the identical environment).

### 2.2 Train MAPPO

```bash
# Quick demo (~15 s): defaults from configs/experiment.yaml are 50 episodes
python scripts/training/train.py --config configs/experiment.yaml

# Full run (~3-4 min). `tee` saves the log so the training curve can be plotted from it
python scripts/training/train.py --config configs/experiment.yaml --num-episodes 1000 \
  2>&1 | tee logs/runs/train_1000ep.log
```

Outputs: `checkpoints/best_model.pt` (+ `best_model_metadata.json`), periodic `model_ep*.pt`,
`checkpoints/latest_model.pt`, `results/resolved_config.yaml`.

Best-checkpoint selection uses a **3-point moving average** of the lexicographic score tuple
over `eval_episodes` (default 10) evaluation episodes, so one lucky early evaluation cannot win.

### 2.3 Evaluate the trained model (~15 s)

```bash
python scripts/evaluation/evaluate.py --config configs/experiment.yaml \
  --checkpoint checkpoints/best_model.pt
# -> results/processed/eval_results.json
# options: --episodes 20  --seed 42  --output <path>
```

`evaluate.py` **refuses** a checkpoint whose stored `target_rejection_rate` differs from the active
config (configuration-mismatch error) instead of silently evaluating the wrong policy.

### 2.4 Side-by-side table in the terminal

```bash
python scripts/analysis/print_table.py
# explicit sources:
python scripts/analysis/print_table.py --mappo results/processed/eval_results.json \
                                       --baselines results/processed/baseline_comparison.csv
```

Prints Completion rate, Avg latency, Total energy, SLA violation rate for MAPPO, FCFS,
Round Robin, MET, Random Feasible.

### 2.5 Plots

```bash
# Training curve, rebuilt from the training log of 2.2
python scripts/analysis/plot_training_curve.py            # -> plots/training_curve.png
python scripts/analysis/plot_training_curve.py --log logs/runs/train_1000ep.log --window 10

# Rejection-cap trade-off (from results/rejection_ablation.csv)
python scripts/analysis/generate_ablation_plot.py         # -> plots/rejection_tradeoff.png

open plots/training_curve.png plots/rejection_tradeoff.png   # macOS
```

---

## 3. Panel showcase script (recommended order, ~6 min with the quick demo)

| Step | Command | What to say |
|------|---------|-------------|
| 1 | `pytest` | "67 automated tests incl. system invariants - no resource is ever over-allocated." |
| 2 | `python scripts/evaluation/run_baselines.py --config configs/experiment.yaml` | "Four classical schedulers on the *same* environment and seeds - fair comparison." |
| 3 | `python scripts/training/train.py --config configs/experiment.yaml` | "MAPPO, CTDE: decentralized actors, centralized critic. Watch reward and `Eval Ep` lines." |
| 4 | `python scripts/evaluation/evaluate.py --config configs/experiment.yaml --checkpoint checkpoints/best_model.pt` | "Deterministic evaluation of the selected checkpoint; mismatch guard protects against config drift." |
| 5 | `python scripts/analysis/print_table.py` | "Headline comparison on completion, latency, energy, SLA." |
| 6 | `open plots/training_curve.png plots/rejection_tradeoff.png` | "Training progress and why the rejection cap is 20%." |

> For the final numbers shown to the panel, use the **full 1000-episode run** (2.2), then re-run
> steps 4-5, so the table is built from a checkpoint you just produced.

---

## 4. Reading the training log

```
Eval Ep 10 | Comp Rate: 32.3% | Rej: 0.202 | Raw Score: (...) | Smoothed: (...)
```

The score tuple is compared **lexicographically** (earlier entries dominate):

| # | Entry | Direction |
|---|-------|-----------|
| 1 | task completion rate | higher is better |
| 2 | task rejection rate (negated) | lower rejection is better |
| 3 | SLA violation rate (negated) | lower is better |
| 4 | throughput | higher is better |
| 5 | avg task latency (negated) | lower is better |
| 6 | energy per completed task (negated) | lower is better |

Full metric glossary and cross-question prep: [`docs/UNDERSTAND.md`](docs/UNDERSTAND.md).

---

## 5. Longer experiments (optional, slow)

```bash
python scripts/tuning/run_ablation.py            # rejection-cap ablation (0, 0.1, 0.2, 0.3, 0.4, 1.0)
python scripts/tuning/run_hparam_sweep.py        # staged hyper-parameter sweep + final multi-seed runs
./scripts/experiments/run_all_tuning.sh          # ablation + sweep orchestration
python scripts/training/train_parallel.py        # 3 seeds (42/101/202) in parallel, 500 episodes each
python scripts/diagnostics/run_diagnostics.py    # reward/checkpoint diagnostics
```

> Not re-run during the cleanup (long-running): the commands in this section. Paths inside them were checked, but expect minutes to hours of runtime.
>
> `train_parallel.py` rewrites `configs/marl.yaml` and writes `configs/experiment_seed*.yaml`.
> Review with `git diff configs/` afterwards.

---

## 6. Repository map

```
marl-resource-allocation/
├── README.md            project overview            ├── run.md       this file
├── CHANGELOG.md                                      ├── setup.py / requirements.txt / pytest.ini
├── configs/             environment, marl, reward, experiment YAML (+ per-seed experiment files)
├── marl/                library: environment, reward, agents, algorithms, baselines,
│                        training, evaluation, adapters, communication, prediction, utils
├── scripts/
│   ├── training/        train.py, train_parallel.py
│   ├── evaluation/      evaluate.py, run_baselines.py, evaluate_final.py, run_final_test.py
│   ├── analysis/        print_table.py, plot_training_curve.py, generate_*_plot/report.py, visualize.py
│   ├── tuning/          run_ablation.py, run_hparam_sweep.py, auto_sweep.py, hyperparameter_tuner.py
│   ├── diagnostics/     audit_*.py, debug_*.py, run_diagnostics.py, sanity_check.py
│   └── experiments/     run_all_tuning.sh
├── tests/               pytest suite (67 tests)
├── docs/                UNDERSTAND.md (presentation guide), ARCHITECTURE_NOTES, DECISIONS,
│                        REQUIREMENTS, RESEARCH_NOTES
├── results/
│   ├── processed/       canonical eval_results.json + baseline_comparison.csv
│   ├── archive/         earlier/conflicting copies of those files (kept, not deleted)
│   └── ablation_*, sweep_*, final_run_*, *.csv   raw experiment output
├── plots/               training_curve.png, rejection_tradeoff.png
├── logs/runs/           curated stdout logs (train / eval / baselines)
├── experiments/         tuning configs, logs, plots
└── checkpoints/         model weights (git-ignored; regenerate with train.py)
```

Git policy: weights (`*.pt`), caches and per-run output are git-ignored; CSV/JSON/YAML/PNG
research evidence stays tracked. See `.gitignore`.
