# MARL Resource Allocation Framework

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)

A modular Multi-Agent Reinforcement Learning framework for adaptive resource allocation in heterogeneous environments.

## Table of Contents
1. [Project Objective](#1-project-objective)
2. [Why MARL?](#2-why-marl)
3. [Overall Architecture](#3-overall-architecture)
4. [MARL Formulation](#4-marl-formulation)
5. [Agent Definition](#5-agent-definition)
6. [Heterogeneous Resources](#6-heterogeneous-resources)
7. [Task Representation](#7-task-representation)
8. [Partial Observability](#8-partial-observability)
9. [CTDE](#9-ctde)
10. [Reward Function](#10-reward-function)
11. [Dynamic Rewards](#11-dynamic-rewards)
12. [Matrix-V](#12-matrix-v)
13. [Communication](#13-communication)
14. [Configuration](#14-configuration)
15. [Installation](#15-installation)
16. [Training](#16-training)
17. [Evaluation](#17-evaluation)
18. [Baselines](#18-baselines)
19. [Metrics](#19-metrics)
20. [Experiments](#20-experiments)
21. [Reproducibility](#21-reproducibility)
22. [Integration With GNN](#22-integration-with-gnn)
23. [Project Structure](#23-project-structure)
24. [Component Status](#24-component-status)
25. [Future Extensions](#25-future-extensions)

---

## 1. Project Objective
The objective of this framework is to solve the problem of adaptive resource allocation in heterogeneous, decentralized compute environments. We aim to efficiently assign tasks to resources while optimizing for multiple conflicting objectives (e.g., latency, energy, SLA compliance, utilization, throughput).

## 2. Why MARL?
A centralized scheduler poses a single point of failure and becomes a bottleneck as the number of nodes scales. Multi-Agent Reinforcement Learning (MARL) allows us to distribute the scheduling logic across multiple autonomous agents, each managing its local cluster while cooperating to achieve global system objectives, ensuring better scalability and robustness.

## 3. Overall Architecture
The system pipeline flows as follows:
`Workload → GNN Embedding → MARL Policy → Allocation Action → Execution (Env) → Reward Engine → Learning`

## 4. MARL Formulation
- **Agents:** Schedulers managing individual clusters.
- **States:** Global system state containing all agent, cluster, resource, and task information.
- **Observations:** Local view for each agent (own cluster state + current task).
- **Actions:** Discrete selection of a valid resource within the agent's cluster, or "Defer".
- **Transitions:** Handled by the environment's decomposed execution model.
- **Rewards:** Multi-objective composite of local and global metrics.
- **Episodes:** Fixed length horizon processing batches of incoming workload.

## 5. Agent Definition
Agents follow a hierarchical abstraction: **Agent → Cluster → Resources**. An agent is responsible for placement decisions within its designated cluster. A cluster consists of multiple heterogeneous resources.

## 6. Heterogeneous Resources
Resources are modeled via `ResourceState` and vary in capacity. Key attributes include:
- `cpu`: Processing cores
- `memory`: Available RAM
- `gpu`: Accelerator units
- `bandwidth`: Network capacity
- `power`: Base and active power consumption rate
- `speed`: Processing multiplier

## 7. Task Representation
Tasks are structurally defined with the following fields:
- `work_units`: The abstract amount of computation required.
- `cpu_req`, `mem_req`: Resource constraints.
- `deadline`: Absolute time by which the task must complete.
- `priority`: Weight for SLA violations.
- `embedding_slot`: Placeholder for downstream GNN embeddings.

## 8. Partial Observability
Agents operate under partial observability. They only observe their local task queue and the capacities/utilization of their own managed cluster. Only the centralized critic (used during training) has access to the full global state of all clusters.

## 9. CTDE
**Centralized Training, Decentralized Execution (CTDE)** is the core learning paradigm. During training, a centralized critic network evaluates actions using the global state to reduce variance. During execution/inference, decentralized actor networks make decisions using only local observations.

## 10. Reward Function
The reward engine uses 7 components, blending local ($R_{local}$) and global ($R_{global}$) metrics:
- **Local:** Latency (completion vs deadline), Energy (power $\times$ duration), SLA (violations weighted by priority), Queue waiting time.
- **Global:** Utilization (efficiency), Load Balance (variance penalty), Throughput (completion rate).

Formula: $R = \alpha \cdot R_{local} + (1 - \alpha) \cdot R_{global}$
(Default $\alpha$ configurable via YAML).

## 11. Dynamic Rewards
**Status: 🧪 Experimental**
A module for dynamically adjusting the weights of reward components. Weights are constrained: $\sum w_i = 1$, with per-component $min$ and $max$ bounds to ensure no objective is completely ignored. Not yet formally validated.

## 12. Matrix-V
**Status: 📋 Placeholder**
A generic interface (`marl/prediction/value_prediction.py`) for a learned value predictor (Matrix-V). Awaiting finalized formulation from the broader research design.

## 13. Communication
**Status: 🧪 Experimental**
An inter-agent communication manager is implemented to allow limited message passing between agents. Disabled by default to focus on pure CTDE without communication overhead.

## 14. Configuration
Parameters are driven by 4 YAML files located in `configs/`:
- `environment.yaml`: Topology, resources, queues.
- `marl.yaml`: MAPPO hyperparameters, PPO buffers.
- `reward.yaml`: Weights, normalizers, $\alpha$.
- `experiment.yaml`: Training loops, seeds, logging.

## 15. Installation
```bash
pip install -r requirements.txt
```

## 16. Training
Run training using the provided script:
```bash
python scripts/training/train.py --config configs/experiment.yaml
```

## 17. Evaluation
Evaluate a trained checkpoint:
```bash
python scripts/evaluation/evaluate.py --checkpoint path/to/checkpoint.pt
```

## 18. Baselines
The framework includes non-learning and single-agent baselines for comparison:
- FCFS (First Come First Serve)
- Round Robin
- SJF (Shortest Job First)
- Random Feasible
- Centralized DQN (Global single agent)

## 19. Metrics
16 Key Performance Indicators (KPIs) are tracked:
- Avg/Max Latency (ms)
- Total Energy (Joules)
- Carbon Emissions ($gCO_2e$)
- SLA Violation Rate (%)
- Total Throughput (tasks/s)
- Queue Wait Time (ms)
- Resource Utilization (%)
- CPU/Mem Allocation Efficiency
- Load Variance
- Task Drop Rate
- Network Transfer Time
- Execution Time
- Step Time
- Agent Defer Rate

## 20. Experiments
Experiments can be structured by altering the YAML configs or using Hydra-like sweeping (planned). The logger natively outputs to CSV, JSON, and TensorBoard.

## 21. Reproducibility
Full reproducibility is enforced via `marl/utils/seeding.py` which seeds Python's `random`, NumPy, and PyTorch (including CuDNN deterministic flags). Environment states are logged with their seeds.

## 22. Integration With GNN
The teammate's Graph Neural Network encoder output can be injected via `marl/adapters/gnn_adapter.py`. Tasks have a reserved `embedding_slot` for the latent vector representation.

## 23. Project Structure
```text
marl-resource-allocation/
├── configs/               # YAML configuration files
├── marl/
│   ├── adapters/          # GNN and Workload interfaces
│   ├── algorithms/        # MAPPO implementation
│   ├── baselines/         # Heuristic schedulers
│   ├── communication/     # Inter-agent messaging
│   ├── environment/       # Resources, clusters, execution models
│   ├── evaluation/        # Metrics and test suites
│   ├── prediction/        # Matrix-V placeholder
│   ├── reward/            # Reward engine
│   └── utils/             # Loggers, checkpointing, seeding
├── scripts/               # Train, Evaluate runners
├── tests/                 # Invariant testing
└── requirements.txt
```

## 24. Component Status
| Component | Status | Description |
|-----------|--------|-------------|
| **Core Env** | ✅ Implemented | Agents, clusters, resources |
| **Execution Model** | ✅ Implemented | Queue + Network + Execution |
| **MAPPO Agent** | ✅ Implemented | CTDE, parameter sharing |
| **Action Masking** | ✅ Implemented | Filters invalid assignments |
| **Reward Engine** | ✅ Implemented | 7-component, normalized |
| **Baselines** | 📈 Baseline | FCFS, RR, SJF, DQN |
| **Dynamic Weights** | 🧪 Experimental | Constrained dynamic allocation |
| **Communication** | 🧪 Experimental | Inter-agent messaging |
| **Carbon Emissions**| 🧪 Experimental | Configurable intensity model |
| **Matrix-V** | 📋 Placeholder | Generic prediction interface |
| **GNN Adapter** | 📋 Placeholder | Awaiting upstream integration |

## 25. Future Extensions
- [ ] Investigate performance impacts of dynamic reward weights.
- [ ] Integrate complete Matrix-V architecture.
- [ ] Connect fully with the finalized GNN workload model.
- [ ] Introduce dynamic agent-to-cluster reassignment.
- [ ] Add hierarchical action spaces for massive clusters.

## Main Commands

**Install Dependencies:**
```bash
pip install -r requirements.txt
```

**Run Tests:**
```bash
pytest tests/
```

**Training:**
```bash
python scripts/training/train.py --config configs/experiment.yaml
```

**Evaluation:**
```bash
python scripts/evaluation/evaluate.py --config configs/experiment.yaml --checkpoint checkpoints/best_model.pt
```

**Run Baselines:**
```bash
python scripts/evaluation/run_baselines.py --config configs/experiment.yaml
```

**Hyperparameter Tuning:**
```bash
python scripts/tuning/run_hparam_sweep.py
```

**Diagnostics:**
```bash
python scripts/diagnostics/run_diagnostics.py
```

**Run Full Experiments Pipeline:**
```bash
./scripts/experiments/run_all_tuning.sh
```
