# Requirements Tracking

| ID | Requirement | Source | Status | Implementation | Notes |
|----|-------------|--------|--------|---------------|-------|
| R001 | Multi-agent environment | Project PPT | ✅ Done | `marl/environment/marl_env.py` | 3 agents, configurable |
| R002 | Heterogeneous resources | Project PPT | ✅ Done | `marl/environment/resource.py` | CPU/memory/GPU/bandwidth/power/speed |
| R003 | Agent → Cluster → Resource hierarchy | Design Review | ✅ Done | `marl/environment/cluster.py` | Configurable mapping |
| R004 | Multi-objective reward | Project PPT | ✅ Done | `marl/reward/` | 7 components, weighted |
| R005 | Local task queues | Project PPT | ✅ Done | `marl/environment/task_queue.py` | Environment-owned, configurable ordering |
| R006 | CTDE architecture | Project PPT | ✅ Done | `marl/algorithms/mappo.py` | Centralized critic, decentralized actors |
| R007 | Partial observability | Project PPT | ✅ Done | `marl/environment/marl_env.py` | Local obs + global state |
| R008 | Action masking | Design Review | ✅ Done | `marl/environment/marl_env.py` | Prevents infeasible allocations |
| R009 | Adaptive/dynamic reward weights | Project PPT | 🧪 Experimental | `marl/reward/dynamic_weight_selector.py` | Constrained (Σwi=1, min/max bounds) |
| R010 | Matrix-V prediction | Project PPT | 📋 Placeholder | `marl/prediction/value_prediction.py` | Generic interface, awaiting formulation |
| R011 | Classical baselines | Project PPT | ✅ Done | `marl/baselines/` | FCFS, Round Robin, SJF, Random Feasible |
| R012 | Centralized DQN baseline | Project PPT | ✅ Done | `marl/baselines/centralized_dqn.py` | Single-agent with global state |
| R013 | Inter-agent communication | Project PPT | 🧪 Experimental | `marl/communication/` | Disabled by default |
| R014 | GNN adapter interface | Integration | 📋 Placeholder | `marl/adapters/gnn_adapter.py` | Awaiting teammate's GNN |
| R015 | Workload adapter interface | Integration | ✅ Done | `marl/adapters/workload_adapter.py` | Simple encoder implemented |
| R016 | Decomposed execution model | Design Review | ✅ Done | `marl/environment/execution_model.py` | Queue wait + network + exec time |
| R017 | Energy model | Design Review | ✅ Done | `marl/environment/energy_model.py` | energy = power × duration |
| R018 | Carbon emission model | Design Review | 🧪 Experimental | `marl/environment/energy_model.py` | carbon = energy × intensity |
| R019 | SLA tracking | Design Review | ✅ Done | `marl/environment/sla.py` | violation = completion > deadline |
| R020 | Explicit work_units field | Design Review | ✅ Done | `marl/environment/task.py` | Not inferred from cpu_req |
| R021 | Network topology | Project PPT | ✅ Done | `marl/environment/network.py` | Line, grid, full, random, custom |
| R022 | Reproducibility (seeding) | Project PPT | ✅ Done | `marl/utils/seeding.py` | Python, NumPy, PyTorch |
| R023 | Checkpointing | Project PPT | ✅ Done | `marl/utils/checkpoint.py` | Best, latest, numbered |
| R024 | Experiment logging | Project PPT | ✅ Done | `marl/utils/logger.py` | CSV, JSON, TensorBoard |
| R025 | Raw physical metrics evaluation | Design Review | ✅ Done | `marl/evaluation/metrics.py` | 16 KPIs |
| R026 | Fair baseline comparison | Design Review | ✅ Done | `marl/evaluation/evaluator.py` | Same env, same workload |
| R027 | Parameter sharing (configurable) | Project PPT | ✅ Done | `configs/marl.yaml` | Shared or individual actors |
| R028 | Local/global reward blending | Design Review | ✅ Done | `marl/reward/reward_engine.py` | R = α·R_local + (1-α)·R_global |
| R029 | Constrained dynamic weights | Design Review | 🧪 Experimental | `marl/reward/dynamic_weight_selector.py` | wi≥0, Σwi=1, min/max bounds |
| R030 | System invariants testing | Design Review | ✅ Done | `tests/test_invariants.py` | No over-allocation, valid lifecycle |

## Status Legend

| Symbol | Meaning |
|--------|--------|
| ✅ Done | Working, tested, documented |
| 🧪 Experimental | Functional but not validated as research contribution |
| 📋 Placeholder | Interface only, awaiting finalized design |
| 🔲 Future Work | Not yet built |
| 🔄 In Progress | Currently being implemented |
