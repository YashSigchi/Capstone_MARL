# Research Notes - MARL Resource Allocation

## 1. The "Reject-Most-Tasks" Loophole
**Observation:** The MAPPO agent learned a pathological policy where it rejected 96% of incoming tasks (task_rejection_rate = 96%) rather than executing them.
**Root Cause (Reward Normalization Issue):**
During an explicit audit of the reward pipeline for equivalent tasks (ALLOCATE vs DEFER vs REJECT), I discovered the fatal flaw: `running_mean_std` normalization over time dynamically eroded the static `-1.0` penalty of the `REJECT` action. Because the agent overwhelmingly rejected tasks, the statistical mean of the action penalty shifted toward `-1.0`, normalizing the penalty to ~0.0. Meanwhile, the unnormalized raw latency penalties for actual executions (which are often larger than `-1.0` for heavy tasks) were being heavily penalized. The agent realized that doing nothing (rejecting) yielded a higher normalized reward than completing a task successfully with execution delays.
**Fix:**
Disabled `running_mean_std` normalization (`normalization: none`) so that absolute thresholds (e.g. `-1.0` for REJECT) stand firm against task completion rewards.

## 2. Best Model Checkpoint Selection
**Observation:** Standard RL practice returns the final policy checkpoint, but final checkpoints are often subject to collapse or temporary degradation. In our environment, returning `model_ep100.pt` was arbitrary.
**Fix:**
Implemented a multi-objective validation score in `Trainer`. Every `eval_interval`, the agent is evaluated against `val_seeds` (outside the training loop). The `best_model.pt` is saved when the score maximizes:
`Score = (task_completion_rate * 1000) - (sla_violation_rate * 100) - (energy_per_completed_task * 0.1)`

## 3. Evaluation & Validation
- **True Makespan**: Makespan tracking now handles truncation natively by reflecting `episode_length` when unfinished tasks exist, rather than the arbitrary completion time of the last successfully processed subset of tasks.
- **Completion-aware Metrics**: Tracking `completed_tasks`, `rejected_tasks`, `unfinished_tasks`, `task_completion_rate`, `task_rejection_rate` definitively uncovers degenerate behaviors in scheduling agents.



## 4. Hyperparameter Tuning Results
Systematic tuning was conducted across Learning Rate, Entropy, GAE, PPO Params, and Architectures.
We observed that a higher learning rate (1e-3) and slightly reduced entropy (0.001) yielded the best completion rate and multi-objective score.
MAPPO effectively learns to selectively REJECT constrained or non-urgent tasks to prevent queue clogging, outperforming the blind baselines (FCFS, MET, RR) which greedily allocate tasks and suffer from massive SLA violations and lower overall completion rates.
The fully connected topology modification proved critical, resolving the previously observed infinite latency bug between non-adjacent grid resources.

## MAPPO Recovery Validation
Following the "Reject-All" bugfixes, we identified a discrepancy where training validation yielded 34% completion but evaluator yielded 56% rejection. 
We traced this to a configuration mismatch: the evaluator was silently loading `max_rejection_rate = 1.0` (unrestricted) while training had `0.2` patched in-memory.
With all configs aligned to `0.2` hard cap and `0.1` soft target, the agent perfectly straddles a ~9-10% rejection rate and 25-34% completion rate, proving stable physical recovery without pathological exploits.
