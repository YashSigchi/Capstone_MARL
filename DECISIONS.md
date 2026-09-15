# Decision Log

This document records important design decisions with rationale,
alternatives considered, current status, and potential future changes.

---

### D001: MAPPO as Initial MARL Algorithm

**Decision:** Use Multi-Agent PPO (MAPPO) with Centralized Training,
Decentralized Execution (CTDE) as the initial algorithm.

**Reason:** The environment is cooperative and partially observable.
CTDE is required for scalability. MAPPO has strong empirical performance
in cooperative multi-agent settings and supports parameter sharing.

**Alternatives:** QMIX (value decomposition), MADDPG (actor-critic for
mixed settings), IPPO (independent PPO without centralized critic).

**Status:** Implemented as primary algorithm.

**Future:** Compare against QMIX and MADDPG after environment stabilization.
The BaseMARLAlgorithm interface supports adding new algorithms.

---

### D002: Agent → Cluster → Resource Hierarchy

**Decision:** Agents manage clusters (not individual resources).
Each cluster contains a set of heterogeneous resources.

**Reason:** This provides a natural abstraction layer. Agents reason
about their cluster, and the mapping is configurable. This allows
testing different organizational structures without code changes.

**Alternatives:** Direct agent-to-resource mapping (1:1), flat agent-to-all mapping.

**Status:** Implemented.

**Future:** May support dynamic cluster reassignment.

---

### D003: Environment-Owned Task Queues

**Decision:** Task queues are managed by the environment.
The MARL policy only decides resource allocation, not queue mechanics.

**Reason:** Separates task selection (environment responsibility) from
resource assignment (agent responsibility). This simplifies the action
space and prevents agents from gaming queue ordering.

**Alternatives:** Agent-managed queues (agents choose both which task
and which resource).

**Status:** Implemented.

**Future:** Could add task selection as a hierarchical action later.

---

### D004: Discrete Action Space

**Decision:** Each agent chooses from a discrete set:
{Resource_0, Resource_1, ..., Resource_N, Defer}.

**Reason:** Natural mapping for task-to-resource assignment.
Action masking can efficiently handle infeasible actions.
Simpler to implement and debug than continuous actions.

**Alternatives:** Continuous action space (embedding-based), hierarchical
actions (select cluster then resource).

**Status:** Implemented.

**Future:** Hierarchical actions may be beneficial for large action spaces.

---

### D005: Decomposed Execution Time Model

**Decision:** Separate total completion time into:
queue_waiting_time + network_time + execution_time.

**Reason:** Each component captures a different system aspect.
This enables targeted optimization (e.g., reduce network time by
placing data-heavy tasks near their data source).

**Alternatives:** Single monolithic execution time estimate.

**Status:** Implemented (SimpleExecutionModel).

**Future:** Replace with learned execution predictor or trace-based model.

---

### D006: SLA Violation = completion_time > deadline

**Decision:** A task's SLA is violated if and only if its total
completion time exceeds its deadline.

**Reason:** Simple, measurable, and aligns with real-world SLA contracts.
Deadline slack (deadline - completion_time) provides a continuous signal.

**Alternatives:** Probabilistic SLA (P(completion < deadline) > threshold),
tiered SLA with different penalty levels.

**Status:** Implemented.

**Future:** May add tiered penalties based on SLA class (gold/silver/bronze).

---

### D007: Running Mean/Std Reward Normalization

**Decision:** Use Welford's online algorithm for per-component
reward normalization.

**Reason:** Adapts to changing reward distributions during training.
Prevents any single component from dominating due to scale differences.

**Alternatives:** Min-max normalization, fixed range normalization.

**Status:** Implemented (default). Other methods also available.

**Future:** May need tuning if reward distributions are highly non-stationary.

---

### D008: Constrained Dynamic Reward Weights

**Decision:** Dynamic weights must satisfy:
w_i ≥ 0, Σw_i = 1, min_i ≤ w_i ≤ max_i.

**Reason:** Ensures critical objectives (e.g., SLA with min=0.10)
cannot be effectively disabled by the dynamic selector.

**Alternatives:** Unconstrained weights, learned weight network.

**Status:** Experimental.

**Future:** Validate whether dynamic weights improve generalization.

---

### D009: PyTorch as Deep Learning Framework

**Decision:** Use PyTorch for all neural network components.

**Reason:** Team familiarity, strong research community support,
dynamic computation graphs for debugging.

**Alternatives:** JAX (faster compilation), TensorFlow.

**Status:** Implemented.

**Future:** No planned change.

---

### D010: YAML Configuration System

**Decision:** Use 4 YAML config files: environment, marl, reward, experiment.

**Reason:** Human-readable, supports nested structures, widely used in ML.
Separating configs by concern allows independent modification.

**Alternatives:** JSON, TOML, Python dataclasses, Hydra.

**Status:** Implemented.

**Future:** May migrate to Hydra for more complex experiment sweeps.
