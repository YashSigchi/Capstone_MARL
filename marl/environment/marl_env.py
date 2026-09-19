"""Main MARL Resource Allocation Environment.

This module implements the central multi-agent environment for resource
allocation and task scheduling. It follows a standard multi-agent API
with reset(), step(), observation/action spaces, and separate terminated
and truncated signals.

Decision model per timestep:
    1. Environment generates/presents schedulable tasks to agents
    2. Agents observe local state + task features
    3. Agents choose: eligible resource index OR defer/reject
    4. Environment executes allocations, advances running tasks
    5. Rewards are computed externally via RewardEngine

Status: Implemented
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from marl.environment.cluster import Cluster
from marl.environment.energy_model import EnergyModel
from marl.environment.execution_model import ExecutionModel, SimpleExecutionModel
from marl.environment.network import NetworkTopology
from marl.environment.resource import ResourceState
from marl.environment.sla import SLATracker
from marl.environment.task import TaskGenerator, TaskRepresentation
from marl.environment.task_queue import TaskQueue


# ---------------------------------------------------------------------------
# Running-task bookkeeping
# ---------------------------------------------------------------------------

class _RunningTask:
    """Internal tracker for a task currently executing on a resource."""

    __slots__ = (
        "task", "resource_id", "remaining_steps",
        "exec_time", "network_time", "queue_wait_time",
    )

    def __init__(
        self,
        task: TaskRepresentation,
        resource_id: str,
        remaining_steps: int,
        exec_time: float,
        network_time: float,
        queue_wait_time: float,
    ) -> None:
        self.task = task
        self.resource_id = resource_id
        self.remaining_steps = remaining_steps
        self.exec_time = exec_time
        self.network_time = network_time
        self.queue_wait_time = queue_wait_time


# ---------------------------------------------------------------------------
# Main Environment
# ---------------------------------------------------------------------------

class MARLResourceAllocationEnv:
    """Multi-agent environment for heterogeneous resource allocation.

    Hierarchy:  Agent → Cluster → Resources

    At each timestep the environment:
        1. Generates arriving tasks (Poisson / deterministic / burst).
        2. Distributes tasks to per-agent queues (round-robin or
           least-loaded).
        3. Presents the next schedulable task to each agent.
        4. Agents choose an action (resource index or defer).
        5. Valid allocations start task execution; deferred tasks remain
           in the queue.
        6. Running tasks advance; completed tasks release resources.

    The environment is *not* a ``gym.Env`` subclass because the standard
    single-agent API does not fit multi-agent control.  It mirrors the
    PettingZoo-style ``reset() → step()`` contract with per-agent dicts.

    Status: Implemented
    """

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #

    def __init__(self, config: dict) -> None:
        """Initialise the environment from a configuration dict.

        Args:
            config: Merged configuration containing *at least* the keys
                defined in ``configs/environment.yaml`` plus agent mapping
                from ``configs/marl.yaml``.
        """
        self.config = config

        # ---- Resources ------------------------------------------------
        self.resources: Dict[str, ResourceState] = {}
        for r_cfg in config["resources"]:
            rid = r_cfg["id"]
            self.resources[rid] = ResourceState.from_config(r_cfg)
        self.resource_ids: List[str] = list(self.resources.keys())
        self.num_resources: int = len(self.resource_ids)

        # ---- Clusters -------------------------------------------------
        cluster_cfgs = config.get("clusters", [])
        self.clusters: Dict[str, Cluster] = {}
        for c_cfg in cluster_cfgs:
            cid = c_cfg["id"]
            self.clusters[cid] = Cluster.from_config(c_cfg)

        # ---- Agents ---------------------------------------------------
        agent_mapping = config.get("agent_mapping", {})
        self.num_agents: int = config.get("num_agents", len(agent_mapping))
        # Map agent_id (int) → cluster_id (str)
        self.agent_cluster_map: Dict[int, str] = {}
        for agent_key, cluster_id in agent_mapping.items():
            aid = int(agent_key)
            self.agent_cluster_map[aid] = cluster_id
            # Ensure cluster has correct agent_id
            if cluster_id in self.clusters:
                self.clusters[cluster_id].agent_id = aid
        self.agent_ids: List[int] = sorted(self.agent_cluster_map.keys())

        # ---- Topology -------------------------------------------------
        topo_cfg = config.get("topology", {})
        topo_type = topo_cfg.get("type", "fully_connected") if isinstance(topo_cfg, dict) else topo_cfg
        topo_kwargs: dict = {}
        if isinstance(topo_cfg, dict):
            topo_kwargs["default_bandwidth"] = topo_cfg.get("default_bandwidth", 10.0)
            topo_kwargs["default_latency"] = topo_cfg.get("default_latency", 1.0)
        self.topology: NetworkTopology = NetworkTopology.from_type(
            topo_type, self.resource_ids, **topo_kwargs
        )

        # ---- Execution & Energy models ---------------------------------
        self.execution_model: ExecutionModel = SimpleExecutionModel()
        carbon_intensity = config.get("carbon_intensity", 0.5)
        self.energy_model: EnergyModel = EnergyModel(carbon_intensity=carbon_intensity)

        # ---- SLA -------------------------------------------------------
        self.sla_tracker: SLATracker = SLATracker()

        # ---- Task generation -------------------------------------------
        task_cfg = config.get("task_distributions", {})
        arrival_cfg = config.get("task_arrival", {})
        self._arrival_model: str = arrival_cfg.get("model", "poisson") if isinstance(arrival_cfg, dict) else "poisson"
        self._arrival_rate: float = arrival_cfg.get("rate", 3.0) if isinstance(arrival_cfg, dict) else float(arrival_cfg if arrival_cfg else 3.0)
        self._rng: np.random.Generator = np.random.default_rng()
        self._task_generator: TaskGenerator = TaskGenerator(task_cfg, self._rng)
        self._task_counter: int = 0
        self._max_tasks: int = config.get("max_tasks_per_episode", 200)

        # ---- Queues (environment-owned, one per agent) -----------------
        queue_ordering = config.get("queue_ordering", "fifo")
        self.queues: Dict[int, TaskQueue] = {
            aid: TaskQueue(ordering=queue_ordering)
            for aid in self.agent_ids
        }

        # ---- Episode bookkeeping ---------------------------------------
        self._episode_length: int = config.get("episode_length", 200)
        self._current_step: int = 0
        self._running_tasks: List[_RunningTask] = []
        self._completed_tasks: List[TaskRepresentation] = []
        self._rejected_tasks: List[TaskRepresentation] = []
        self._all_tasks: List[TaskRepresentation] = []
        
        # ---- Rejection Control -----------------------------------------
        rej_cfg = config.get("rejection", {})
        self._rejection_enabled = rej_cfg.get("enabled", True)
        self._max_rejection_rate = rej_cfg.get("max_rejection_rate", 1.0)
        self._rejected_task_records = []

        # ---- Track per-step info for reward computation ----------------
        self._step_info: Dict[int, dict] = {}
        # Per-step cumulative energy
        self._total_energy: float = 0.0
        self._total_carbon: float = 0.0

        # ---- Observation / action dims (computed once) -----------------
        self._obs_dim: Optional[int] = None
        self._global_state_dim: Optional[int] = None

    # ------------------------------------------------------------------ #
    # Space accessors
    # ------------------------------------------------------------------ #

    def observation_space(self, agent_id: int) -> spaces.Box:
        """Return the observation space for the given agent."""
        dim = self._get_obs_dim()
        return spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)

    def action_space(self, agent_id: int) -> spaces.Discrete:
        """Return the action space for the given agent.

        Actions 0..N-1 correspond to the N resources in the agent's
        cluster.  Action N is *defer / reject*.
        """
        cluster_id = self.agent_cluster_map[agent_id]
        n_resources = self.clusters[cluster_id].num_resources
        return spaces.Discrete(n_resources + 2)  # +1 for defer, +1 for reject

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #

    def reset(
        self,
        seed: Optional[int] = None,
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, dict]]:
        """Reset the environment to the start of a new episode.

        Args:
            seed: Optional random seed.

        Returns:
            observations: ``{agent_id: obs_vector}``
            infos: ``{agent_id: info_dict}``
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            self._task_generator = TaskGenerator(
                self.config.get("task_distributions", {}), self._rng
            )

        # Reset resources
        for r in self.resources.values():
            r.reset()

        # Reset queues
        for q in self.queues.values():
            q.clear()

        # Reset bookkeeping
        self._current_step = 0
        self._task_counter = 0
        self._running_tasks = []
        self._completed_tasks = []
        self._rejected_tasks = []
        self._all_tasks = []
        self._rejected_task_records = []
        self._step_info = {}
        self._total_energy = 0.0
        self._total_carbon = 0.0
        self._makespan = 0.0
        self.sla_tracker.reset()

        # Generate initial tasks
        self._generate_and_distribute_tasks()

        # Build observations
        observations: Dict[int, np.ndarray] = {}
        infos: Dict[int, dict] = {}
        for aid in self.agent_ids:
            observations[aid] = self.get_local_observation(aid)
            infos[aid] = {"action_mask": self.get_action_mask(aid)}

        return observations, infos

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #

    def step(
        self,
        actions: Dict[int, int],
    ) -> Tuple[
        Dict[int, np.ndarray],   # observations
        Dict[int, float],        # rewards (placeholder zeros)
        Dict[int, bool],         # terminated
        Dict[int, bool],         # truncated
        Dict[int, dict],         # infos
    ]:
        """Execute one environment step.

        The environment does **not** compute rewards internally.  It
        populates ``infos[agent_id]["reward_info"]`` with the data
        needed by :class:`RewardEngine` so the caller (e.g.
        :class:`Trainer`) can compute them externally.

        Args:
            actions: ``{agent_id: action_index}``

        Returns:
            observations, rewards (zeros), terminated, truncated, infos
        """
        self._step_info = {}
        step_energy = 0.0
        step_carbon = 0.0

        # ---- 1. Process agent actions ---------------------------------
        for aid in self.agent_ids:
            action = actions.get(aid, self._get_defer_action(aid))
            self._process_action(aid, action)

        # ---- 2. Advance running tasks ---------------------------------
        still_running: List[_RunningTask] = []
        for rt in self._running_tasks:
            rt.remaining_steps -= 1
            if rt.remaining_steps <= 0:
                # Task completed
                completion_time = float(self._current_step + 1)
                rt.task.completion_time = completion_time
                rt.task.status = "completed"
                self._completed_tasks.append(rt.task)
                self._makespan = max(self._makespan, completion_time)

                # Release resources
                resource = self.resources[rt.resource_id]
                resource.release(rt.task)

                # SLA
                self.sla_tracker.record(rt.task, completion_time)

                # Energy
                energy = self.energy_model.compute_energy(resource, rt.exec_time)
                carbon = self.energy_model.compute_carbon(energy)
                step_energy += energy
                step_carbon += carbon
            else:
                still_running.append(rt)
        self._running_tasks = still_running

        self._total_energy += step_energy
        self._total_carbon += step_carbon

        # ---- 3. Advance timestep --------------------------------------
        self._current_step += 1

        # ---- 4. Generate new tasks ------------------------------------
        if self._task_counter < self._max_tasks:
            self._generate_and_distribute_tasks()

        # ---- 5. Build per-agent reward info ---------------------------
        all_utils = [r.utilization() for r in self.resources.values()]
        sys_avg_util = float(np.mean(all_utils)) if all_utils else 0.0

        for aid in self.agent_ids:
            info = self._step_info.get(aid, self._default_step_info(aid))
            # Augment with global data
            info["all_resources"] = list(self.resources.values())
            info["all_utilizations"] = all_utils
            info["system_avg_utilization"] = sys_avg_util
            info["total_completed_tasks"] = len(self._completed_tasks)
            info["total_tasks_this_step"] = max(1, len(self._all_tasks))
            info["total_energy"] = self._total_energy
            info["total_carbon"] = self._total_carbon
            info["total_sla_violations"] = len(
                [r for r in self.sla_tracker.records if r.violated]
            )
            info["total_tasks_processed"] = (
                len(self._completed_tasks) + len(self._rejected_tasks)
            )
            
            info["current_rejection_rate"] = len(self._rejected_tasks) / max(1, self._task_counter)
            info["max_rejection_rate"] = getattr(self, "_max_rejection_rate", 1.0)
            info["target_rejection_rate"] = self.config.get("reward", {}).get("admission", {}).get("target_rejection_rate", 0.0)
            info["admission_base_penalty"] = self.config.get("reward", {}).get("admission", {}).get("base_penalty", 5.0)
            info["admission_excess_penalty_scale"] = self.config.get("reward", {}).get("admission", {}).get("excess_penalty_scale", 10.0)
            info["rejection_enabled"] = getattr(self, "_rejection_enabled", True)
            
            # True makespan semantics: if there are unfinished tasks, the true makespan 
            # is at least the current step. If all are done, it's the exact time the last task finished.
            unfinished = sum(q.length for q in self.queues.values()) + len(self._running_tasks)
            info["unfinished_tasks"] = unfinished
            info["makespan"] = self._makespan if unfinished == 0 else float(self._current_step)

            self._step_info[aid] = info

        # ---- 6. Check termination / truncation -----------------------
        all_done = (
            len(self._completed_tasks) + len(self._rejected_tasks) >= len(self._all_tasks)
            and self._task_counter >= self._max_tasks
            and all(q.is_empty for q in self.queues.values())
            and len(self._running_tasks) == 0
        )
        time_up = self._current_step >= self._episode_length

        terminated: Dict[int, bool] = {aid: all_done for aid in self.agent_ids}
        truncated: Dict[int, bool] = {aid: time_up and not all_done for aid in self.agent_ids}

        # ---- 7. Build observations ------------------------------------
        observations: Dict[int, np.ndarray] = {}
        rewards: Dict[int, float] = {}
        infos: Dict[int, dict] = {}
        for aid in self.agent_ids:
            observations[aid] = self.get_local_observation(aid)
            rewards[aid] = 0.0  # computed externally by RewardEngine
            infos[aid] = {
                "action_mask": self.get_action_mask(aid),
                "reward_info": self._step_info.get(aid, self._default_step_info(aid)),
            }

        return observations, rewards, terminated, truncated, infos

    # ------------------------------------------------------------------ #
    # Observations
    # ------------------------------------------------------------------ #

    def get_local_observation(self, agent_id: int) -> np.ndarray:
        """Build the local observation vector for one agent.

        Components (normalised where practical):
            - Per-resource in cluster: utilisation, available CPU/mem/GPU, est exec time
              (flattened, zero-padded to max cluster size)
            - Queue length (normalised by 20)
            - Average waiting time (normalised by 50)
            - Current task features (zeros if no task)
        """
        cluster = self._get_cluster(agent_id)
        resources = cluster.get_resources(self.resources)
        task = self.get_current_task(agent_id)

        obs_parts: List[float] = []

        # Per-resource features (5 per resource)
        max_resources = max(c.num_resources for c in self.clusters.values())
        for i in range(max_resources):
            if i < len(resources):
                r = resources[i]
                
                # Compute estimated execution time if task is present and feasible
                est_time = 0.0
                if task is not None and r.can_execute(task):
                    est_time = self.execution_model.compute_execution_time(task, r) / 100.0
                
                obs_parts.extend([
                    r.utilization(),
                    r.available_cpu() / max(r.cpu_capacity, 1.0),
                    r.available_memory() / max(r.memory_capacity, 1.0),
                    r.available_gpu() / max(r.gpu_capacity, 1.0) if r.gpu_capacity > 0 else 0.0,
                    est_time
                ])
            else:
                obs_parts.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        # Queue info
        queue = self.queues[agent_id]
        obs_parts.append(queue.length / 20.0)
        obs_parts.append(queue.average_waiting_time(float(self._current_step)) / 50.0)

        # Current task features
        if task is not None:
            slack = task.deadline - float(self._current_step)
            obs_parts.extend([
                task.cpu_req / 64.0,
                task.memory_req / 128.0,
                task.gpu_req / 4.0,
                task.work_units / 100.0,
                task.deadline / 50.0,
                task.priority / 5.0,
                task.data_size / 100.0,
                task.get_waiting_time(float(self._current_step)) / 50.0,
                slack / 50.0, # current slack
                1.0,  # has_task flag
            ])
        else:
            obs_parts.extend([0.0] * 10)

        return np.array(obs_parts, dtype=np.float32)

    def get_global_state(self) -> np.ndarray:
        """Build the global state vector for the centralized critic.

        Concatenates:
            - All agent local observations
            - Global resource utilisation stats
            - Global queue stats
            - SLA summary
        """
        parts: List[float] = []

        # All local observations
        for aid in self.agent_ids:
            parts.extend(self.get_local_observation(aid).tolist())

        # Global resource stats
        utils = [r.utilization() for r in self.resources.values()]
        parts.append(float(np.mean(utils)) if utils else 0.0)
        parts.append(float(np.std(utils)) if utils else 0.0)
        parts.append(float(np.max(utils)) if utils else 0.0)
        parts.append(float(np.min(utils)) if utils else 0.0)

        # Global queue stats
        total_queue = sum(q.length for q in self.queues.values())
        parts.append(total_queue / (20.0 * self.num_agents))

        # Running tasks
        parts.append(len(self._running_tasks) / max(self.num_resources, 1))

        # Completion progress
        total_tasks = max(len(self._all_tasks), 1)
        parts.append(len(self._completed_tasks) / total_tasks)
        parts.append(len(self._rejected_tasks) / total_tasks)

        # SLA
        parts.append(self.sla_tracker.get_violation_rate())

        # Time progress
        parts.append(self._current_step / max(self._episode_length, 1))

        return np.array(parts, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Action mask
    # ------------------------------------------------------------------ #

    def get_action_mask(self, agent_id: int) -> np.ndarray:
        """Return a binary mask over the agent's action space.

        ``mask[i] = 1`` if action *i* is valid, ``0`` otherwise.
        The last action (defer) is always valid.
        """
        cluster = self._get_cluster(agent_id)
        resources = cluster.get_resources(self.resources)
        n_actions = len(resources) + 2  # +1 for defer, +1 for reject

        mask = np.zeros(n_actions, dtype=np.float32)
        task = self.get_current_task(agent_id)

        if task is not None:
            for i, r in enumerate(resources):
                if r.can_execute(task):
                    mask[i] = 1.0
                    
        # DEFER is always valid
        mask[-2] = 1.0
        
        # Enforce rejection control
        if not getattr(self, "_rejection_enabled", True):
            mask[-1] = 0.0
        else:
            total_arrived = max(1, self._task_counter)
            current_rej_rate = len(self._rejected_tasks) / total_arrived
            max_rate = getattr(self, "_max_rejection_rate", 1.0)
            if current_rej_rate >= max_rate:
                mask[-1] = 0.0
            else:
                mask[-1] = 1.0
                
        return mask

    # ------------------------------------------------------------------ #
    # Task / resource accessors (used by baselines)
    # ------------------------------------------------------------------ #

    def get_current_task(self, agent_id: int) -> Optional[TaskRepresentation]:
        """Return the next schedulable task for this agent (peek, no pop)."""
        return self.queues[agent_id].peek()

    def get_agent_resources(self, agent_id: int) -> List[ResourceState]:
        """Return the resources belonging to this agent's cluster."""
        cluster = self._get_cluster(agent_id)
        return cluster.get_resources(self.resources)

    def get_agent_resource_ids(self, agent_id: int) -> List[str]:
        """Return resource IDs for the agent's cluster."""
        cluster = self._get_cluster(agent_id)
        return list(cluster.resource_ids)

    @property
    def current_step(self) -> int:
        """Current simulation timestep."""
        return self._current_step

    @property
    def completed_tasks(self) -> List[TaskRepresentation]:
        return list(self._completed_tasks)

    @property
    def rejected_tasks(self) -> List[TaskRepresentation]:
        return list(self._rejected_tasks)

    @property
    def all_tasks(self) -> List[TaskRepresentation]:
        return list(self._all_tasks)

    # ------------------------------------------------------------------ #
    # Dimension helpers
    # ------------------------------------------------------------------ #

    def _get_obs_dim(self) -> int:
        """Compute observation dimension (cached)."""
        if self._obs_dim is None:
            max_resources = max(c.num_resources for c in self.clusters.values())
            resource_features = max_resources * 5  # added est exec time
            queue_features = 2
            task_features = 10                     # added slack
            self._obs_dim = resource_features + queue_features + task_features
        return self._obs_dim

    def get_obs_dim(self) -> int:
        """Public accessor for observation dimension."""
        return self._get_obs_dim()

    def get_action_dim(self, agent_id: int) -> int:
        """Action dimension for the given agent."""
        cluster = self._get_cluster(agent_id)
        return cluster.num_resources + 2

    def get_global_state_dim(self) -> int:
        """Dimension of the global state vector."""
        if self._global_state_dim is None:
            obs_dim = self._get_obs_dim()
            agent_obs = obs_dim * self.num_agents
            global_extra = 10  # util stats + queue + running + completion + sla + time
            self._global_state_dim = agent_obs + global_extra
        return self._global_state_dim

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _get_cluster(self, agent_id: int) -> Cluster:
        cluster_id = self.agent_cluster_map[agent_id]
        return self.clusters[cluster_id]

    def _get_defer_action(self, agent_id: int) -> int:
        """Return the defer action index for the agent."""
        cluster = self._get_cluster(agent_id)
        return cluster.num_resources  # last action

    def _get_reject_action(self, agent_id: int) -> int:
        """Return the reject action index for the agent."""
        cluster = self._get_cluster(agent_id)
        return cluster.num_resources + 1

    def save_rejection_analysis(self, filepath: str) -> None:
        import json
        with open(filepath, 'w') as f:
            json.dump(self._rejected_task_records, f, indent=2)

    def close(self) -> None:
        pass

    def _generate_and_distribute_tasks(self) -> None:
        """Generate arriving tasks and distribute to agent queues."""
        # Determine number of new tasks
        if self._arrival_model == "poisson":
            n_new = self._rng.poisson(self._arrival_rate)
        elif self._arrival_model == "deterministic":
            n_new = int(self._arrival_rate)
        elif self._arrival_model == "burst":
            burst_interval = self.config.get("task_arrival", {}).get("burst_interval", 20)
            burst_size = self.config.get("task_arrival", {}).get("burst_size", 10)
            if self._current_step % burst_interval == 0:
                n_new = burst_size
            else:
                n_new = max(1, int(self._arrival_rate * 0.3))
        else:
            n_new = int(self._arrival_rate)

        # Cap total tasks
        remaining = self._max_tasks - self._task_counter
        n_new = min(n_new, remaining)
        if n_new <= 0:
            return

        # Generate tasks
        new_tasks: List[TaskRepresentation] = []
        for _ in range(n_new):
            tid = f"task_{self._task_counter}"
            task = self._task_generator.generate_task(tid, float(self._current_step))
            task.status = "queued"
            new_tasks.append(task)
            self._all_tasks.append(task)
            self._task_counter += 1

        # Distribute to queues (round-robin across agents by default)
        for i, task in enumerate(new_tasks):
            target_agent = self.agent_ids[i % self.num_agents]
            self.queues[target_agent].enqueue(task)

    def _process_action(self, agent_id: int, action: int) -> None:
        """Process one agent's action for the current step."""
        cluster = self._get_cluster(agent_id)
        resources = cluster.get_resources(self.resources)
        defer_action = len(resources)
        reject_action = len(resources) + 1
        task = self.queues[agent_id].peek()

        info: Dict[str, Any] = {
            "agent_id": agent_id,
            "action_was_defer": True,
            "action_was_reject": False,
            "allocated_task": None,
            "allocated_resource": None,
            "execution_time": 0.0,
            "network_time": 0.0,
            "queue_waiting_time": 0.0,
            "total_completion_time": 0.0,
            "energy_consumed": 0.0,
            "carbon_emitted": 0.0,
            "sla_violated": False,
            "sla_slack": 0.0,
            "task_priority": 0,
            "local_queue_length": self.queues[agent_id].length,
            "local_avg_waiting_time": self.queues[agent_id].average_waiting_time(
                float(self._current_step)
            ),
            "local_resources": resources,
        }

        feasible_resources = [r for r in resources if r.can_execute(task)] if task else []
        info["feasible_count"] = len(feasible_resources)

        if task is None or action == defer_action:
            # No task or agent chose to defer
            self._step_info[agent_id] = info
            return
            
        if action == reject_action:
            # Agent chose to reject the task
            task = self.queues[agent_id].dequeue()
            task.status = "rejected"
            self._rejected_tasks.append(task)
            
            # Record detailed context for rejection analysis
            feasible_resources = [r for r in resources if r.can_execute(task)]
            feasible_count = len(feasible_resources)
            if feasible_count > 0:
                min_exec = min(self.execution_model.compute_execution_time(task, r) for r in feasible_resources)
            else:
                min_exec = -1.0
                
            self._rejected_task_records.append({
                "task_id": task.task_id,
                "work_units": getattr(task, "work_units", 0.0),
                "cpu_req": task.cpu_req,
                "memory_req": task.memory_req,
                "gpu_req": task.gpu_req,
                "deadline": getattr(task, "deadline", 0.0),
                "priority": task.priority,
                "feasible_resource_count": feasible_count,
                "min_exec_time": min_exec,
                "current_step": self._current_step,
                "queue_length": self.queues[agent_id].length
            })

            info["action_was_defer"] = False
            info["action_was_reject"] = True
            info["allocated_task"] = task
            self._step_info[agent_id] = info
            return

        # Validate action
        if action < 0 or action >= len(resources):
            # Invalid action index → treat as defer
            self._step_info[agent_id] = info
            return

        resource = resources[action]

        if not resource.can_execute(task):
            # Resource cannot handle this task → treat as defer
            self._step_info[agent_id] = info
            return

        # ---- Valid allocation -----------------------------------------
        # Dequeue the task
        self.queues[agent_id].dequeue()

        # Compute times
        queue_wait = self.execution_model.compute_queue_waiting_time(
            task, float(self._current_step)
        )
        # Network time: from a "source" node to the resource
        # We use the first resource in the cluster as a proxy source
        src_id = cluster.resource_ids[0]
        net_time = self.execution_model.compute_network_time(
            task, src_id, resource.resource_id, self.topology
        )
        exec_time = self.execution_model.compute_execution_time(task, resource)
        total_time = queue_wait + net_time + exec_time

        # Allocate on resource
        resource.allocate(task)
        task.assigned_resource_id = resource.resource_id
        task.start_time = float(self._current_step)
        task.status = "running"

        # Compute execution steps (ceiling)
        exec_steps = max(1, int(math.ceil(exec_time)))

        # Track running task
        self._running_tasks.append(_RunningTask(
            task=task,
            resource_id=resource.resource_id,
            remaining_steps=exec_steps,
            exec_time=exec_time,
            network_time=net_time,
            queue_wait_time=queue_wait,
        ))

        # Energy for this allocation
        energy = self.energy_model.compute_energy(resource, exec_time)
        carbon = self.energy_model.compute_carbon(energy)

        # Predicted completion time
        predicted_completion = float(self._current_step) + exec_steps
        sla_violated = self.sla_tracker.check_violation(task, predicted_completion)
        sla_slack = self.sla_tracker.compute_slack(task, predicted_completion)

        info.update({
            "action_was_defer": False,
            "allocated_task": task,
            "allocated_resource": resource,
            "execution_time": exec_time,
            "network_time": net_time,
            "queue_waiting_time": queue_wait,
            "total_completion_time": total_time,
            "energy_consumed": energy,
            "carbon_emitted": carbon,
            "sla_violated": sla_violated,
            "sla_slack": sla_slack,
            "task_priority": task.priority,
        })
        self._step_info[agent_id] = info

    def _default_step_info(self, agent_id: int) -> Dict[str, Any]:
        """Return default (no-op) step info for an agent."""
        cluster = self._get_cluster(agent_id)
        resources = cluster.get_resources(self.resources)
        return {
            "agent_id": agent_id,
            "action_was_defer": True,
            "action_was_reject": False,
            "allocated_task": None,
            "allocated_resource": None,
            "execution_time": 0.0,
            "network_time": 0.0,
            "queue_waiting_time": 0.0,
            "total_completion_time": 0.0,
            "energy_consumed": 0.0,
            "carbon_emitted": 0.0,
            "sla_violated": False,
            "sla_slack": 0.0,
            "task_priority": 0,
            "local_queue_length": self.queues[agent_id].length,
            "local_avg_waiting_time": self.queues[agent_id].average_waiting_time(
                float(self._current_step)
            ),
            "local_resources": resources,
            "all_resources": list(self.resources.values()),
            "all_utilizations": [r.utilization() for r in self.resources.values()],
            "system_avg_utilization": 0.0,
            "total_completed_tasks": len(self._completed_tasks),
            "total_tasks_this_step": max(1, len(self._all_tasks)),
            "total_energy": self._total_energy,
            "total_carbon": self._total_carbon,
            "total_sla_violations": 0,
            "total_tasks_processed": (
                len(self._completed_tasks) + len(self._rejected_tasks)
            ),
            "current_rejection_rate": len(self._rejected_tasks) / max(1, self._task_counter) if getattr(self, "_task_counter", 0) > 0 else 0.0,
            "max_rejection_rate": getattr(self, "_max_rejection_rate", 1.0),
            "rejection_enabled": getattr(self, "_rejection_enabled", True),
            "makespan": getattr(self, "_makespan", 0.0) if (sum(q.length for q in self.queues.values()) + len(self._running_tasks)) == 0 else float(self._current_step),
            "unfinished_tasks": sum(q.length for q in self.queues.values()) + len(self._running_tasks)
        }

    # ------------------------------------------------------------------ #
    # Convenience: create from YAML files
    # ------------------------------------------------------------------ #

    @classmethod
    def from_config_files(
        cls,
        env_config_path: str,
        marl_config_path: str,
    ) -> "MARLResourceAllocationEnv":
        """Construct environment from separate YAML config files.

        Merges environment config with agent mapping from MARL config.
        """
        import yaml  # local import to avoid hard top-level dep
        with open(env_config_path, "r") as f:
            env_cfg = yaml.safe_load(f)
        with open(marl_config_path, "r") as f:
            marl_cfg = yaml.safe_load(f)

        # Merge agent mapping into env config
        env_cfg["agent_mapping"] = marl_cfg.get("agent_mapping", {})
        env_cfg["num_agents"] = marl_cfg.get("num_agents", len(env_cfg["agent_mapping"]))

        return cls(env_cfg)
