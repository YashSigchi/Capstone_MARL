"""Evaluation module: runs episodes and collects raw physical/system metrics.

Status: Implemented
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from marl.baselines.base_scheduler import Scheduler
from marl.evaluation.metrics import MetricsTracker, EpisodeMetrics


class Evaluator:
    """Runs evaluation episodes and aggregates raw physical metrics.

    Supports evaluating both MARL agents and baseline schedulers
    on the identical environment.

    Status: Implemented
    """

    def __init__(self, env: Any,
                 reward_engine: Optional[Any] = None,
                 seed: Optional[int] = None):
        self.env = env
        self.reward_engine = reward_engine
        self.seed = seed
        self._episode_run_counter: int = 0
        self.metrics_tracker = MetricsTracker()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_agents(self, agents: Dict[int, Any],
                        num_episodes: int = 10,
                        deterministic: bool = True) -> Dict[str, Any]:
        """Evaluate MARL agents. Returns aggregated metrics."""
        self.metrics_tracker.reset()
        self._episode_run_counter = 0

        def action_fn(env_ref, obs_dict):
            actions = {}
            for agent_id, agent in agents.items():
                obs = obs_dict[agent_id]
                mask = env_ref.get_action_mask(agent_id)
                if deterministic and hasattr(agent, 'select_action_deterministic'):
                    action = agent.select_action_deterministic(obs, mask)
                else:
                    action, _, _ = agent.select_action(obs, mask)
                actions[agent_id] = action
            return actions

        for _ in range(num_episodes):
            self._run_episode(action_fn)

        return self.metrics_tracker.get_summary()

    def evaluate_scheduler(self, scheduler: Scheduler,
                           num_episodes: int = 10) -> Dict[str, Any]:
        """Evaluate a baseline scheduler. Returns aggregated metrics."""
        self.metrics_tracker.reset()
        self._episode_run_counter = 0

        def scheduler_action_fn(env_ref, obs_dict):
            return scheduler.select_actions(env_ref)

        for _ in range(num_episodes):
            scheduler.reset()
            self._run_episode(scheduler_action_fn)

        return self.metrics_tracker.get_summary()

    def compare(self, results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        """Compare multiple algorithms side-by-side."""
        df = pd.DataFrame(results).T
        return df

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _extract_step_metrics(self, infos: Dict[int, dict],
                              rewards: Dict[int, float]) -> dict:
        """Extract MetricsTracker-compatible step_info from env infos.

        The env's infos dict is keyed by agent_id, each value containing
        a 'reward_info' sub-dict with fields like:
            execution_time, network_time, queue_waiting_time,
            energy_consumed, carbon_emitted, sla_violated, etc.
        We aggregate across all agents for this step.
        """
        step_latencies: List[float] = []
        step_queue_waits: List[float] = []
        step_utilization: List[float] = []
        step_energy = 0.0
        step_carbon = 0.0
        step_completed = 0
        step_rejected = 0
        step_new_tasks = 0
        step_reward = sum(rewards.values()) if isinstance(rewards, dict) else rewards

        for aid, agent_info in infos.items():
            if not isinstance(agent_info, dict):
                continue
            ri = agent_info.get('reward_info', agent_info)

            # Latency: total_completion_time for tasks that were allocated
            comp_time = ri.get('total_completion_time', 0.0)
            if comp_time > 0 and not ri.get('action_was_defer', True):
                step_latencies.append(comp_time)

            # Queue waiting time
            qwt = ri.get('queue_waiting_time', 0.0)
            if qwt > 0:
                step_queue_waits.append(qwt)
            # Also capture local average queue wait
            local_qwt = ri.get('local_avg_waiting_time', 0.0)
            if local_qwt > 0 and qwt == 0:
                step_queue_waits.append(local_qwt)

            # Utilization — all_utilizations is a list of per-resource utils
            all_utils = ri.get('all_utilizations', [])
            all_resources = ri.get('all_resources', [])
            if all_utils and not step_utilization:
                # Only capture once per step (same across agents)
                step_utilization = [float(u) for u in all_utils]
                # Also capture individual dimensions if available
                if all_resources:
                    step_cpu_utilization = [r.current_cpu_usage / max(r.cpu_capacity, 1e-6) for r in all_resources]
                    step_memory_utilization = [r.current_memory_usage / max(r.memory_capacity, 1e-6) for r in all_resources]

            # Energy & carbon
            step_energy += ri.get('energy_consumed', 0.0)
            step_carbon += ri.get('carbon_emitted', 0.0)

        result: dict = {'reward': step_reward}

        if step_latencies:
            result['latency'] = step_latencies
        if step_queue_waits:
            result['queue_wait'] = step_queue_waits
        if step_utilization:
            result['utilization'] = step_utilization
            result['cpu_utilization'] = step_cpu_utilization
            result['memory_utilization'] = step_memory_utilization

        result['energy'] = step_energy
        result['carbon'] = step_carbon
        result['completed_tasks'] = step_completed
        result['rejected_tasks'] = step_rejected
        result['new_tasks'] = step_new_tasks

        return result

    def _extract_episode_end_metrics(self, episode_metrics: EpisodeMetrics) -> None:
        """Patch episode metrics with data queried directly from the env
        at the end of the episode (SLA, utilization, energy totals, etc.)."""
        env = self.env

        # --- Completed / rejected / total tasks ---
        completed = getattr(env, '_completed_tasks', [])
        rejected = getattr(env, '_rejected_tasks', [])
        all_tasks = getattr(env, '_all_tasks', [])

        total = max(len(all_tasks), 1)
        n_completed = len(completed)
        n_rejected = len(rejected)

        episode_metrics.task_completion_rate = n_completed / total
        episode_metrics.task_rejection_rate = n_rejected / total
        episode_metrics.throughput = (
            n_completed / episode_metrics.makespan
            if episode_metrics.makespan > 0 else 0.0
        )

        # --- Task latencies ---
        latencies: List[float] = []
        for task in completed:
            if task.completion_time is not None and task.start_time is not None:
                latencies.append(task.completion_time - task.arrival_time)

        if latencies:
            episode_metrics.avg_task_latency = float(np.mean(latencies))
            episode_metrics.p95_latency = float(np.percentile(latencies, 95))
            episode_metrics.p99_latency = float(np.percentile(latencies, 99))

        # --- Queue waiting times ---
        queue_waits: List[float] = []
        for task in completed:
            if task.start_time is not None:
                queue_waits.append(task.start_time - task.arrival_time)
        if queue_waits:
            episode_metrics.avg_queue_waiting_time = float(np.mean(queue_waits))

        # --- Resource utilization ---
        # Only fall back to end-of-episode point-in-time utilizations if the
        # step tracker did not capture any per-step utilization samples.
        # Point-in-time at end-of-episode is misleading: if all tasks finished,
        # current_usage is zero even though the episode had active utilization.
        step_avg_cpu = getattr(episode_metrics, 'avg_cpu_utilization', 0.0)
        step_avg_mem = getattr(episode_metrics, 'avg_memory_utilization', 0.0)
        step_imbalance = getattr(episode_metrics, 'resource_imbalance', 0.0)
        if step_avg_cpu == 0.0 and step_avg_mem == 0.0 and step_imbalance == 0.0:
            resources = getattr(env, 'resources', {})
            if resources:
                utils = [r.utilization() for r in resources.values()]
                cpu_utils = [
                    r.current_cpu_usage / max(r.cpu_capacity, 1e-6)
                    for r in resources.values()
                ]
                mem_utils = [
                    r.current_memory_usage / max(r.memory_capacity, 1e-6)
                    for r in resources.values()
                ]
                episode_metrics.avg_cpu_utilization = float(np.mean(cpu_utils))
                episode_metrics.avg_memory_utilization = float(np.mean(mem_utils))
                episode_metrics.resource_imbalance = float(np.std(utils))

        # --- Energy & carbon ---
        episode_metrics.total_energy = float(getattr(env, '_total_energy', 0.0))
        episode_metrics.total_carbon = float(getattr(env, '_total_carbon', 0.0))
        # --- SLA ---
        sla_tracker = getattr(env, 'sla_tracker', None)
        if sla_tracker is not None:
            records = getattr(sla_tracker, 'records', [])
            if records:
                n_violations = sum(1 for r in records if r.violated)
                episode_metrics.sla_violation_rate = n_violations / max(len(records), 1)
                high_prio_violations = sum(
                    1 for r in records
                    if r.violated and getattr(r, 'priority', 0) >= 4
                )
                episode_metrics.high_priority_sla_violations = float(high_prio_violations)
                slacks = [r.slack for r in records if hasattr(r, 'slack')]
                if slacks:
                    episode_metrics.avg_deadline_slack = float(np.mean(slacks))
                    
        # --- New metrics ---
        episode_metrics.makespan = float(getattr(env, '_makespan', 0.0))
        episode_metrics.completed_tasks = float(n_completed)
        episode_metrics.unfinished_tasks = float(sum(q.length for q in env.queues.values()) + len(getattr(env, '_running_tasks', [])))
        if n_completed > 0:
            episode_metrics.energy_per_completed_task = episode_metrics.total_energy / n_completed
            episode_metrics.carbon_per_completed_task = episode_metrics.total_carbon / n_completed
        else:
            episode_metrics.energy_per_completed_task = 0.0
            episode_metrics.carbon_per_completed_task = 0.0



    def _run_episode(self, action_fn) -> EpisodeMetrics:
        """Run one episode using the provided action function."""
        ep_seed = None
        if self.seed is not None:
            ep_seed = self.seed + self._episode_run_counter
            self._episode_run_counter += 1
        obs_dict, infos = self.env.reset(seed=ep_seed)
        done = False
        steps = 0
        ep_reward = 0.0

        while not done:
            actions = action_fn(self.env, obs_dict)
            next_obs, rewards, terminated, truncated, infos = self.env.step(actions)

            # Compute reward via reward engine if available
            if self.reward_engine is not None:
                step_reward = 0.0
                for aid in self.env.agent_ids:
                    reward_info = infos.get(aid, {}).get('reward_info', {})
                    result = self.reward_engine.compute(reward_info)
                    step_reward += result.total
                rewards = {aid: step_reward / len(self.env.agent_ids)
                           for aid in self.env.agent_ids}

            # Extract step-level metrics for the tracker
            step_info = self._extract_step_metrics(infos, rewards)
            self.metrics_tracker.record_step(step_info)

            ep_reward += step_info['reward']

            is_terminated = all(terminated.values()) if isinstance(terminated, dict) else terminated
            is_truncated = all(truncated.values()) if isinstance(truncated, dict) else truncated
            done = is_terminated or is_truncated

            obs_dict = next_obs
            steps += 1

        # Finalize episode in tracker
        episode_metrics = self.metrics_tracker.finalize_episode(steps)

        # Override with accurate end-of-episode metrics from env
        episode_metrics.total_reward = ep_reward
        self._extract_episode_end_metrics(episode_metrics)

        return episode_metrics
