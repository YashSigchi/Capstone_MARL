import pytest
from marl.environment.marl_env import MARLResourceAllocationEnv
from marl.environment.task import TaskRepresentation


def _make_task(task_id='t1', cpu=1, mem=1, gpu=0, work=10, deadline=50,
               priority=1, data_size=1, comm=0, gpu_req=None):
    """Helper to create a task with all required fields."""
    return TaskRepresentation(
        task_id=task_id,
        cpu_req=cpu,
        memory_req=mem,
        gpu_req=gpu_req if gpu_req is not None else gpu,
        work_units=work,
        deadline=deadline,
        priority=priority,
        data_size=data_size,
        communication_req=comm,
        dependencies=[],
        arrival_time=0.0,
        sla_class='silver',
    )


def test_mask_always_allows_defer(small_config):
    env = MARLResourceAllocationEnv(small_config)
    env.reset(seed=42)
    mask = env.get_action_mask(0)
    assert mask[-1] == 1.0 and mask[-2] == 1.0


def test_mask_blocks_overloaded_resource(small_config):
    env = MARLResourceAllocationEnv(small_config)
    env.reset(seed=42)
    # Create a task that exceeds all resource capacities
    task = _make_task(cpu=1000, mem=1000, gpu=100)
    env.queues[0].clear()
    env.queues[0].enqueue(task)
    mask = env.get_action_mask(0)
    # All resource actions should be blocked, only defer allowed
    assert sum(mask[:-2]) == 0.0
    assert mask[-1] == 1.0 and mask[-2] == 1.0


def test_mask_blocks_inactive_resource(small_config):
    env = MARLResourceAllocationEnv(small_config)
    env.reset(seed=42)
    env.resources['R1'].active = False
    task = _make_task(cpu=1, mem=1)
    env.queues[0].clear()
    env.queues[0].enqueue(task)
    mask = env.get_action_mask(0)
    assert mask[0] == 0.0  # R1 is inactive


def test_mask_allows_feasible_resource(small_config):
    env = MARLResourceAllocationEnv(small_config)
    env.reset(seed=42)
    task = _make_task(cpu=1, mem=1)
    env.queues[0].clear()
    env.queues[0].enqueue(task)
    mask = env.get_action_mask(0)
    assert mask[0] == 1.0  # R1 should be feasible


def test_mask_gpu_requirement(small_config):
    env = MARLResourceAllocationEnv(small_config)
    env.reset(seed=42)
    # R1 has GPU=1, R2 has GPU=0. Agent 0 has [R1, R2].
    task = _make_task(cpu=1, mem=1, gpu_req=1)
    env.queues[0].clear()
    env.queues[0].enqueue(task)
    mask = env.get_action_mask(0)
    assert mask[0] == 1.0  # R1 has GPU
    assert mask[1] == 0.0  # R2 has no GPU


def test_mask_no_task(small_config):
    env = MARLResourceAllocationEnv(small_config)
    env.reset(seed=42)
    env.queues[0].clear()
    mask = env.get_action_mask(0)
    # No task: only defer is valid
    assert mask[-1] == 1.0 and mask[-2] == 1.0
    # Resource actions should be 0 when no task
    assert sum(mask[:-2]) == 0.0


def test_mask_consistency(small_config):
    env = MARLResourceAllocationEnv(small_config)
    env.reset(seed=42)
    mask = env.get_action_mask(0)
    assert len(mask) == env.get_action_dim(0)
