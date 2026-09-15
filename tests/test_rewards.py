import pytest
import numpy as np
from marl.reward.reward_engine import RewardEngine
from marl.reward.reward_components import REWARD_COMPONENTS

def test_latency_reward_range(small_config):
    engine = RewardEngine(small_config)
    info = {'total_completion_time': 10.0, 'sla_violated': False}
    if 'latency' in REWARD_COMPONENTS:
        val = REWARD_COMPONENTS['latency']['fn'](info)
        assert val <= 0.0


def test_energy_reward_negative(small_config):
    info = {'energy_consumed': 50.0}
    if 'energy' in REWARD_COMPONENTS:
        val = REWARD_COMPONENTS['energy']['fn'](info)
        assert val <= 0.0

def test_utilization_reward_optimal_range(small_config):
    info = {'system_avg_utilization': 0.6}
    if 'utilization' in REWARD_COMPONENTS:
        val = REWARD_COMPONENTS['utilization']['fn'](info)
        assert val > 0.0

def test_load_balance_perfect(small_config):
    info = {'all_utilizations': [0.5, 0.5, 0.5, 0.5]}
    if 'load_balance' in REWARD_COMPONENTS:
        val = REWARD_COMPONENTS['load_balance']['fn'](info)
        assert val == 0.0

def test_load_balance_imbalanced(small_config):
    info = {'all_utilizations': [1.0, 0.0, 1.0, 0.0]}
    if 'load_balance' in REWARD_COMPONENTS:
        val = REWARD_COMPONENTS['load_balance']['fn'](info)
        assert val < 0.0

def test_sla_violation_penalty(small_config):
    info = {'sla_violated': True, 'task_priority': 5}
    if 'sla' in REWARD_COMPONENTS:
        val = REWARD_COMPONENTS['sla']['fn'](info)
        assert val < 0.0

def test_sla_met_reward(small_config):
    info = {'sla_violated': False, 'task_priority': 5}
    if 'sla' in REWARD_COMPONENTS:
        val = REWARD_COMPONENTS['sla']['fn'](info)
        assert val >= 0.0

def test_queue_reward_negative(small_config):
    info = {'local_queue_length': 10, 'local_avg_waiting_time': 20.0}
    if 'queue' in REWARD_COMPONENTS:
        val = REWARD_COMPONENTS['queue']['fn'](info)
        assert val <= 0.0

def test_throughput_reward_range(small_config):
    info = {'total_tasks_this_step': 5, 'total_completed_tasks': 2}
    if 'throughput' in REWARD_COMPONENTS:
        val = REWARD_COMPONENTS['throughput']['fn'](info)
        assert 0.0 <= val <= 1.0

def test_reward_normalizer_running_mean(small_config):
    engine = RewardEngine(small_config)
    # Need at least 2 samples for running_mean_std to activate
    engine.normalizer.update('energy', -100.0)
    engine.normalizer.update('energy', -200.0)
    engine.normalizer.update('energy', -50.0)
    norm_val = engine.normalizer.normalize('energy', -100.0)
    # After normalization with running stats, value should differ from raw
    assert norm_val != -100.0

def test_reward_engine_returns_all_components(small_config):
    engine = RewardEngine(small_config)
    info = {'action_was_defer': True}
    res = engine.compute(info)
    assert len(res.components) == len(REWARD_COMPONENTS)

def test_reward_engine_local_global_blend(small_config):
    small_config['reward_blend_alpha'] = 1.0
    engine = RewardEngine(small_config)
    info = {'action_was_defer': True}
    res = engine.compute(info)
    assert res.total == res.local

def test_dynamic_weight_constraints(small_config):
    small_config['mode'] = 'dynamic'
    small_config['dynamic_weights_constraints'] = {'energy': {'min': 0.1, 'max': 0.5}}
    engine = RewardEngine(small_config)
    weights = engine.get_current_weights()
    if 'energy' in weights:
        assert 0.1 <= weights['energy'] <= 0.5
    assert np.isclose(sum(weights.values()), 1.0)

def test_action_penalty_deferred(small_config):
    info = {'action_was_defer': True}
    if 'action_penalty' in REWARD_COMPONENTS:
        val = REWARD_COMPONENTS['action_penalty']['fn'](info)
        assert val == -0.1

def test_action_penalty_reject(small_config):
    info = {'action_was_reject': True}
    if 'action_penalty' in REWARD_COMPONENTS:
        val = REWARD_COMPONENTS['action_penalty']['fn'](info)
        assert val == -1.0
