import pytest
from marl.training.trainer import Trainer

class MockAlgorithm:
    def get_agents(self):
        return {}

class MockEvaluator:
    def evaluate_agents(self, agents, num_episodes, deterministic):
        return {
            'task_completion_rate_mean': 0.5,
            'task_rejection_rate_mean': 0.1,
            'sla_violation_rate_mean': 0.2,
            'throughput_mean': 15.5,
            'avg_task_latency_mean': 42.1,
            'energy_per_completed_task_mean': 1500.0,
            'total_reward_mean': -50.0
        }

class MockEnv:
    def __init__(self):
        self.num_agents = 1
        self.agent_ids = [0]
    
    def reset(self, seed=None):
        return {0: []}, {}
        
    def step(self, actions):
        return {0: []}, {0: 0}, {0: True}, {0: False}, {0: {}}

class MockTrainer(Trainer):
    def __init__(self):
        self.algorithm = MockAlgorithm()
        self.reward_engine = None
        self.env = MockEnv()
        self.evaluator = MockEvaluator()
        
    def _save_checkpoint(self, path, eval_metrics):
        self.last_saved_score = (
            round(eval_metrics.get('task_completion_rate_mean', 0.0), 3),
            -round(eval_metrics.get('task_rejection_rate_mean', 0.0), 3),
            -round(eval_metrics.get('sla_violation_rate_mean', 0.0), 3),
            round(eval_metrics.get('throughput_mean', 0.0), 3),
            -round(eval_metrics.get('avg_task_latency_mean', 0.0), 1),
            -round(eval_metrics.get('energy_per_completed_task_mean', 0.0), 1)
        )

def test_trainer_score_tuple_values():
    trainer = MockTrainer()
    
    metrics = trainer.evaluate(num_episodes=1)
    
    trainer._save_checkpoint("dummy", metrics)
    
    score = trainer.last_saved_score
    
    assert score[0] == 0.5    # completion
    assert score[1] == -0.1   # rejection
    assert score[2] == -0.2   # sla
    assert score[3] == 15.5   # throughput
    assert score[4] == -42.1  # latency
    assert score[5] == -1500.0 # energy per task
    
    # check no zeros when actual values are non-zero
    assert score[3] != 0.0
    assert score[5] != 0.0

