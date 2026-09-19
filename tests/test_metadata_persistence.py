import yaml
import torch
import os
from marl.training.trainer import Trainer

class MockAlgorithm:
    def get_agents(self): return {}
    def save(self, path, metadata):
        torch.save({'metadata': metadata}, path)

class MockEvaluator:
    def evaluate_agents(self, agents, num_episodes, deterministic):
        return {
            'task_completion_rate_mean': 0.65,
            'task_rejection_rate_mean': 0.15,
            'sla_violation_rate_mean': 0.05,
            'avg_task_latency_mean': 12.3,
            'energy_per_completed_task_mean': 543.2
        }

class MockEnv:
    def __init__(self):
        self.num_agents = 1
        self.agent_ids = [0]
    
    def reset(self, seed=None):
        return {0: []}, {}
        
    def step(self, actions):
        return {0: []}, {0: 0}, {0: True}, {0: False}, {0: {}}
        
    def get_global_state(self): return [0]
    def get_action_mask(self, aid):
        return [1]

class MockAgent:
    def select_action(self, obs, mask):
        return 0, 0, 0
    def get_value(self, state): return 0.0
    def select_action_deterministic(self, obs, mask):
        return 0

class MockAlgorithmWithAgent(MockAlgorithm):
    def get_agents(self):
        return {0: MockAgent()}
    def update(self, buffer):
        return {'policy_loss': 0.1, 'value_loss': 0.2}

class MockBuffer:
    def reset(self): pass
    def insert(self, **kwargs): pass
    def compute_returns_and_advantages(self, **kwargs): pass
    def get(self): return {}

class MockTrainer(Trainer):
    def __init__(self, save_dir):
        self.config = {
            'checkpoint_dir': save_dir,
            'eval_interval': 1,
            'num_episodes': 1,
            'steps_per_episode': 1
        }
        self.algorithm = MockAlgorithmWithAgent()
        self.reward_engine = None
        self.env = MockEnv()
        self.evaluator = MockEvaluator()
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.agent_ids = [0]
        
        self.num_episodes = 1
        self.steps_per_episode = 1
        self.eval_interval = 1
        self.eval_episodes = 1
        import collections
        self.recent_scores = collections.deque(maxlen=3)
        self.save_interval = 1
        self.logger = None
        self.buffer = MockBuffer()

def test_checkpoint_metadata_match(tmp_path):
    save_dir = str(tmp_path / "checkpoints")
    trainer = MockTrainer(save_dir)
    
    # Run train step to trigger the actual inline evaluation and save
    trainer.train()
    
    best_path = os.path.join(save_dir, 'best_model.pt')
    assert os.path.exists(best_path)
    
    ckpt = torch.load(best_path)
    loaded_metadata = ckpt.get('metadata', {})
    
    assert loaded_metadata.get('validation_completion_rate') == 0.65
    assert loaded_metadata.get('validation_rejection_rate') == 0.15
    assert loaded_metadata.get('validation_sla_violation_rate') == 0.05
    assert loaded_metadata.get('validation_latency') == 12.3
    assert loaded_metadata.get('validation_energy_per_completed_task') == 543.2

