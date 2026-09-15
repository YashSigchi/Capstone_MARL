import os
import pytest
import yaml
from marl.environment.marl_env import MARLResourceAllocationEnv
from marl.training.trainer import Trainer
from marl.agents.mappo_agent import MAPPOAgent
from marl.reward.reward_engine import RewardEngine

class DummyAlgorithm:
    def __init__(self, env):
        self.agents = {aid: MAPPOAgent(
            obs_dim=env.get_obs_dim(),
            global_state_dim=env.get_global_state_dim(),
            action_dim=env.get_action_dim(aid),
            agent_id=aid,
            config={}
        ) for aid in env.agent_ids}
        self.gamma = 0.99
        self.gae_lambda = 0.95
        
    def get_agents(self):
        return self.agents
        
    def update(self, buffer):
        return {'policy_loss': 0.0, 'value_loss': 0.0}
        
    def save(self, path):
        with open(path, 'w') as f:
            f.write("dummy_model")
            
    def load(self, path):
        pass

def test_checkpoint_selection(tmpdir):
    with open('configs/environment.yaml', 'r') as f:
        env_config = yaml.safe_load(f)
    with open('configs/marl.yaml', 'r') as f:
        marl_config = yaml.safe_load(f)
    env_config.update(marl_config)
    env_config['max_steps'] = 5
    
    env = MARLResourceAllocationEnv(env_config)
    algo = DummyAlgorithm(env)
    
    config = {
        'num_episodes': 2,
        'steps_per_episode': 5,
        'eval_interval': 1,
        'save_interval': 2,
        'checkpoint_dir': str(tmpdir)
    }
    
    trainer = Trainer(env, algo, None, config=config)
    trainer.train()
    
    assert os.path.exists(os.path.join(tmpdir, 'latest_model.pt'))
    assert os.path.exists(os.path.join(tmpdir, 'best_model.pt'))
    assert os.path.exists(os.path.join(tmpdir, 'best_model_metadata.json'))
