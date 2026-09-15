import argparse
import sys
import os
import yaml
import json

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from marl.environment.marl_env import MARLResourceAllocationEnv
from marl.reward.reward_engine import RewardEngine
from marl.algorithms.mappo import MAPPO
from marl.training.trainer import Trainer
from marl.utils.seeding import set_all_seeds
from marl.utils.logger import ExperimentLogger

def load_configs(experiment_config_path: str) -> dict:
    """Load and merge all referenced config files."""
    with open(experiment_config_path, 'r') as f:
        exp_config = yaml.safe_load(f)
        
    merged = dict(exp_config)
    
    # Config files are referenced relative to the experiment config's directory
    base_dir = os.path.dirname(os.path.abspath(experiment_config_path))
    
    # Map experiment.yaml keys to their config filenames
    config_refs = {
        'environment_config': None,
        'marl_config': None,
        'reward_config': None,
    }
    
    for key in config_refs:
        ref_path = exp_config.get(key)
        if ref_path:
            full_path = os.path.join(base_dir, ref_path)
            if os.path.exists(full_path):
                with open(full_path, 'r') as f:
                    sub_config = yaml.safe_load(f)
                    merged.update(sub_config)
            else:
                print(f"Warning: referenced config '{ref_path}' not found at {full_path}")
            
    return merged

def main():
    parser = argparse.ArgumentParser(description="Train MARL agents for resource allocation.")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment.yaml")
    parser.add_argument("--num-episodes", type=int, help="Override num_episodes from config")
    parser.add_argument("--seed", type=int, help="Override seed")
    parser.add_argument("--log-dir", type=str, help="Override log directory")
    parser.add_argument("--checkpoint-dir", type=str, help="Override checkpoint directory")
    parser.add_argument("--device", type=str, help="Override device (cpu/cuda)")
    
    args = parser.parse_args()
    
    # Load and merge configs
    config = load_configs(args.config)
    
    # Overrides
    if args.num_episodes is not None:
        config['num_episodes'] = args.num_episodes
    if args.seed is not None:
        config['seed'] = args.seed
    if args.log_dir is not None:
        config['log_dir'] = args.log_dir
    if args.checkpoint_dir is not None:
        config['save_dir'] = args.checkpoint_dir
    if args.device is not None:
        config['device'] = args.device
        
    seed = config.get('seed', 42)
    set_all_seeds(seed)
    
    # Create env
    env = MARLResourceAllocationEnv(config)
    
    # Ensure env attributes used by Trainer are present or hacked
    # Trainer expects env.observation_space[0].shape[0] if it's a list, otherwise shape[0]
    # But marl_env.py has observation_space as a method. We might need to monkeypatch it for Trainer.
    if callable(env.observation_space):
        class DummySpace:
            def __init__(self, shape, n=None):
                self.shape = shape
                self.n = n
        obs_dim = env.get_obs_dim()
        # Mocking for Trainer
        env.observation_space = DummySpace((obs_dim,))
        
    if callable(env.action_space):
        class DummySpace:
            def __init__(self, n):
                self.n = n
        act_dim = env.get_action_dim(env.agent_ids[0])
        env.action_space = DummySpace(act_dim)

    # Create reward engine
    reward_engine = RewardEngine(config)
    
    # Create algorithm
    mappo_config = dict(config)
    mappo_config['num_agents'] = env.num_agents
    mappo_config['obs_dim'] = env.get_obs_dim()
    mappo_config['action_dim'] = env.get_action_dim(env.agent_ids[0])
    mappo_config['global_state_dim'] = env.get_global_state_dim()
    
    algorithm = MAPPO(mappo_config)
    algorithm.setup(config, mappo_config)
    
    # Create logger
    log_dir = config.get('log_dir', './logs')
    logger = ExperimentLogger(log_dir=log_dir, experiment_name=config.get('experiment_name', 'marl_train'))
    logger.log_config(config)
    
    # Create trainer
    trainer = Trainer(
        env=env,
        algorithm=algorithm,
        reward_engine=reward_engine,
        config=config,
        logger=logger
    )
    
    print(f"Starting training for {config.get('num_episodes', 1000)} episodes...")
    history = trainer.train()
    
    print("Training finished.")
    if history.get('episode_reward'):
        print(f"Final Episode Reward: {history['episode_reward'][-1]:.2f}")
    
    logger.close()

if __name__ == "__main__":
    main()
