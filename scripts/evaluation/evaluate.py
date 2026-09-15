import argparse
import sys
import os
import yaml
import json
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from marl.environment.marl_env import MARLResourceAllocationEnv
from marl.algorithms.mappo import MAPPO
from marl.reward.reward_engine import RewardEngine
from marl.evaluation.evaluator import Evaluator
from marl.utils.seeding import set_all_seeds

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
    parser = argparse.ArgumentParser(description="Evaluate MARL agents.")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="eval_results.json", help="Path to save results JSON")
    parser.add_argument("--deterministic", action="store_true", default=True, help="Use deterministic actions")
    
    args = parser.parse_args()
    
    config = load_configs(args.config)
    set_all_seeds(args.seed)
    
    env = MARLResourceAllocationEnv(config)
    
    # Mock spaces if needed by anything (not strictly needed for pure eval if we just use get_obs_dim)
    mappo_config = dict(config)
    mappo_config['num_agents'] = env.num_agents
    mappo_config['obs_dim'] = env.get_obs_dim()
    mappo_config['action_dim'] = env.get_action_dim(env.agent_ids[0])
    mappo_config['global_state_dim'] = env.get_global_state_dim()
    
    algorithm = MAPPO(mappo_config)
    algorithm.setup(config, mappo_config)
    
    print(f"Loading checkpoint from {args.checkpoint}...")
    algorithm.load(args.checkpoint)
    
    evaluator = Evaluator(env=env, reward_engine=RewardEngine(config), seed=args.seed)
    
    print(f"Running evaluation for {args.episodes} episodes...")
    agents = algorithm.get_agents()
    
    # Wrapper for Evaluator which expects agent actions
    metrics = evaluator.evaluate_agents(
        agents=agents,
        num_episodes=args.episodes,
        deterministic=args.deterministic
    )
    
    print("\n=== Evaluation Metrics ===")
    metrics_series = pd.Series(metrics)
    print(metrics_series.to_string())
    
    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
