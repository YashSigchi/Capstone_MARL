import argparse
import sys
import os
import yaml
import json
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from marl.environment.marl_env import MARLResourceAllocationEnv
from marl.reward.reward_engine import RewardEngine
from marl.evaluation.evaluator import Evaluator
from marl.utils.seeding import set_all_seeds

from marl.baselines.fcfs import FCFSScheduler
from marl.baselines.round_robin import RoundRobinScheduler
from marl.baselines.met import METScheduler
from marl.baselines.random_feasible import RandomFeasibleScheduler

BASELINE_MAP = {
    'fcfs': FCFSScheduler,
    'round_robin': RoundRobinScheduler,
    'met': METScheduler,
    'random_feasible': RandomFeasibleScheduler
}

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
    parser = argparse.ArgumentParser(description="Run baseline comparisons.")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment.yaml")
    parser.add_argument("--baselines", nargs='+', default=list(BASELINE_MAP.keys()), help="List of baselines to run")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes per baseline")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="results/processed/baseline_comparison.csv", help="Path to save comparison CSV")
    
    args = parser.parse_args()
    
    config = load_configs(args.config)
    set_all_seeds(args.seed)
    
    env = MARLResourceAllocationEnv(config)
    evaluator = Evaluator(env=env, reward_engine=RewardEngine(config), seed=args.seed)
    
    results = {}
    
    for baseline_name in args.baselines:
        if baseline_name not in BASELINE_MAP:
            print(f"Warning: Unknown baseline '{baseline_name}'. Skipping.")
            continue
            
        print(f"Running baseline: {baseline_name} for {args.episodes} episodes...")
        scheduler_class = BASELINE_MAP[baseline_name]
        scheduler = scheduler_class()

        if evaluator.reward_engine is not None:
            evaluator.reward_engine.reset()

        metrics = evaluator.evaluate_scheduler(scheduler, num_episodes=args.episodes)
        results[baseline_name] = metrics
        
    if not results:
        print("No valid baselines were run.")
        return
        
    comparison_df = evaluator.compare(results)
    print("\n=== Baseline Comparison ===")
    print(comparison_df.to_string())
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    comparison_df.to_csv(args.output)
    print(f"\nComparison saved to {args.output}")

if __name__ == "__main__":
    main()
