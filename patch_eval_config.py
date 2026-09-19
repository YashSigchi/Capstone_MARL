import re

with open('scripts/evaluation/evaluate.py', 'r') as f:
    content = f.read()

old_block = """    print("FINAL RESOLVED EVALUATION CONFIGURATION")
    print("="*50)
    print(f"rejection.enabled: {config.get('rejection', {}).get('enabled', True)}")
    print(f"rejection.max_rejection_rate: {config.get('rejection', {}).get('max_rejection_rate', 1.0)}")
    print(f"reward.admission.target_rejection_rate: {config.get('reward', {}).get('admission', {}).get('target_rejection_rate', 0.0)}")
    print(f"reward.admission.base_penalty: {config.get('reward', {}).get('admission', {}).get('base_penalty', 5.0)}")
    print(f"reward.admission.excess_penalty_scale: {config.get('reward', {}).get('admission', {}).get('excess_penalty_scale', 10.0)}")
    
    # Optional print MAPPO
    print(f"MAPPO config keys: {list(config.get('actor', {}).keys())}")
    print("="*50 + "\\n")"""

new_block = """    from marl.utils.config import print_resolved_config
    print_resolved_config(config, title="FINAL RESOLVED EVALUATION CONFIGURATION")"""

content = content.replace(old_block, new_block)

with open('scripts/evaluation/evaluate.py', 'w') as f:
    f.write(content)
