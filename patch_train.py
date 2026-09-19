import re

with open('scripts/training/train.py', 'r') as f:
    content = f.read()

old_print = "print(f\"MAPPO config keys: {list(config.get('actor', {}).keys())}\")"
new_print = """mappo_keys = ['lr_actor', 'lr_critic', 'clip_range', 'entropy_coef', 'gamma', 'gae_lambda', 'num_epochs']
    print(f"MAPPO config keys: {[k for k in mappo_keys if k in config]}")
    for k in mappo_keys:
        if k in config:
            print(f"MAPPO {k}: {config[k]}")"""

content = content.replace(old_print, new_print)

with open('scripts/training/train.py', 'w') as f:
    f.write(content)
