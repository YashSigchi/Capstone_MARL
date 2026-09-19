import re

with open('scripts/training/train.py', 'r') as f:
    content = f.read()

old_block = """    print("FINAL RESOLVED CONFIGURATION")
    print("="*50)
    print(f"rejection.enabled: {config.get('rejection', {}).get('enabled', True)}")
    print(f"rejection.max_rejection_rate: {config.get('rejection', {}).get('max_rejection_rate', 1.0)}")
    
    adm_cfg = config.get('reward', {}).get('admission', {})
    if not adm_cfg: adm_cfg = config.get('admission', {})
    print(f"reward.admission.target_rejection_rate: {adm_cfg.get('target_rejection_rate', 0.0)}")
    print(f"reward.admission.base_penalty: {adm_cfg.get('base_penalty', 5.0)}")
    print(f"reward.admission.excess_penalty_scale: {adm_cfg.get('excess_penalty_scale', 10.0)}")
    
    mappo_keys = ['lr_actor', 'lr_critic', 'clip_range', 'entropy_coef', 'gamma', 'gae_lambda', 'num_epochs']
    print(f"MAPPO config keys: {[k for k in mappo_keys if k in config]}")
    for k in mappo_keys:
        if k in config:
            print(f"MAPPO {k}: {config[k]}")
    print("="*50 + "\\n")"""

new_block = """    from marl.utils.config import print_resolved_config
    print_resolved_config(config)"""

content = content.replace(old_block, new_block)

with open('scripts/training/train.py', 'w') as f:
    f.write(content)
