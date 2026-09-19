from typing import Dict, Any

def print_resolved_config(config: Dict[str, Any], title: str = "FINAL RESOLVED CONFIGURATION"):
    """Print the final resolved configuration prominently."""
    print("=" * 50)
    print(title)
    print("=" * 50)
    
    # Rejection settings
    print(f"rejection.enabled: {config.get('environment_config', {}).get('rejection', {}).get('enabled', True) if isinstance(config.get('environment_config'), dict) else config.get('rejection', {}).get('enabled', True)}")
    print(f"rejection.max_rejection_rate: {config.get('environment_config', {}).get('rejection', {}).get('max_rejection_rate', 1.0) if isinstance(config.get('environment_config'), dict) else config.get('rejection', {}).get('max_rejection_rate', 1.0)}")
    
    # Admission reward settings
    adm_cfg = config.get('reward_config', {}).get('admission', {}) if isinstance(config.get('reward_config'), dict) else config.get('reward', {}).get('admission', {})
    if not adm_cfg: adm_cfg = config.get('admission', {})
    print(f"reward.admission.target_rejection_rate: {adm_cfg.get('target_rejection_rate', 0.0)}")
    print(f"reward.admission.base_penalty: {adm_cfg.get('base_penalty', 5.0)}")
    print(f"reward.admission.excess_penalty_scale: {adm_cfg.get('excess_penalty_scale', 10.0)}")
    
    # MAPPO settings
    mappo_keys = ['lr_actor', 'lr_critic', 'clip_range', 'entropy_coef', 'gamma', 'gae_lambda', 'num_epochs']
    print(f"MAPPO config keys: {[k for k in mappo_keys if k in config]}")
    for k in mappo_keys:
        if k in config:
            print(f"MAPPO {k}: {config[k]}")
            
    print("=" * 50 + "\n")
