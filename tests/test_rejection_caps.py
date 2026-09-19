import yaml

def test_rejection_caps_match():
    with open('configs/environment.yaml') as f:
        env_cfg = yaml.safe_load(f)
    
    with open('configs/reward.yaml') as f:
        rew_cfg = yaml.safe_load(f)
        
    max_rej = env_cfg.get('rejection', {}).get('max_rejection_rate', 0.20)
    target_rej = rew_cfg.get('reward', {}).get('admission', {}).get('target_rejection_rate', 0.10)
    
    assert max_rej == target_rej, (
        f"Contradictory configuration: max_rejection_rate ({max_rej}) != "
        f"target_rejection_rate ({target_rej}). The agent will naturally gravitate "
        f"toward the physical cap because latency penalties outweigh admission penalties. "
        f"They should be perfectly aligned to avoid conflicting signals."
    )
