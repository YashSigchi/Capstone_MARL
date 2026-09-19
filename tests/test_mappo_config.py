import yaml
from marl.algorithms.mappo import MAPPO

def test_mappo_config_resolution():
    config = {
        'lr_actor': 1e-5,
        'lr_critic': 2e-5,
        'clip_range': 0.1,
        'num_agents': 2,
        'actor': {
            'hidden_sizes': [64, 64]
        },
        'critic': {
            'hidden_sizes': [128, 128]
        }
    }
    
    mappo = MAPPO(config)
    mappo.obs_dim = 10
    mappo.action_dim = 5
    mappo.global_state_dim = 20
    mappo.setup(env_config={}, algorithm_config={})
    
    assert mappo.lr_actor == 1e-5
    assert mappo.lr_critic == 2e-5
    assert mappo.clip_range == 0.1
    
    # Check if network hidden sizes were actually picked up
    if hasattr(mappo, 'actors') and len(mappo.actors) > 0:
        actor = mappo.actors[0]
        # internal linear layers match config
        # roughly: state_dim -> 64 -> 64 -> action_dim
        first_layer_out_features = actor.net[0].out_features
        assert first_layer_out_features == 64, f"Expected 64, got {first_layer_out_features}"
        
    critic = mappo.critic
    first_layer_out_features_critic = critic.net[0].out_features
    assert first_layer_out_features_critic == 128, f"Expected 128, got {first_layer_out_features_critic}"
