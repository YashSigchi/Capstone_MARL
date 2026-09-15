import pytest
import numpy as np
import torch
from marl.agents.actor import ActorNetwork
from marl.agents.critic import CentralizedCritic
from marl.agents.mappo_agent import MAPPOAgent

def test_actor_output_shape():
    obs_dim = 10
    action_dim = 4
    actor = ActorNetwork(obs_dim, action_dim, [32], 'relu')
    obs = torch.randn(2, obs_dim)
    mask = torch.ones(2, action_dim)
    dist = actor(obs, mask)
    assert dist.probs.shape == (2, action_dim)

def test_actor_action_masking():
    obs_dim = 10
    action_dim = 4
    actor = ActorNetwork(obs_dim, action_dim, [32], 'relu')
    obs = torch.randn(1, obs_dim)
    mask = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    dist = actor(obs, mask)
    probs = dist.probs.detach().numpy()[0]
    assert probs[1] == 0.0
    assert probs[3] == 0.0

def test_actor_valid_actions_only():
    obs_dim = 10
    action_dim = 4
    actor = ActorNetwork(obs_dim, action_dim, [32], 'relu')
    obs = torch.randn(1, obs_dim)
    mask = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    dist = actor(obs, mask)
    action = dist.sample()
    assert action.item() == 3

def test_critic_output_shape():
    global_dim = 20
    critic = CentralizedCritic(global_dim, [32], 'relu')
    state = torch.randn(5, global_dim)
    val = critic(state)
    assert val.shape == (5,) or val.shape == (5, 1)

def test_mappo_agent_select_action(small_config):
    agent = MAPPOAgent(0, 10, 3, 20, {'device': 'cpu'})
    obs = np.random.randn(10)
    mask = np.array([1.0, 0.0, 1.0])
    action, log_prob, val = agent.select_action(obs, mask)
    assert isinstance(action, int)
    assert isinstance(log_prob, float)
    assert isinstance(val, float)
    assert action in [0, 2]

def test_mappo_agent_deterministic(small_config):
    agent = MAPPOAgent(0, 10, 3, 20, {'device': 'cpu'})
    obs = np.random.randn(10)
    mask = np.array([0.0, 1.0, 0.0])
    action = agent.select_action_deterministic(obs, mask)
    assert action == 1

def test_parameter_sharing_same_weights(small_config):
    actor = ActorNetwork(10, 3, [32], 'relu')
    agent1 = MAPPOAgent(0, 10, 3, 20, {'device': 'cpu'}, actor=actor)
    agent2 = MAPPOAgent(1, 10, 3, 20, {'device': 'cpu'}, actor=actor)
    
    assert list(agent1.actor.parameters())[0].data.equal(list(agent2.actor.parameters())[0].data)
