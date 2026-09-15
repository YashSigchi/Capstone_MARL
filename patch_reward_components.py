with open('marl/reward/reward_components.py', 'r') as f:
    content = f.read()

old_dict = """REWARD_COMPONENTS = {
    'latency': {'fn': compute_latency_reward, 'scope': 'local'},
    'energy': {'fn': compute_energy_reward, 'scope': 'local'},
    'utilization': {'fn': compute_utilization_reward, 'scope': 'global'},
    'load_balance': {'fn': compute_load_balance_reward, 'scope': 'global'},
    'sla': {'fn': compute_sla_reward, 'scope': 'local'},
    'queue': {'fn': compute_queue_reward, 'scope': 'local'},
    'throughput': {'fn': compute_throughput_reward, 'scope': 'global'}
}"""

new_dict = """REWARD_COMPONENTS = {
    'action_penalty': {'fn': compute_action_penalty, 'scope': 'local'},
    'latency': {'fn': compute_latency_reward, 'scope': 'local'},
    'energy': {'fn': compute_energy_reward, 'scope': 'local'},
    'utilization': {'fn': compute_utilization_reward, 'scope': 'global'},
    'load_balance': {'fn': compute_load_balance_reward, 'scope': 'global'},
    'sla': {'fn': compute_sla_reward, 'scope': 'local'},
    'queue': {'fn': compute_queue_reward, 'scope': 'local'},
    'throughput': {'fn': compute_throughput_reward, 'scope': 'global'}
}"""

content = content.replace(old_dict, new_dict)

with open('marl/reward/reward_components.py', 'w') as f:
    f.write(content)
