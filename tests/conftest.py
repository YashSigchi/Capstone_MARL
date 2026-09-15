import pytest

@pytest.fixture
def small_config():
    return {
        'num_resources': 4,
        'resources': [
            {'id': 'R1', 'cpu_capacity': 16, 'memory_capacity': 32, 'gpu_capacity': 1, 'network_bandwidth': 10, 'power_rate': 100, 'processing_speed': 1.0},
            {'id': 'R2', 'cpu_capacity': 8, 'memory_capacity': 16, 'gpu_capacity': 0, 'network_bandwidth': 10, 'power_rate': 50, 'processing_speed': 0.8},
            {'id': 'R3', 'cpu_capacity': 32, 'memory_capacity': 64, 'gpu_capacity': 2, 'network_bandwidth': 20, 'power_rate': 200, 'processing_speed': 1.5},
            {'id': 'R4', 'cpu_capacity': 8, 'memory_capacity': 16, 'gpu_capacity': 0, 'network_bandwidth': 5, 'power_rate': 40, 'processing_speed': 0.6},
        ],
        'clusters': [
            {'id': 'C0', 'resources': ['R1', 'R2']},
            {'id': 'C1', 'resources': ['R3', 'R4']},
        ],
        'topology': {'type': 'fully_connected', 'default_bandwidth': 10.0, 'default_latency': 1.0},
        'episode_length': 20,
        'max_tasks_per_episode': 30,
        'task_arrival': {'model': 'deterministic', 'rate': 2},
        'task_distributions': {
            'cpu_req': {'min': 1, 'max': 8},
            'memory_req': {'min': 1, 'max': 16},
            'gpu_req': {'min': 0, 'max': 1},
            'work_units': {'min': 2, 'max': 20},
            'deadline': {'min': 5, 'max': 30},
            'priority': {'min': 1, 'max': 5},
            'data_size': {'min': 1, 'max': 50},
            'communication_req': {'min': 0, 'max': 1},
            'sla_classes': ['gold', 'silver', 'bronze'],
            'sla_class_weights': [0.2, 0.5, 0.3],
        },
        'queue_ordering': 'fifo',
        'carbon_intensity': 0.5,
        'num_agents': 2,
        'agent_mapping': {0: 'C0', 1: 'C1'},
        'reward_blend_alpha': 0.5,
        'invalid_action_penalty': -1.0,
        'normalization': 'running_mean_std',
        'mode': 'fixed',
        'weights': {'utilization': 0.2, 'energy': 0.2, 'latency': 0.2, 'sla': 0.2, 'load_balance': 0.1, 'throughput': 0.1},
    }
