from marl.evaluation.evaluator import Evaluator
from marl.evaluation.metrics import MetricsTracker

def test_cpu_memory_utilization_divergence():
    class DummyResource:
        def __init__(self):
            self.cpu_capacity = 100
            self.memory_capacity = 100
            self.current_cpu_usage = 10.0
            self.current_memory_usage = 90.0
        def utilization(self): return 0.5

    class DummyEnv:
        def __init__(self):
            self.resources = {'R1': DummyResource()}
            self.agent_ids = [0]
            self._total_energy = 0
            self._total_carbon = 0
            self.sla_tracker = None
            self.queues = {}

    env = DummyEnv()
    evaluator = Evaluator(env, None)
    
    # Mock step info
    step_info = {'reward': 0.0}
    
    # test _extract_episode_end_metrics directly
    import collections
    class EpisodeMetrics:
        def __init__(self):
            self.makespan = 10
            self.task_completion_rate = 0.0
            self.task_rejection_rate = 0.0
            self.throughput = 0.0
    em = EpisodeMetrics()
    evaluator._extract_episode_end_metrics(em)
    assert em.avg_cpu_utilization == 0.1
    assert em.avg_memory_utilization == 0.9
    
    # also test step metrics tracking directly
    tracker = MetricsTracker()
    infos = {0: {'reward_info': {'all_utilizations': [0.5], 'all_resources': [DummyResource()]}}}
    extracted = evaluator._extract_step_metrics(infos, {0: 0.0})
    tracker.record_step(extracted)
    tracker.finalize_episode(10)
    summary = tracker.get_summary()
    
    assert 'avg_cpu_utilization_mean' in summary
    assert summary['avg_cpu_utilization_mean'] == 0.1
    assert summary['avg_memory_utilization_mean'] == 0.9

