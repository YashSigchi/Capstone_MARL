with open('RESEARCH_NOTES.md', 'a') as f:
    f.write("\n\n## 4. Hyperparameter Tuning Results\n")
    f.write("Systematic tuning was conducted across Learning Rate, Entropy, GAE, PPO Params, and Architectures.\n")
    f.write("We observed that a higher learning rate (1e-3) and slightly reduced entropy (0.001) yielded the best completion rate and multi-objective score.\n")
    f.write("MAPPO effectively learns to selectively REJECT constrained or non-urgent tasks to prevent queue clogging, outperforming the blind baselines (FCFS, MET, RR) which greedily allocate tasks and suffer from massive SLA violations and lower overall completion rates.\n")
    f.write("The fully connected topology modification proved critical, resolving the previously observed infinite latency bug between non-adjacent grid resources.\n")

with open('CHANGELOG.md', 'a') as f:
    f.write("\n## [0.2.0] - Hyperparameter Tuning & Network Fix\n")
    f.write("### Added\n")
    f.write("- Extensive automated hyperparameter tuning suite in `hyperparameter_tuner.py`.\n")
    f.write("- Parallelized final training script `run_final_parallel.py`.\n")
    f.write("### Changed\n")
    f.write("- Modified default MARL configuration: LR=1e-3, Entropy=0.001 based on sweeping.\n")
    f.write("- Changed environment topology from `grid` to `fully_connected` to resolve infinite latency penalties on non-adjacent node data transfers.\n")
