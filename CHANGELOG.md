# Changelog

## [0.1.1] - Next
### Fixed
- Fixed reward normalization loophole where `running_mean_std` effectively removed the REJECT action penalty, causing a degenerate 96% rejection rate.
- Fixed makespan calculation for truncated episodes to accurately reflect workload incompletion.
- Fixed `Trainer` evaluation to correctly update random seeds avoiding zero-variance deterministic outputs.

### Added
- Best model tracking: `Trainer` now saves `best_model.pt` based on a validation score combining completion rate, SLA violations, and energy efficiency.
- Added `test_checkpointing.py` to verify checkpoint generation and validation.
- Enhanced evaluator metrics with `completion_rate`, `rejection_rate`, and unfinished task counts.

## [0.2.0] - Hyperparameter Tuning & Network Fix
### Added
- Extensive automated hyperparameter tuning suite in `scripts/tuning/hyperparameter_tuner.py`.
- Parallelized final training script `scripts/training/train_parallel.py`.
### Changed
- Modified default MARL configuration: LR=1e-3, Entropy=0.001 based on sweeping.
- Changed environment topology from `grid` to `fully_connected` to resolve infinite latency penalties on non-adjacent node data transfers.
