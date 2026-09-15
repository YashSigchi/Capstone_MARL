with open('/Users/yashsigchi/.gemini/antigravity/brain/22756797-9b2a-4d03-ba60-7ecf7acbb292/DEBUG_AUDIT.md', 'r') as f:
    content = f.read()

content += """

## 16. Reward Normalization Loophole (Reject-Most-Tasks)
- **Issue**: Agent rejected 96% of tasks and ignored execution.
- **Root Cause**: `running_mean_std` normalization eroded the fixed `-1.0` reject penalty. If the agent always rejects, the mean penalty becomes `-1.0`, meaning the normalized penalty becomes `0.0`, masking the punishment.
- **Fix**: Disabled normalization (`normalization: none`) in `reward.yaml`.
- **Test**: `audit_rewards.py` verified that without normalization, REJECT (-1.0) is strictly worse than ALLOCATE for feasible tasks.

## 17. Best Checkpoint Selection
- **Issue**: Training only provided the final checkpoint (`latest_model.pt`), ignoring training instability and intermediate peaks.
- **Root Cause**: No validation evaluation existed in `Trainer`.
- **Fix**: Added validation eval loops against `val_seeds`. Created a scoring function favoring `task_completion_rate` to extract `best_model.pt`.
- **Test**: `test_checkpointing.py` verifies both checkpoints and metadata are written successfully.
"""

with open('/Users/yashsigchi/.gemini/antigravity/brain/22756797-9b2a-4d03-ba60-7ecf7acbb292/DEBUG_AUDIT.md', 'w') as f:
    f.write(content)
