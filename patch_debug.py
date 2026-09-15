with open('/Users/yashsigchi/.gemini/antigravity/brain/22756797-9b2a-4d03-ba60-7ecf7acbb292/DEBUG_AUDIT.md', 'r') as f:
    content = f.read()

content = content.replace(
    "## 5. SJF Validation",
    "## 5. SJF -> MET Validation"
)
content = content.replace(
    "**Issue**: FCFS and SJF baselines produced identical results.",
    "**Issue**: FCFS and SJF baselines produced identical results. SJF was not scheduling the queue, it just picked the fastest resource for the head."
)
content = content.replace(
    "**Fix**: Implemented a deterministic test (`test_sjf_deterministic.py`) where R1 is faster than R0 to definitively prove FCFS picks R0 and SJF picks R1.",
    "**Fix**: Renamed SJF to MET (Minimum Execution Time). Implemented a deterministic test (`test_met_deterministic.py`) where R1 is faster than R0 to definitively prove FCFS picks R0 and MET picks R1."
)

content = content.replace(
    "**Issue**: `makespan = 200` currently equals the episode horizon, and unfinished tasks are ignored.",
    "**Issue**: `makespan = 200` currently equals the episode horizon, and unfinished tasks are ignored. True makespan requires semantics for truncated episodes."
)
content = content.replace(
    "**Fix**: Initialized `_makespan` tracking the completion time of the last finished task. Added `unfinished_tasks` metric to step infos representing `sum(q.length) + len(running_tasks)`.",
    "**Fix**: Tracked the completion time of the last finished task. Modified step info so if there are unfinished tasks at truncation, `makespan` correctly defaults to the episode length. Added `unfinished_tasks` metric."
)

content = content.replace(
    "**Fix**: Increased `defer` penalties and added a severe `-1.0` penalty for `reject` actions across latency, SLA, and queue components to strongly incentivize task completion.",
    "**Fix**: Removed action penalties from latency and SLA components to avoid conflating action validity with physical metrics. Added an independent `action_penalty` component to `reward.yaml` and `RewardEngine` to explicitly penalize defer/reject."
)

content = content.replace(
    "## 9. MAPPO Correctness",
    "## 9. MAPPO & GAE Correctness"
)
content = content.replace(
    "**Test**: `test_integration.py` successfully completes MAPPO forward pass and GAE computation.",
    "**Test**: Added `tests/test_gae.py` with explicit numerical validation of the GAE calculation covering both termination and truncation edge cases."
)

with open('/Users/yashsigchi/.gemini/antigravity/brain/22756797-9b2a-4d03-ba60-7ecf7acbb292/DEBUG_AUDIT.md', 'w') as f:
    f.write(content)
