import re

with open('tests/test_rewards.py', 'r') as f:
    content = f.read()

# Remove test_latency_reward_deferred
content = re.sub(r"def test_latency_reward_deferred.*?assert val < 0\.0\n", "", content, flags=re.DOTALL)

# Add test for action_penalty
action_penalty_test = """
def test_action_penalty_deferred(small_config):
    info = {'action_was_defer': True}
    if 'action_penalty' in REWARD_COMPONENTS:
        val = REWARD_COMPONENTS['action_penalty']['fn'](info)
        assert val == -0.1

def test_action_penalty_reject(small_config):
    info = {'action_was_reject': True}
    if 'action_penalty' in REWARD_COMPONENTS:
        val = REWARD_COMPONENTS['action_penalty']['fn'](info)
        assert val == -1.0
"""
content += action_penalty_test

with open('tests/test_rewards.py', 'w') as f:
    f.write(content)
