with open('tests/test_toy_env.py', 'r') as f:
    content = f.read()

content = content.replace("reward_config['reward']", "reward_config")

with open('tests/test_toy_env.py', 'w') as f:
    f.write(content)
