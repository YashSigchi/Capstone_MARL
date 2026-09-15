with open('audit_rewards.py', 'r') as f:
    content = f.read()

content = content.replace("config['reward']['normalization']", "config['normalization']")
content = content.replace("engine = RewardEngine(config['reward'])", "engine = RewardEngine(config)")

with open('audit_rewards.py', 'w') as f:
    f.write(content)
