with open('audit_rewards.py', 'r') as f:
    content = f.read()

content = content.replace("config['normalization'] = 'none'", "config['normalization'] = 'running_mean_std'")

with open('audit_rewards.py', 'w') as f:
    f.write(content)
