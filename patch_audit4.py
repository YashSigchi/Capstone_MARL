with open('audit_rewards.py', 'r') as f:
    content = f.read()

content = content.replace("total_completion_time': 5.0", "total_completion_time': 30.0")
content = content.replace("'sla_violated': False", "'sla_violated': True")

with open('audit_rewards.py', 'w') as f:
    f.write(content)
