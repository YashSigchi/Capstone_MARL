with open('audit_rewards.py', 'r') as f:
    content = f.read()

content = content.replace("'allocated_task': True,", "from types import SimpleNamespace\n    'allocated_task': SimpleNamespace(deadline=20.0),")

with open('audit_rewards.py', 'w') as f:
    f.write(content)
