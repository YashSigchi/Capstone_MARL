with open('audit_rewards.py', 'r') as f:
    content = f.read()

content = content.replace("from types import SimpleNamespace", "")
content = "from types import SimpleNamespace\n" + content

with open('audit_rewards.py', 'w') as f:
    f.write(content)
