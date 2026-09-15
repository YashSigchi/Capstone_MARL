with open('tests/test_gae.py', 'r') as f:
    content = f.read()

content = content.replace("RolloutBuffer(num_steps, num_agents", "RolloutBuffer(num_agents, num_steps")

with open('tests/test_gae.py', 'w') as f:
    f.write(content)
