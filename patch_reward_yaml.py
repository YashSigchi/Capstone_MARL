with open('configs/reward.yaml', 'r') as f:
    content = f.read()

content = content.replace("weights:\n  latency: 0.25", "weights:\n  action_penalty: 0.20\n  latency: 0.15")
content = content.replace("  sla: 0.20", "  sla: 0.10")
content = content.replace("    latency:\n      min: 0.05\n      max: 0.40", "    action_penalty:\n      min: 0.0\n      max: 1.0\n    latency:\n      min: 0.05\n      max: 0.40")

with open('configs/reward.yaml', 'w') as f:
    f.write(content)
