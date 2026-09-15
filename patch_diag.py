with open('run_diagnostics.py', 'r') as f:
    content = f.read()

content = content.replace(
    "'power_rate': 250, 'processing_speed': 1.0 + (i*0.2)}",
    "'power_rate': 250, 'processing_speed': 1.0 + (i*0.2), 'network_bandwidth': 10.0}"
)

with open('run_diagnostics.py', 'w') as f:
    f.write(content)
