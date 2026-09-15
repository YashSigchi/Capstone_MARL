with open('configs/experiment.yaml', 'r') as f:
    content = f.read()

content = content.replace("eval_interval: 50", "eval_interval: 10")
content = content.replace("save_interval: 100", "save_interval: 50")

with open('configs/experiment.yaml', 'w') as f:
    f.write(content)
