import yaml
from marl.training.trainer import Trainer

# We just want to inspect the output of Trainer.evaluate()
with open("configs/experiment.yaml") as f:
    config = yaml.safe_load(f)

trainer = Trainer(config)
trainer.evaluate(num_episodes=1)
