import re
import matplotlib.pyplot as plt
import os

episodes = []
rewards = []

with open('train_out.log', 'r') as f:
    for line in f:
        # Example: Episode 320/1000  Reward: -27.59  Policy Loss: -0.0211  Value Loss: 0.1897
        m = re.search(r'Episode\s+(\d+)/\d+\s+Reward:\s+([-+]?\d*\.\d+|\d+)', line)
        if m:
            episodes.append(int(m.group(1)))
            rewards.append(float(m.group(2)))

plt.figure(figsize=(10, 6))
plt.plot(episodes, rewards, label='Train Reward')
import pandas as pd
rew = pd.Series(rewards).rolling(5).mean()
plt.plot(episodes, rew, label='Train Reward (Smoothed)', color='orange', linewidth=2)
plt.title('MAPPO Training Curve')
plt.xlabel('Episode')
plt.ylabel('Episode Reward')
plt.legend()
plt.tight_layout()
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/training_curve.png')
print(f"Parsed {len(episodes)} episodes.")
