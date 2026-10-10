"""Plot the MAPPO training curve from a training stdout log.

`Trainer.train()` returns its history in memory but does not write it to disk,
so the reward curve is rebuilt by parsing the `Episode N/M  Reward: X` lines
that `scripts/training/train.py` prints.

Usage (from repo root):
    python scripts/training/train.py --config configs/experiment.yaml --num-episodes 1000 \
        | tee logs/runs/train_1000ep.log
    python scripts/analysis/plot_training_curve.py
"""
import argparse
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

EPISODE_RE = re.compile(r"Episode\s+(\d+)/\d+\s+Reward:\s+([-+]?\d*\.?\d+)")


def parse_log(path):
    episodes, rewards = [], []
    with open(path, "r") as f:
        for line in f:
            m = EPISODE_RE.search(line)
            if m:
                episodes.append(int(m.group(1)))
                rewards.append(float(m.group(2)))
    return episodes, rewards


def main():
    parser = argparse.ArgumentParser(description="Plot training curve from a training log.")
    parser.add_argument("--log", default="logs/runs/train_1000ep.log", help="Training stdout log")
    parser.add_argument("--output", default="plots/training_curve.png", help="Output image path")
    parser.add_argument("--window", type=int, default=5, help="Moving-average window (logged points)")
    args = parser.parse_args()

    if not os.path.exists(args.log):
        sys.exit(f"Log file not found: {args.log}")

    episodes, rewards = parse_log(args.log)
    if not episodes:
        sys.exit(f"No 'Episode N/M Reward: X' lines found in {args.log}")

    smoothed = pd.Series(rewards).rolling(args.window, min_periods=1).mean()

    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards, alpha=0.4, label="Train reward")
    plt.plot(episodes, smoothed, color="orange", linewidth=2, label=f"Smoothed (window={args.window})")
    plt.title("MAPPO Training Curve")
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward")
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    plt.savefig(args.output)
    print(f"Parsed {len(episodes)} logged episodes -> saved {args.output}")


if __name__ == "__main__":
    main()
