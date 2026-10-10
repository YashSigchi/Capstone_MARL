"""Print the MAPPO-vs-baselines comparison table in the terminal.

Usage (from repo root):
    python scripts/analysis/print_table.py
    python scripts/analysis/print_table.py --mappo results/processed/eval_results.json \
                                           --baselines results/processed/baseline_comparison.csv
"""
import argparse
import json
import sys

import pandas as pd

BASELINES = [("fcfs", "FCFS"), ("round_robin", "Round Robin"), ("met", "MET"), ("random_feasible", "Random Feasible")]

# (label, metric key, format spec)
ROWS = [
    ("Completion rate", "task_completion_rate_mean", ".3f"),
    ("Avg latency", "avg_task_latency_mean", ".2f"),
    ("Total energy", "total_energy_mean", ",.0f"),
    ("SLA violation rate", "sla_violation_rate_mean", ".3f"),
]


def build_rows(df, mappo):
    header = ["Metric", "MAPPO"] + [name for _, name in BASELINES]
    rows = []
    for label, key, fmt in ROWS:
        row = [label, format(mappo[key], fmt)]
        for bkey, _ in BASELINES:
            row.append(format(df.loc[bkey, key], fmt) if bkey in df.index else "n/a")
        rows.append(row)
    return header, rows


def main():
    parser = argparse.ArgumentParser(description="Print MAPPO vs baselines comparison table.")
    parser.add_argument("--mappo", default="results/processed/eval_results.json")
    parser.add_argument("--baselines", default="results/processed/baseline_comparison.csv")
    args = parser.parse_args()

    try:
        df = pd.read_csv(args.baselines, index_col=0)
        with open(args.mappo) as f:
            mappo = json.load(f)
    except FileNotFoundError as e:
        sys.exit(f"Missing results file: {e.filename}\n"
                 "Generate it first (see run.md, section 'Evaluate').")

    header, rows = build_rows(df, mappo)
    widths = [max(len(r[i]) for r in [header] + rows) for i in range(len(header))]
    line = lambda r: " | ".join(c.ljust(w) for c, w in zip(r, widths))
    print(line(header))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(line(r))


if __name__ == "__main__":
    main()
