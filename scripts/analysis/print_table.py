import pandas as pd
import json

def load_data():
    try:
        df = pd.read_csv('results/baseline_comparison.csv', index_col=0)
    except FileNotFoundError:
        print("Error: results/baseline_comparison.csv not found.")
        return None, None
        
    try:
        with open('results/eval_results.json', 'r') as f:
            mappo_data = json.load(f)
    except FileNotFoundError:
        print("Error: results/eval_results.json not found.")
        return None, None
        
    return df, mappo_data

def format_table(df, mappo):
    baselines = ['fcfs', 'round_robin', 'met', 'random_feasible']
    names = ['FCFS', 'Round Robin', 'MET', 'Random Feasible']
    
    # Header
    print(f"{'Metric':<20} | {'MAPPO':<10} | {'FCFS':<10} | {'Round Robin':<12} | {'MET':<10} | {'Random Feasible':<15}")
    print("-" * 90)
    
    # 1. Completion rate
    metric = 'task_completion_rate_mean'
    row = f"{'Completion rate':<20} | {mappo[metric]:<10.3f} "
    for b in baselines:
        val = df.loc[b, metric] if b in df.index else 0.0
        width = 12 if b == 'round_robin' else (15 if b == 'random_feasible' else 10)
        row += f"| {val:<{width}.3f} "
    print(row)
    
    # 2. Avg latency
    metric = 'avg_task_latency_mean'
    row = f"{'Avg latency':<20} | {mappo[metric]:<10.2f} "
    for b in baselines:
        val = df.loc[b, metric] if b in df.index else 0.0
        width = 12 if b == 'round_robin' else (15 if b == 'random_feasible' else 10)
        row += f"| {val:<{width}.2f} "
    print(row)
    
    # 3. Total energy
    metric = 'total_energy_mean'
    row = f"{'Total energy':<20} | {mappo[metric]:<10,.0f} "
    for b in baselines:
        val = df.loc[b, metric] if b in df.index else 0.0
        width = 12 if b == 'round_robin' else (15 if b == 'random_feasible' else 10)
        row += f"| {val:<{width},.0f} "
    print(row)
    
    # 4. SLA violation rate
    metric = 'sla_violation_rate_mean'
    row = f"{'SLA violation rate':<20} | {mappo[metric]:<10.3f} "
    for b in baselines:
        val = df.loc[b, metric] if b in df.index else 0.0
        width = 12 if b == 'round_robin' else (15 if b == 'random_feasible' else 10)
        row += f"| {val:<{width}.3f} "
    print(row)

if __name__ == '__main__':
    df, mappo = load_data()
    if df is not None and mappo is not None:
        format_table(df, mappo)
