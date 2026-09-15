import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import re

EXP_DIR = 'experiments'
RESULTS_FILE = os.path.join(EXP_DIR, 'tuning_results.csv')
PLOTS_DIR = os.path.join(EXP_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

if not os.path.exists(RESULTS_FILE):
    print("No tuning results found yet.")
    exit(0)

df = pd.read_csv(RESULTS_FILE)

# 1. Hyperparameter sensitivity plots (Bar plots for Stage A, B, C, D, E)
def plot_sensitivity(stage_prefix, param_col, title):
    stage_df = df[df['run_id'].str.startswith(stage_prefix)]
    if stage_df.empty: return
    
    plt.figure(figsize=(10, 5))
    sns.barplot(data=stage_df, x=param_col, y='completion_rate', capsize=.1)
    plt.title(f"{title} - Completion Rate")
    plt.ylabel("Completion Rate")
    plt.savefig(os.path.join(PLOTS_DIR, f"{stage_prefix}_completion.png"))
    plt.close()
    
    plt.figure(figsize=(10, 5))
    sns.barplot(data=stage_df, x=param_col, y='SLA_rate', capsize=.1)
    plt.title(f"{title} - SLA Violation Rate")
    plt.ylabel("SLA Violation Rate")
    plt.savefig(os.path.join(PLOTS_DIR, f"{stage_prefix}_sla.png"))
    plt.close()

plot_sensitivity('StageA', 'learning_rate', 'Stage A (Learning Rate)')
plot_sensitivity('StageB', 'entropy_coef', 'Stage B (Entropy)')
# plot_sensitivity('StageC', 'gae_lambda', 'Stage C (GAE Lambda)')
plot_sensitivity('StageD', 'clip_range', 'Stage D (PPO Clip Range)')
plot_sensitivity('StageE', 'actor_architecture', 'Stage E (Architecture)')

# 2. Extract Training Curves from logs
logs = glob.glob(os.path.join(EXP_DIR, 'logs', '*.log'))
curves = []
for log in logs:
    run_id = os.path.basename(log).replace('.log', '')
    with open(log, 'r') as f:
        lines = f.readlines()
    for line in lines:
        m = re.search(r"Episode (\d+)/\d+\s+Reward:\s+([-0-9.]+)", line)
        if m:
            ep = int(m.group(1))
            reward = float(m.group(2))
            curves.append({'run_id': run_id, 'episode': ep, 'reward': reward})

if curves:
    curves_df = pd.DataFrame(curves)
    plt.figure(figsize=(12, 6))
    for run in curves_df['run_id'].unique():
        if "Baseline" in run or "StageA" in run:
            run_data = curves_df[curves_df['run_id'] == run]
            plt.plot(run_data['episode'], run_data['reward'], label=run, alpha=0.5)
    plt.title("Training Reward vs Episode (Subset)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "training_curves_reward.png"))
    plt.close()

print("Plots generated in experiments/plots/")
