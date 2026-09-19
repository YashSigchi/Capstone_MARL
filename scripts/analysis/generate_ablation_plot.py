import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('results/rejection_ablation.csv')

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

mappo_df = df[df['method'] == 'MAPPO'].groupby('rejection_cap').mean(numeric_only=True).reset_index()
met_df = df[df['method'] == 'MET'].groupby('rejection_cap').mean(numeric_only=True).reset_index()

axes[0,0].plot(mappo_df['rejection_cap'], mappo_df['completion_rate'], label='MAPPO', marker='o')
axes[0,0].plot(met_df['rejection_cap'], met_df['completion_rate'], label='MET', marker='s', linestyle='--')
axes[0,0].set_title('Completion Rate vs Rejection Cap')
axes[0,0].legend()

axes[0,1].plot(mappo_df['rejection_cap'], mappo_df['sla_violation_rate'], label='MAPPO', marker='o')
axes[0,1].plot(met_df['rejection_cap'], met_df['sla_violation_rate'], label='MET', marker='s', linestyle='--')
axes[0,1].set_title('SLA Violation Rate vs Rejection Cap')

axes[1,0].plot(mappo_df['rejection_cap'], mappo_df['throughput'], label='MAPPO', marker='o')
axes[1,0].plot(met_df['rejection_cap'], met_df['throughput'], label='MET', marker='s', linestyle='--')
axes[1,0].set_title('Throughput vs Rejection Cap')

axes[1,1].plot(mappo_df['rejection_cap'], mappo_df['avg_latency'], label='MAPPO', marker='o')
axes[1,1].plot(met_df['rejection_cap'], met_df['avg_latency'], label='MET', marker='s', linestyle='--')
axes[1,1].set_title('Average Latency vs Rejection Cap')

plt.tight_layout()
plt.savefig('plots/rejection_tradeoff.png')
print("Saved plots/rejection_tradeoff.png")
