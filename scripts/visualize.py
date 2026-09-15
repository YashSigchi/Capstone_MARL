import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob

# Use seaborn style
sns.set_theme(style="whitegrid")

def plot_training_curve(df, output_path, format):
    plt.figure(figsize=(10, 6))
    if 'episode_reward' in df.columns:
        sns.lineplot(data=df, x='step' if 'step' in df.columns else df.index, y='episode_reward')
        plt.title('Training Reward Curve')
        plt.xlabel('Episode / Step')
        plt.ylabel('Total Reward')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'training_reward_curve.{format}'))
    plt.close()

def plot_reward_components(df, output_path, format):
    # Depending on how logger saved it, might need to extract components
    # Usually logged as scalar tags, so columns might be named 'reward_components/R_energy', etc.
    comp_cols = [c for c in df.columns if 'reward' in c and c != 'episode_reward' and c != 'total_reward']
    if not comp_cols:
        return
        
    plt.figure(figsize=(10, 6))
    for col in comp_cols:
        sns.lineplot(data=df, x='step' if 'step' in df.columns else df.index, y=col, label=col)
    
    plt.title('Reward Components Breakdown')
    plt.xlabel('Episode / Step')
    plt.ylabel('Component Value')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'reward_components.{format}'))
    plt.close()

def plot_metric_over_time(df, col_name, title, ylabel, output_path, format):
    if col_name not in df.columns:
        # try mean alternative
        if f"{col_name}_mean" in df.columns:
            col_name = f"{col_name}_mean"
        else:
            return
            
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='step' if 'step' in df.columns else df.index, y=col_name)
    plt.title(title)
    plt.xlabel('Episode / Step')
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f'{col_name}_over_time.{format}'))
    plt.close()

def plot_baseline_comparison(results_dir, output_path, format):
    comp_files = glob.glob(os.path.join(results_dir, '*comparison*.csv'))
    if not comp_files:
        return
        
    df = pd.read_csv(comp_files[0], index_col=0)
    
    # Plot SLA and Energy side-by-side if they exist
    metrics_to_plot = []
    if 'total_reward_mean' in df.columns: metrics_to_plot.append('total_reward_mean')
    if 'makespan_mean' in df.columns: metrics_to_plot.append('makespan_mean')
    if 'total_energy_mean' in df.columns: metrics_to_plot.append('total_energy_mean')
    if 'sla_violation_rate_mean' in df.columns: metrics_to_plot.append('sla_violation_rate_mean')
    
    for metric in metrics_to_plot:
        plt.figure(figsize=(8, 5))
        df[metric].plot(kind='bar', color=sns.color_palette("husl", len(df)))
        plt.title(f'Baseline Comparison: {metric.replace("_mean", "").replace("_", " ").title()}')
        plt.ylabel(metric.replace("_mean", "").replace("_", " ").title())
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'baseline_{metric}.{format}'))
        plt.close()

def plot_latency_distribution(df, output_path, format):
    if 'avg_task_latency' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x='avg_task_latency', kde=True)
        plt.title('Task Latency Distribution (Average per Episode)')
        plt.xlabel('Average Latency')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'task_latency_dist.{format}'))
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize MARL training and evaluation metrics.")
    parser.add_argument("--results-dir", type=str, help="Directory containing result JSON/CSV files")
    parser.add_argument("--log-file", type=str, help="Path to training metrics CSV")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for output plots")
    parser.add_argument("--format", type=str, default="png", choices=["png", "pdf", "svg"], help="Plot format")
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.log_file and os.path.exists(args.log_file):
        print(f"Reading log file {args.log_file}...")
        try:
            df = pd.read_csv(args.log_file)
            
            # 1. Training reward curve
            plot_training_curve(df, args.output_dir, args.format)
            
            # 2. Reward components
            plot_reward_components(df, args.output_dir, args.format)
            
            # 3. Resource utilization over time
            plot_metric_over_time(df, 'avg_cpu_utilization', 'CPU Utilization Over Training', 'Utilization', args.output_dir, args.format)
            
            # 4. Queue length over time
            plot_metric_over_time(df, 'avg_queue_waiting_time', 'Average Queue Waiting Time', 'Time', args.output_dir, args.format)
            
            # 5. Task latency distribution
            plot_latency_distribution(df, args.output_dir, args.format)
            
            # 6. SLA violation rate over training
            plot_metric_over_time(df, 'sla_violation_rate', 'SLA Violation Rate', 'Rate', args.output_dir, args.format)
            
            # 7. Energy consumption over training
            plot_metric_over_time(df, 'total_energy', 'Total Energy Consumption', 'Energy', args.output_dir, args.format)
            
            # 9. Load balance (variance of utilizations) over time
            plot_metric_over_time(df, 'resource_imbalance', 'Resource Imbalance (Utilization Std Dev)', 'Imbalance', args.output_dir, args.format)
            
            # 10. Throughput over training
            plot_metric_over_time(df, 'throughput', 'System Throughput', 'Tasks / Step', args.output_dir, args.format)
            
            print(f"Training plots saved to {args.output_dir}")
        except Exception as e:
            print(f"Error processing log file: {e}")
            
    if args.results_dir and os.path.exists(args.results_dir):
        print(f"Reading baseline results from {args.results_dir}...")
        try:
            # 8. Baseline comparison bar chart
            plot_baseline_comparison(args.results_dir, args.output_dir, args.format)
            print(f"Baseline comparison plots saved to {args.output_dir}")
        except Exception as e:
            print(f"Error processing baseline comparisons: {e}")
            
if __name__ == "__main__":
    main()
