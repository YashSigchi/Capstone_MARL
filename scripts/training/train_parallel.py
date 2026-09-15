import subprocess
import yaml
import os

EXP_DIR = 'experiments'
BEST_CFG_PATH = os.path.join(EXP_DIR, 'configs', 'Best_Config.yaml')

with open('configs/marl.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

cfg['lr_actor'] = 1e-3
cfg['lr_critic'] = 1e-3
cfg['entropy_coef'] = 0.001

with open(BEST_CFG_PATH, 'w') as f:
    yaml.dump(cfg, f)

with open('configs/marl.yaml', 'w') as f:
    yaml.dump(cfg, f)

seeds = [42, 101, 202]
processes = []

for seed in seeds:
    with open('configs/experiment.yaml', 'r') as f:
        exp_cfg = yaml.safe_load(f)
    exp_cfg['seed'] = seed
    
    exp_cfg_path = f'configs/experiment_seed{seed}.yaml'
    with open(exp_cfg_path, 'w') as f:
        yaml.dump(exp_cfg, f)
        
    chk_dir = os.path.join(EXP_DIR, 'checkpoints', f'run_seed{seed}')
    os.makedirs(chk_dir, exist_ok=True)
        
    cmd = [
        "/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11", 
        "scripts/training/train.py", 
        "--config", exp_cfg_path, 
        "--num-episodes", "500",
        "--checkpoint-dir", chk_dir
    ]
    log_file = open(os.path.join(EXP_DIR, 'logs', f'Final_Run_seed{seed}.log'), 'w')
    p = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
    processes.append((p, log_file, seed))

print("Started 3 parallel training runs for 500 episodes each...")
for p, log_file, seed in processes:
    p.wait()
    log_file.close()

print("All training runs completed.")
