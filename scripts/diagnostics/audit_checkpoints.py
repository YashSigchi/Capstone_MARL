import os, json, datetime, sys

chk_dir = "checkpoints"
print(f"=== CHECKPOINT PROVENANCE ===")
best_model = os.path.join(chk_dir, "best_model.pt")
if os.path.exists(best_model):
    st = os.stat(best_model)
    print(f"best_model.pt Modified: {datetime.datetime.fromtimestamp(st.st_mtime)}")
    
meta_file = os.path.join(chk_dir, "best_model_metadata.json")
if os.path.exists(meta_file):
    with open(meta_file, 'r') as f:
        meta = json.load(f)
        print("Metadata loaded:")
        print(json.dumps(meta, indent=2))
        
import subprocess
print("\n=== RUNNING TRAINER TO DUMP CONFIG ===")
subprocess.run([sys.executable, "scripts/training/train.py", "--config", "configs/experiment.yaml", "--num-episodes", "1"])
print("\n=== RUNNING EVALUATOR TO DUMP CONFIG ===")
subprocess.run([sys.executable, "scripts/evaluation/evaluate.py", "--config", "configs/experiment.yaml", "--checkpoint", "checkpoints/best_model.pt", "--episodes", "1"])

