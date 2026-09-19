completion, sla penalty, latency, energy penalty, thfroighput, energy per task


python scripts/training/train.py --config configs/experiment.yaml --num-episodes 1000      

python scripts/evaluation/evaluate.py --config configs/experiment.yaml --checkpoint checkpoints/best_model.pt

python scripts/evaluation/run_baselines.py --config configs/experiment.yaml      

python scripts/analysis/print_table.py