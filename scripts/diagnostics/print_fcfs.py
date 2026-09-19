import json
with open('results/reject_all_reward_breakdown.json', 'r') as f:
    d = json.load(f)
print("REJECT:", d["components"])
