#!/bin/bash
set -e

# Step 1: Directories
mkdir -p scripts/experiments
mkdir -p results/processed
mkdir -p results/reports

# Step 2: Move active entry points
mv run_all_tuning.sh scripts/experiments/

# Step 3: Update imports/path references
# Add ROOT_DIR cd to run_all_tuning.sh
cat << 'INNER_EOF' > tmp.sh
#!/bin/bash
set -e
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
INNER_EOF
tail -n +3 scripts/experiments/run_all_tuning.sh >> tmp.sh
mv tmp.sh scripts/experiments/run_all_tuning.sh
chmod +x scripts/experiments/run_all_tuning.sh

# Step 4: Move generated artifacts
mv eval_results.json results/processed/ 2>/dev/null || true
mv baseline_comparison.csv results/processed/ 2>/dev/null || true

# Rename documentation
mv understand.md ARCHITECTURE_NOTES.md 2>/dev/null || true

# Step 5: Remove confirmed one-time patch scripts & obsolete shell scripts
rm -f fix_*.py modify_*.py patch_*.py prep_artifacts.py generate_reports.py print_df.py print_rej.py revert_pool.py
rm -f run_phase5.sh run_remainder.sh
rm -f phase_p_out.txt ablation.log ablation_output.log remainder.log master.log

# Step 6: Remove build/cache artifacts
rm -rf marl_resource_allocation.egg-info
rm -rf .pytest_cache
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true

# Step 8: Add/update .gitignore
cat << 'INNER_EOF' > .gitignore
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
.pytest_cache/
.coverage
htmlcov/
dist/
build/
INNER_EOF

echo "Cleanup successfully executed!"
