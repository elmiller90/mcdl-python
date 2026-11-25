#!/bin/bash
#SBATCH --job-name=banco_bt
#SBATCH --array=0-3
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=06:00:00
#SBATCH --output=logs/banco_bt_%A_%a.out
#SBATCH --error=logs/banco_bt_%A_%a.err
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --mail-user=elmiller90@gmail.com

set -euo pipefail

cd "$HOME/projects/mcdl-python/mcdl-python"
source .venv/bin/activate

mkdir -p logs

LAMBDAS=(0 1e-5 3e-5 1e-4)
LT="${LAMBDAS[$SLURM_ARRAY_TASK_ID]}"

echo "Job ID: $SLURM_JOB_ID"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "lambda_t: $LT"
echo "Host: $(hostname)"
echo "Start: $(date)"

python -u -m experiments.dictionary_learning.banco_barycentric_temporal_20db \
  --prefit-file \
  results/dictionary_learning/prefits/banco_20db_lambda024_iter100.npz \
  --lambda-t-values "$LT" \
  --lambda-s 0.24 \
  --beta-v 0.10 \
  --prefit-iters 100 \
  --temporal-iters 50 \
  --output-dir \
  "results/dictionary_learning/banco_barycentric_temporal_20db/lt_${LT}"

echo "Finish: $(date)"