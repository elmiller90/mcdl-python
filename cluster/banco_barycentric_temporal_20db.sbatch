#!/bin/bash
#SBATCH --job-name=banco_bt20
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=04:00:00
#SBATCH --output=logs/banco_bt20_%j.out
#SBATCH --error=logs/banco_bt20_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=elmiller90@gmail.com

set -euo pipefail

cd "$HOME/projects/mcdl-python/mcdl-python"
source .venv/bin/activate

mkdir -p logs

python -u -m experiments.dictionary_learning.banco_barycentric_temporal_20db \
  --lambda-t-values 3e-5 \
  --lambda-s 0.24 \
  --beta-v 0.10 \
  --prefit-iters 100 \
  --temporal-iters 100
