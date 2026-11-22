#!/bin/bash -l
#SBATCH --job-name=mcdl-snr
#SBATCH --partition=batch
#SBATCH --qos=normal
#SBATCH --array=0-4
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:20:00
#SBATCH --output=cluster/mcdl-snr.%A_%a.out
#SBATCH --error=cluster/mcdl-snr.%A_%a.err

set -euo pipefail

module purge
module load python/3.13.2

cd "$HOME/projects/mcdl-python/mcdl-python"
source .venv/bin/activate

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

SNRS=(30 25 20 15 10)
SNR="${SNRS[$SLURM_ARRAY_TASK_ID]}"
SEED=100

OUT="results/dictionary_learning/paired/snr${SNR}_seed${SEED}.json"
mkdir -p "$(dirname "$OUT")"

echo "job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID"
echo "host=$(hostname)"
echo "snr=$SNR seed=$SEED"
echo "output=$OUT"

python -m experiments.dictionary_learning.paired_temporal_trial \
    --snr-db "$SNR" \
    --seed "$SEED" \
    --output "$OUT"