#!/bin/bash
#SBATCH --job-name=dpa1_pt2_nvt10ps
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gpus=3090:1
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o slurm-%j.out
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/models/finetuned_ordinary.pt2}"
DATA_PATH="${DATA_PATH:-$REPO_ROOT/data/water64_wrapped.data}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-$REPO_ROOT/outputs/nvt_run}"

cd "$REPO_ROOT"
mkdir -p "$(dirname "$OUTPUT_PREFIX")"
module purge
module load DeepMD-kit/dpa1-master-cu126-kk
export OMP_NUM_THREADS=1
export DP_INTRA_OP_PARALLELISM_THREADS=1
export DP_INTER_OP_PARALLELISM_THREADS=1

echo "Model: $MODEL_PATH"
echo "Data: $DATA_PATH"
sha256sum "$MODEL_PATH" "$DATA_PATH"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
lmp -k on g 1 -sf kk -var datafile "$DATA_PATH" -var model "$MODEL_PATH" -var output_prefix "$OUTPUT_PREFIX" -log "${OUTPUT_PREFIX}.log" -in inputs/in.nvt.lammps
