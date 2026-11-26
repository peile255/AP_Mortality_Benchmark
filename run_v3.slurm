#!/bin/bash
#SBATCH --job-name=AP_V3
#SBATCH --partition=aoraki_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/AP_V3_%j.out
#SBATCH --error=logs/AP_V3_%j.err
set -euo pipefail
cd "$HOME/AP_Mortality_Benchmark"
module load python/3.13.0-e2wrr3t
source .venv/bin/activate
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLIS_NUM_THREADS=1
python --version
if [[ "${SMOKE:-0}" == 1 ]]; then
    python -u run_v3.py --smoke --jobs "$SLURM_CPUS_PER_TASK" --v2 final_run --out v3_smoke_hpc
else
    python -u run_v3.py --repeats 30 --outer 500 --oob 1000 --jobs "$SLURM_CPUS_PER_TASK" --v2 final_run --out v3_final
fi
