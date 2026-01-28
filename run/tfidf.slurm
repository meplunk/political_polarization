#!/bin/bash
#SBATCH --job-name=tfidf
#SBATCH --partition=standard
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module purge
module load gcc/11.4.0  openmpi/4.1.4
module load python/3.11.4
export PYTHONUNBUFFERED=1

source /scratch/zjh6sn/political_polarization/venv/bin/activate
cd /scratch/zjh6sn/political_polarization

python -u code/stage_03a_tfidf.py