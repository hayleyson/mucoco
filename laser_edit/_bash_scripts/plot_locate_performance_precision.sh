#!/bin/bash
#SBATCH --time=0-01:00:00
#SBATCH --mem=8GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=plot_locate_metrics
#SBATCH --output='laser_edit/_slurm_outs/plot_locate_metrics_%j.out'

set -euo pipefail

source ~/.bashrc
source ~/miniconda3/etc/profile.d/conda.sh
conda activate loc-edit

cd /home/hyeryung/data/mucoco
export PYTHONPATH=.

python3 laser_edit/analyses/charts/src/plot_locate_performance_time_single_plot.py \
  --output-folder-name version_5 \
  --metrics Recall Recall "Exact Match"

python3 laser_edit/analyses/charts/src/plot_locate_performance_time_single_plot.py \
  --output-folder-name version_5 \
  --metrics Precision Precision "Exact Match"

python3 laser_edit/analyses/charts/src/plot_locate_performance_time_single_plot.py \
  --output-folder-name version_5 \
  --metrics F1 F1 "Exact Match"
