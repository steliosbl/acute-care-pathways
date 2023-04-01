#!/bin/bash
#SBATCH -N 1
#SBATCH -c 24
#SBATCH -p cpu
#SBATCH -t 00-12:00:00
#SBATCH --mem=64g
#SBATCH --qos=short
#SBATCH --job-name=acp_1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user kqcr86@durham.ac.uk

eval "$('/home2/kqcr86/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate py310

python transf.py