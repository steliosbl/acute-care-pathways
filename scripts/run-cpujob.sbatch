#!/bin/bash
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -p cpu
#SBATCH -t 03-12:00:00
#SBATCH --mem=32g
#SBATCH --qos=long-low-prio
#SBATCH --job-name=acp_entropy
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user kqcr86@durham.ac.uk

eval "$('/home2/kqcr86/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate py310

python results.py