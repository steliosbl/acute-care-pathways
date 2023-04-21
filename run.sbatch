#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -p ug-gpu-small
#SBATCH --gres=gpu
#SBATCH -t 01-18:00:00
#SBATCH --mem=28g
#SBATCH --qos=short
#SBATCH --job-name=41_PubMedBert
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user kqcr86@durham.ac.uk

eval "$('/home2/kqcr86/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate py310
python transf_fine_4.py --batch_size=56 --model=PubMedBert --experiment=41
