#!/bin/bash
#SBATCH --job-name=acpJupyter
#SBATCH --nodes=1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB
#SBATCH --time=07-00:00:00
#SBATCH --qos=long-low-prio
#SBATCH -p ug-gpu-small

eval "$('/home2/kqcr86/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
conda activate py310

#Pick a random or predefined port
#port=$(shuf -i 6000-9999 -n 1)
port=8572
#Forward the picked port to the prince on the same port. Here log-x is set to be the prince login node.
/usr/bin/ssh -N -f -R $port:localhost:$port ncc1.clients.dur.ac.uk

#Start the notebook
jupyter notebook --no-browser --port $port 