import os
import time

def execute_study(name, directory, script="run_acd_experiment.py", time=False):
    text = f"""#!/bin/bash
#SBATCH -N 1
#SBATCH -c 64
#SBATCH -p multi
#SBATCH --mem=128G      
#SBATCH --time=01-12:0:0  
#SBATCH --job-name=acp{'t' if time else ''}_{name}
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user kqcr86@durham.ac.uk

module load python/3.9.9
python {script} --models={name} {'--time_thresholds' if time else ''} --njobs=12 --trials=10 -o='{directory}/{name}.h5' --storage='sqlite:///{directory}/{name}.db' --verbose 
"""

    with open('sysjob', 'w') as file:
        file.write(text)

    return os.system('sbatch sysjob')

def execute_all(directory):
    studies = ['L1Regression', 'ElasticNetRegression', 'L2Regression', 'LinearSVM', 'LightGBM', 'XGBoost']
    for _ in studies:
        execute_study(_, directory)

def execute_time(directory):
    studies = ['LightGBM', 'L2Regression']
    for _ in studies:
        execute_study(_, directory, time=True)

execute_all('models/salford_run_1')
#execute_time('models/time_tough_with_h1_filtered')