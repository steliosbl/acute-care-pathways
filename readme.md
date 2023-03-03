
## Repo contents
------
The notebook `datasets/salford.ipynb` documents the exploratory analysis of the new Salford dataset.

Supporting code:
 - `datasets/salford.py`: Classes for interacting with the proprietary dataset used for testing.

## Installation instructions
------
*Note: Executing this code requires access to the proprietary Salford Royal Hospital dataset.*
1. (Optionally) create a virtual environment
```
python3 -m venv acpenv
source acpenv/bin/activate
```
2. Clone into directory
```
git clone https://github.com/stelioslogothetis/acute-care-pathways.git
cd acute-care-pathways
```
3. Install requirements via pip
```
pip install -r requirements.txt
```

### Requirements:

 - scikit-Learn >= 1.1.2
 - [SHAP](https://github.com/slundberg/shap) >= 0.41.0
 - [Optuna](https://github.com/optuna/optuna) >= 3.0.2
 - [Shapely](https://github.com/shapely/shapely) >= 1.8.4
 - [imbalanced-Learn](https://github.com/scikit-learn-contrib/imbalanced-learn) >= 0.70.0
 - [transformers](https://huggingface.co/docs/transformers/index) >= 4.26.1
 - [datasets](https://huggingface.co/docs/datasets/index) >= 2.10.1
 - [evaluate](https://huggingface.co/docs/evaluate/index) >= 0.4.0
 - Openpyxl