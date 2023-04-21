# %%
import os, pickle, warnings, itertools
from pathlib import Path
from functools import partial 

import numpy as np
import pandas as pd

from tqdm import tqdm
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

import logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

from salford_datasets.salford import SalfordData, SalfordFeatures, SalfordPrettyPrint, SalfordCombinations
from acd_experiment.salford_adapter import SalfordAdapter

# %%
class Notebook:
    DATA_DIR = Path('data/Salford')
    CACHE_DIR = Path('data/cache')
    IMAGE_DIR = Path('images/shallow')
    SYSTEMATIC_COMPARISON_DIR = Path('data/systematic_comparison/')
    RE_DERIVE = False

# %%
from transformer_experiment.utils.shallow_classifiers import load_salford_dataset, get_train_test_indexes

SAL = load_salford_dataset(Notebook.RE_DERIVE, Notebook.DATA_DIR)
SAL_TRAIN_IDX, SAL_TEST_IDX, SAL_TEST_UNSEEN_IDX, SAL_TEST_IS_UNSEEN = get_train_test_indexes(SAL)

# %%
from acd_experiment.models import Estimator_L1Regression, Estimator_LinearSVM, Estimator_LightGBM, Estimator_L2Regression, Estimator_ElasticNetRegression, Estimator_XGBoost

FEATURE_GROUP_CORRESPONDENCE = {
    'news': 'news',
    'with_phenotype': 'news_with_phenotype',
    'with_labs': 'with_labs',
    'with_services': 'with_notes_labs_and_hospital'
}

PRETTY_PRINT_FEATURE_GROUPS = {
    'Reference': 'Reference',
    'news': 'Vitals',
    'with_phenotype': '& Obs',
    'with_labs': '& Labs',
    'with_services': '& Services',
}

PRETTY_PRINT_ESTIMATORS = dict(
    NEWS2='NEWS2',
    LogisticRegression='LR',
    L1Regression='LR-L1',
    L2Regression='LR-L2',
    ElasticNetRegression='LR-EN',
    XGBoost='XGBoost',
    LightGBM='LightGBM',
    LinearSVM='LinearSVM'
)

PRETTY_PRINT_FEATURE_NAMES = {
    k:(
        v.replace('First Blood ', '')
        .replace('First Obs ', '')
        .replace('Emergency Department', 'ED')
    )
    for k,v in SalfordPrettyPrint.items()
}

ESTIMATORS = {_._name: _ for _ in [
    Estimator_LightGBM,
    Estimator_L2Regression,
    Estimator_XGBoost,
    Estimator_LinearSVM,
    Estimator_L1Regression,
    Estimator_ElasticNetRegression,
]}

STUDY_GRID = list(itertools.product(ESTIMATORS.keys(), FEATURE_GROUP_CORRESPONDENCE.keys()))

# %% [markdown]
# ## Model Training

# %%
from acd_experiment.salford_adapter import SalfordAdapter
from sklearn.calibration import CalibratedClassifierCV
import optuna
from acd_experiment.systematic_comparison import get_xy, PipelineFactory

def run_pretuned(sal, estimator_name, feature_group_name, cv_jobs=4, explain_models=['LightGBM', 'L2Regression']):
    params = optuna.load_study(
        study_name =f'{estimator_name}_None_Within-1_{FEATURE_GROUP_CORRESPONDENCE[feature_group_name]}', storage=f'sqlite:///{Notebook.SYSTEMATIC_COMPARISON_DIR}/{estimator_name}.db'
    ).best_params
    
    estimator = ESTIMATORS[estimator_name]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X, y = SalfordAdapter(sal).xy(
            x=SalfordCombinations[feature_group_name],
            imputation = estimator._requirements['imputation'],
            fillna = estimator._requirements['fillna'],
            ordinal_encoding = estimator._requirements['ordinal'],
            onehot_encoding = estimator._requirements['onehot']
        )
    X_train, y_train = SalfordAdapter(X.loc[SAL_TRAIN_IDX]), y.loc[SAL_TRAIN_IDX].values

    pipeline_factory = PipelineFactory(
        estimator=estimator, resampler=None, X_train=X_train, y_train=y_train,
    )

    model = CalibratedClassifierCV(
        pipeline_factory(**params), cv=cv_jobs, method="isotonic", n_jobs=cv_jobs,
    ).fit(X_train, y_train)


    y_pred_proba = model.predict_proba(X.loc[SAL_TEST_IDX])[:,1]
    y_pred_proba_unseen = y_pred_proba[SAL_TEST_IS_UNSEEN]

    explanations = None
    if estimator_name in explain_models:
        explanations = estimator.explain_calibrated(
            model, X_train, SalfordAdapter(X.loc[SAL_TEST_IDX]), cv_jobs=cv_jobs
        )

    return y_pred_proba, y_pred_proba_unseen, explanations, model
    

if Notebook.RE_DERIVE or True:
    RESULTS = {}
    for estimator_name, feature_group_name in (pbar := tqdm(STUDY_GRID)):
        pbar.set_description(f'Training {estimator_name} on {feature_group_name}')
        RESULTS[(estimator_name, feature_group_name)] = run_pretuned(SAL, estimator_name, feature_group_name)

        with open(Notebook.CACHE_DIR/'shallow_results_2.bin', 'wb') as file:
            pickle.dump(RESULTS, file)
else:
    with open(Notebook.CACHE_DIR/'shallow_results_2.bin', 'rb') as file:
            RESULTS = pickle.load(file)
