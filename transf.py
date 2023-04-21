# %%
import os, pickle, warnings, dataclasses, itertools
from pathlib import Path
from functools import partial 

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats as st

import torch
from torch import nn
from torch.nn import functional as F

from tqdm import tqdm
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from typing import Iterable, Tuple

import logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

import transformers
transformers.logging.set_verbosity_error()

from salford_datasets.salford import SalfordData, SalfordFeatures, SalfordPrettyPrint, SalfordCombinations
from salford_datasets.utils import DotDict

from transformer_experiment.utils import dict_product

# %%
class Notebook:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DATA_DIR = Path('data/Salford/')
    RE_DERIVE = False

# %%
if Notebook.RE_DERIVE:
    SAL = SalfordData.from_raw(
        pd.read_hdf(Notebook.DATA_DIR/'raw_v2.h5', 'table')
    ).augment_derive_all()
    SAL.to_hdf(Notebook.DATA_DIR/'sal_processed_transformers.h5', 'table')
else:
    SAL = SalfordData(pd.read_hdf(Notebook.DATA_DIR/'sal_processed_transformers.h5', 'table'))

# %%
BERTModels = DotDict(
    BioClinicalBert="emilyalsentzer/Bio_ClinicalBERT",
    Bert="distilbert-base-uncased",
    PubMedBert="ml4pubmed/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext_pub_section"
)

# %% [markdown]
# Experiments: 
#  1. Tabular Data only
#  2. Note Embeddings Only
#  3. Tabular & Note Embeddings
#     - One model for both
#     - Ensemble separate models
#  4. Note Transformer
#  5. Text-ified record Transformer
#  6. Note & Text-ified record Transformer
#     - One model for both
#     - Ensemble separate models
# 

# %%
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, roc_auc_score, average_precision_score, fbeta_score, make_scorer
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

CROSS_VALIDATION_METRICS = dict(
    Precision='precision',
    Recall='recall',
    AUC='roc_auc',
    AP='average_precision',
    F1='f1',
    F2=make_scorer(fbeta_score, beta=2)
)

LIGHTGBM_PARAMETERS = dict(
    objective='binary',
    random_state=123,
    metrics=['l2', 'auc'],
    boosting_type='gbdt',
    is_unbalance=True,
    n_jobs=1
)

REGRESSION_PARAMETERS = dict(
    max_iter=5000,
    solver='lbfgs',
    random_state=123,
    penalty='l2'
)

CALIBRATION_PARAMETERS = dict(
    ensemble=True,
    cv=StratifiedKFold(n_splits=4, shuffle=True, random_state=123),
    method='isotonic',
    n_jobs=1
)

CROSS_VALIDATION_PARAMETERS = dict(
    cv=StratifiedKFold(n_splits=4, shuffle=True, random_state=123),
    n_jobs=4,
    scoring=CROSS_VALIDATION_METRICS
)

# %%
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector, make_column_transformer

REGRESSION_PREPROCESSOR = make_column_transformer(
    (OneHotEncoder(), make_column_selector(dtype_include='category')),
    (SimpleImputer(strategy='median'), make_column_selector(dtype_include=np.number)),
    remainder='passthrough'
)

# %%
from sklearn.calibration import IsotonicRegression
def run_shallow_CV_experiments(X_variants, y):
    classifiers = {
        'LightGBM': LGBMClassifier(
            **LIGHTGBM_PARAMETERS
        ),
        'LR-L2': LogisticRegression(   
            **REGRESSION_PARAMETERS
        )
    }

    experiments = itertools.product(X_variants.items(), classifiers.items())

    results = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for (X_name, X), (classifier_name, classifier) in (pbar := tqdm(experiments)):
            pbar.set_description(f'Parallel running 4 CV folds of {classifier_name} with {X_name} embeddings..')
            if classifier_name == 'LR-L2':
                X = REGRESSION_PREPROCESSOR.fit_transform(X)

            results.append(pd.DataFrame.from_dict(
                cross_validate(
                    CalibratedClassifierCV(classifier, **CALIBRATION_PARAMETERS),
                    X, y, **CROSS_VALIDATION_PARAMETERS
                )
            ).assign(Embedding=X_name, Classifier=classifier_name))
    

    return pd.concat(results).groupby(['Embedding', 'Classifier']).mean()

# %% [markdown]
# ## 1. Tabular Classifier

# %%
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def cv_tabular_classifier(sal):
    X = SalfordData(sal[SalfordCombinations.with_services]).convert_str_to_categorical()
    y = sal.CriticalEvent
    X_variants = {
        key: X[columns] for key, columns in SalfordCombinations.items()
    }

    return run_shallow_CV_experiments(X_variants, y)

# %% [markdown]
# ## 2. Note Embedding Classifier

# %%
from transformers import AutoTokenizer, AutoModel
    
def load_tz_to_device(tz_output):
    """ Given the direct output of the tokeniser, loads the tokens to the GPU """
    return dict(map(
        lambda _: (_[0], _[1].to(Notebook.DEVICE)), tz_output.items()
    ))

def split_into_batches(Xt, batch_size):
    """ Given a tensor/ndarray and a batch size, splits it into batches of size up to batch_size along the first dimension """
    return np.array_split(
        Xt, np.ceil(len(Xt)/batch_size)
    )

def get_note_embeddings(X, model_uri=BERTModels.BioClinicalBert):
    tz, model = AutoTokenizer.from_pretrained(model_uri), AutoModel.from_pretrained(model_uri).to(Notebook.DEVICE).eval()
    tz_kwargs = dict(truncation=True, padding=True, return_tensors='pt')

    get_batch_embedding = lambda x: (
        model(
            **load_tz_to_device(tz(list(x), **tz_kwargs))
        )['last_hidden_state'][:, 0, :].cpu()
    )

    with torch.no_grad():
        emb = torch.cat([
            get_batch_embedding(_) for _ in tqdm(split_into_batches(X, 500), desc="Generating embeddings..")
        ])
    
    return emb

def get_note_embeddings_all_BERTs(sal):
    columns = ['AE_TriageNote', 'AE_MainDiagnosis', 'AE_PresentingComplaint']
    avail_idx = sal[columns].notna().any(axis=1)
    X = SalfordData(sal.loc[avail_idx]).tabular_to_text(columns).values

    with torch.no_grad():
        result = {
            model_name: get_note_embeddings(X, model_uri) for model_name, model_uri in BERTModels.items()
        }

    return result, avail_idx

# if Notebook.RE_DERIVE:
#     NOTE_EMBEDDINGS, note_avail_idx = get_note_embeddings_all_BERTs(SAL)
#     with open('data/cache/note_embeddings.bin', 'wb') as file:
#         pickle.dump((NOTE_EMBEDDINGS, note_avail_idx), file)
# else:
with open('data/cache/note_embeddings.bin', 'rb') as file:
    (NOTE_EMBEDDINGS, note_avail_idx) = pickle.load(file)

# %%
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def cv_embedding_only_classifier(sal, embeddings_dict, avail_idx):
    y = sal.loc[avail_idx, 'CriticalEvent'].astype(int)
    X_variants = {
        model_name: X.numpy() for model_name, X in embeddings_dict.items()
    }

    return run_shallow_CV_experiments(X_variants, y)

# %% [markdown]
# ## 3. Tabular & Embedding Classifier

# %% [markdown]
# ### 3.1 One Classifier for Both

# %%
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def cv_tabular_and_embedding_classifier(sal, embeddings_dict, avail_idx):
    X = SalfordData(sal.loc[avail_idx, SalfordCombinations.with_services]).convert_str_to_categorical()
    y = sal.loc[avail_idx, 'CriticalEvent']

    X_variants = {
        transformer: pd.concat((X, pd.DataFrame(embedding).add_prefix('EMBEDDING_').set_index(X.index)), axis=1)
        for transformer, embedding in embeddings_dict.items()
    }

    return run_shallow_CV_experiments(X_variants, y)


# %% [markdown]
# ### 3.2 Ensembles

# %%
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import VotingClassifier

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def cv_tabular_and_embedding_ensemble(sal, embeddings_dict, avail_idx):
    tabular_columns = SalfordCombinations.with_services
    X = SalfordData(sal.loc[avail_idx, tabular_columns]).convert_str_to_categorical()
    y = sal.loc[avail_idx, 'CriticalEvent']

    X_variants = {
        transformer: pd.concat((X, pd.DataFrame(embedding).add_prefix(f'EMBEDDING_').set_index(X.index)), axis=1)
        for transformer, embedding in embeddings_dict.items()
    }

    embedding_selector = make_column_transformer(('passthrough', make_column_selector(pattern='EMBEDDING_'))).set_output(transform='pandas')
    data_selector = make_column_transformer(('passthrough', tabular_columns)).set_output(transform='pandas')

    classifier_factory = {
        'LightGBM': lambda selector: make_pipeline(
            selector, 
            CalibratedClassifierCV(
                LGBMClassifier(**LIGHTGBM_PARAMETERS), **CALIBRATION_PARAMETERS
            )),
        'LR-L2': lambda selector: make_pipeline(
            selector, 
            REGRESSION_PREPROCESSOR, 
            CalibratedClassifierCV(
                LogisticRegression(**REGRESSION_PARAMETERS), **CALIBRATION_PARAMETERS
            ))
    }

    experiments = itertools.product(
        X_variants.items(), 
        itertools.product(classifier_factory.items(), repeat=2)
    )

    cross_validation_parameters = CROSS_VALIDATION_PARAMETERS | dict(
        n_jobs=4
    )

    results = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for (X_name, X), ((cls_name_data, cls_factory_data), (cls_name_embeddings, cls_factory_embeddings)) in (pbar := tqdm(experiments)):
            pbar.set_description(f'Parallel running 4 CV folds of {cls_name_data}-{cls_name_embeddings} with {X_name} embeddings..')

            ensemble = VotingClassifier([
                (f'DATA_{cls_name_data}', cls_factory_data(data_selector)),
                (f'EMB_{cls_name_embeddings}', cls_factory_embeddings(embedding_selector)),
            ], voting='soft')

            results.append(pd.DataFrame.from_dict(
                cross_validate(
                    ensemble,
                    X, y, **cross_validation_parameters
                )
            ).assign(Embedding=X_name, Classifier_Data=cls_name_data, Classifier_Emb=cls_name_embeddings))

        return pd.concat(results).groupby(['Embedding', 'Classifier_Data', 'Classifier_Emb']).mean()


RESULT = cv_tabular_and_embedding_classifier(SAL, NOTE_EMBEDDINGS, note_avail_idx)
#RESULT = cv_tabular_and_embedding_ensemble(SAL, NOTE_EMBEDDINGS, note_avail_idx)
RESULT.to_csv('data/cache/result31.csv')
