import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy import stats as st
import shap

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
    fbeta_score,
    roc_curve,
    average_precision_score,
)

from salford_datasets.salford import SalfordData

def bootstrap_metric(metric, y_true, y_score, n_resamples=9999):
    """ Computes AUROC with 95% confidence intervals by boostrapping """
    res = st.bootstrap(
        data=(y_true.to_numpy(), y_score),
        statistic=metric,
        confidence_level=0.95,
        method="percentile",
        n_resamples=n_resamples,
        vectorized=False,
        paired=True,
        random_state=42,
    )

    return res.confidence_interval.low, res.confidence_interval.high


def roc_auc_ci_bootstrap(y_true, y_score, n_resamples=9999):
    """ Computes AUROC with 95% confidence intervals by boostrapping """
    return bootstrap_metric(roc_auc_score, y_true, y_score, n_resamples)


def average_precision_ci_bootstrap(y_true, y_score, n_resamples):
    return bootstrap_metric(average_precision_score, y_true, y_score, n_resamples)

def get_metrics(y_true, y_pred_proba, y_pred_threshold=0.5, n_resamples=99):
    auc_lower, auc_upper = roc_auc_ci_bootstrap(y_true, y_pred_proba, n_resamples)
    ap_lower, ap_upper = average_precision_ci_bootstrap(
        y_true, y_pred_proba, n_resamples
    )

    y_pred = np.where(y_pred_proba > y_pred_threshold, 1, 0)

    return dict(
        AUC=roc_auc_score(y_true, y_pred_proba),
        AUC_Upper=auc_upper,
        AUC_Lower=auc_lower,
        AP=average_precision_score(y_true, y_pred_proba),
        AP_Upper=ap_upper,
        AP_Lower=ap_lower,
        Accuracy=accuracy_score(y_true, y_pred),
        Precision=precision_score(y_true, y_pred),
        Recall=recall_score(y_true, y_pred),
        F2=fbeta_score(y_true, y_pred, beta=2),
    )

def load_salford_dataset(re_derive=False, data_dir=Path('data/Salford')):
    data_dir = Path(data_dir)
    if re_derive:
        logging.info('Deriving from raw dataset')
        sal = SalfordData.from_raw(
            pd.read_hdf(data_dir/'raw_v2.h5', 'table')
        ).inclusion_exclusion_criteria().augment_derive_all().expand_icd10_definitions().sort_values('AdmissionDate')
        sal.to_hdf(data_dir/'sal_processed_transformers.h5', 'table')
    else:
        logging.info('Loading processed dataset')
        sal = SalfordData(pd.read_hdf(data_dir/'sal_processed_transformers.h5', 'table'))

    return sal

def get_train_test_indexes(sal, test_size=0.33):
    train_idx, test_idx = train_test_split(sal.index, shuffle=False, test_size=test_size)
    sal_train, sal_test = sal.loc[train_idx], sal.loc[test_idx]
    
    is_unseen = (~sal_test.PatientNumber.isin(sal_train.PatientNumber.unique()))
    unseen_idx = sal_test[is_unseen].index

    is_precovid = (sal_test.AdmissionDate < '2020-03-01')
    pre_covid_idx = sal_test[is_precovid].index

    return train_idx, test_idx, unseen_idx, is_unseen
