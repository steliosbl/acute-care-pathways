import logging, itertools
from pathlib import Path
import numpy as np
import pandas as pd
from functools import partial

from scipy import stats as st
import shap

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
    fbeta_score,
    roc_curve,
    average_precision_score,
    precision_recall_curve
)

from aif360.sklearn.metrics.metrics import intersection

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

def _single_confidence_interval_string(score, y_true, y_pred):
    mid = score(y_true, y_pred)
    lower, upper = bootstrap_metric(score, y_true, y_pred, n_resamples=99)
    return f'{mid:.3f} ({lower:.3f}-{upper:.3f})'

def get_decision_metrics(y_true, y_pred, confidence_intervals=False):
    nne = lambda *args, **kwargs: 1/precision_score(*args, **kwargs)

    scores = dict(
        Sensitivity=recall_score,
        Specificity=partial(recall_score, pos_label=0),
        PPV=precision_score,
        NPV=partial(precision_score, pos_label=0),
        Accuracy=accuracy_score,
        F2=partial(fbeta_score, beta=2),
        NNE=nne
    )

    if confidence_intervals:
        compute_scores = lambda y_t, y_p: {
            score_name: _single_confidence_interval_string(score, y_t, y_p)
            for score_name, score in scores.items()
        }
    else:
        compute_scores = lambda y_t, y_p: {
            score_name: score(y_t, y_p)
            for score_name, score in scores.items()
        }

    return compute_scores(y_true, y_pred)

def get_discriminative_metrics(y_true, y_pred_proba, n_resamples=99):
    auc_lower, auc_upper = roc_auc_ci_bootstrap(y_true, y_pred_proba, n_resamples)
    ap_lower, ap_upper = average_precision_ci_bootstrap(
        y_true, y_pred_proba, n_resamples
    )

    return dict(
        AUROC=roc_auc_score(y_true, y_pred_proba),
        AUROC_Upper=auc_upper,
        AUROC_Lower=auc_lower,
        AP=average_precision_score(y_true, y_pred_proba),
        AP_Upper=ap_upper,
        AP_Lower=ap_lower,
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
        sal = SalfordData(pd.read_hdf(data_dir/'sal_processed_transformers.h5', 'table').sort_values('AdmissionDate')).inclusion_exclusion_criteria()

    return sal

def get_train_test_indexes(sal, test_size=0.33):
    train_idx, test_idx = train_test_split(sal.index, shuffle=False, test_size=test_size)
    sal_train, sal_test = sal.loc[train_idx], sal.loc[test_idx]
    
    is_unseen = (~sal_test.PatientNumber.isin(sal_train.PatientNumber.unique()))
    unseen_idx = sal_test[is_unseen].index

    is_precovid = (sal_test.AdmissionDate < '2020-03-01')
    pre_covid_idx = sal_test[is_precovid].index

    return train_idx, test_idx, unseen_idx, is_unseen


def get_decision_threshold(y_true, y_pred_proba, target_recall=0.85):
    """ Given prediction probabilities, sets the prediction threshold to approach the given target recall
    """

    # Get candidate thresholds from the model, and find the one that gives the best fbeta score
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    closest = thresholds[np.abs(recall - target_recall).argmin()]

    return closest


def soft_base_rate(y_true, y_pred=None, *, concentration=1.0, pos_label=1, sample_weight=None):
    return (np.sum(y_true) + concentration)/(y_true.shape[0] + concentration)

def soft_selection_rate(y_true, y_pred, *, concentration=1.0, pos_label=1, sample_weight=None):
    return soft_base_rate(y_pred, concentration=concentration, pos_label=pos_label, sample_weight=sample_weight)

def soft_edf(y_true, y_pred=None, *, prot_attr=None, pos_label=1, concentration=1.0, sample_weight=None):
    rate = soft_base_rate if y_pred is None else soft_selection_rate
    sbr = intersection(rate, y_true, y_pred, prot_attr=prot_attr)
    logsbr = np.log(sbr)
    pos_ratio = max(abs(i - j) for i, j in itertools.permutations(logsbr, 2))
    lognegsbr = np.log(1 - np.array(sbr))
    neg_ratio = max(abs(i - j) for i, j in itertools.permutations(lognegsbr, 2))
    return max(pos_ratio, neg_ratio)

def soft_df_bias_amplification(y_true, y_pred, prot_attr, pos_label=1, concentration=1.0, sample_weight=None):
    eps_true = soft_edf(y_true, prot_attr=prot_attr, pos_label=pos_label,
                            concentration=concentration,
                            sample_weight=sample_weight)
    eps_pred = soft_edf(y_true, y_pred, prot_attr=prot_attr,
                            pos_label=pos_label, concentration=concentration,
                            sample_weight=sample_weight)
    return eps_pred - eps_true

def bootstrap_bias_amplification(y_true, y_score, prot_attr, n_resamples=99):
    center = soft_df_bias_amplification(y_true, y_score, prot_attr)
    res = st.bootstrap(
        data=(y_true.to_numpy(), y_score, prot_attr.to_numpy()),
        statistic=soft_df_bias_amplification,
        confidence_level=0.95,
        method="percentile",
        n_resamples=n_resamples,
        vectorized=False,
        paired=True,
        random_state=42,
    )
    return res.confidence_interval.low, center, res.confidence_interval.high