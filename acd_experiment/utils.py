import numpy as np
import pandas as pd
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


def get_metrics(y_true, y_pred, y_pred_proba, n_resamples=99):
    auc_lower, auc_upper = roc_auc_ci_bootstrap(y_true, y_pred_proba, n_resamples)
    ap_lower, ap_upper = average_precision_ci_bootstrap(
        y_true, y_pred_proba, n_resamples
    )
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


def get_threshold_fpr(y_train, y_pred_proba, target=0.15):
    fpr, tpr, thresholds = roc_curve(y_train, y_pred_proba)
    closest = thresholds[np.abs(fpr - target).argmin()]

    return closest


def get_joined_categorical_string(X, categorical_cols, separator="__"):
    return (
        X[categorical_cols]
        .eq(1)
        .dot(pd.Index([_.split(separator)[1] for _ in categorical_cols]) + ", ")
        .str[:-2]
    )


def group_explanations_by_categorical(explanations, X, categorical_groups):
    idxs_to_exclude = []
    summed_shap_values = []
    if len(explanations.shape) > 2:
        explanations = explanations[:, :, 1]
    for group, col_names in categorical_groups.items():
        idxs = [X.columns.get_loc(_) for _ in col_names]
        idxs_to_exclude += idxs
        summed_shap_values.append(explanations.values[:, idxs].sum(axis=1))

    joined_categorical_data = [
        get_joined_categorical_string(X, _).values[:, np.newaxis]
        for _ in categorical_groups.values()
    ]

    idxs_to_include = list(set(range(X.shape[1])) - set(idxs_to_exclude))

    r = shap.Explanation(
        data=np.concatenate(
            [explanations.data[:, idxs_to_include]] + joined_categorical_data, axis=1,
        ),
        base_values=explanations.base_values,
        values=np.concatenate(
            [explanations.values[:, idxs_to_include]]
            + [_[:, np.newaxis] for _ in summed_shap_values],
            axis=1,
        ),
        feature_names=list(X.columns[idxs_to_include])
        + list(categorical_groups.keys()),
    )

    return r
