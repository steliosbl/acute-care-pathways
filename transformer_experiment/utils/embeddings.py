from collections import defaultdict
import pandas as pd
import numpy as np
from scipy import stats as st
import math, logging
from pathlib import Path

from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.base import clone as clone_estimator
from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    StratifiedShuffleSplit,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
    fbeta_score,
    make_scorer,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
    average_precision_score,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.calibration import CalibrationDisplay
from sklearn.dummy import DummyClassifier

import torch
from torch.nn import functional as F

from salford_datasets.salford import SalfordData
from sklearn.model_selection import train_test_split
from acd_experiment.sci import SCIData, SCICols
import shap


def dict_product(dicts):
    """
    >>> list(dict_product(dict(number=[1,2], character='ab')))
    [{'character': 'a', 'number': 1},
     {'character': 'a', 'number': 2},
     {'character': 'b', 'number': 1},
     {'character': 'b', 'number': 2}]
    """
    return (dict(zip(dicts, x)) for x in product(*dicts.values()))


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

def get_threshold(y_train, y_pred_proba, target=0.85):
    """ Given prediction probabilities, sets the prediction threshold to approach the given target recall
    """

    # Get candidate thresholds from the model, and find the one that gives the best fbeta score
    precision, recall, thresholds = precision_recall_curve(y_train, y_pred_proba)
    closest = thresholds[np.abs(recall - target).argmin()]

    return closest

def plot_confusion_matrix(y_true, y_pred, ax=None, save=None, plot_title=None):
    no_ax = ax is None
    if no_ax:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        plt.rc("axes", titlesize=12)

    ax.grid(False)
    cm_fig = ConfusionMatrixDisplay(
        np.rot90(np.flipud(confusion_matrix(y_true, y_pred, normalize="true"))),
        display_labels=[1, 0],
    ).plot(values_format=".2%", ax=ax, cmap="Purples")

    ax.set_xlabel("True Class")
    ax.set_ylabel("Predicted Class")
    ax.set_title(plot_title)

    if save:
        plt.savefig(save, bbox_inches="tight", dpi=200)

    if no_ax:
        plt.rc("axes", titlesize=12)

    return cm_fig

def evaluate_from_pred(
    y_true,
    y_pred_proba,
    plot_title=None,
    pos_label=1,
    save=None,
    n_resamples=99,
):
    y_pred = np.where(y_pred_proba > get_threshold(y_true, y_pred_proba), 1, 0)

    lower, upper = roc_auc_ci_bootstrap(y_true, y_pred_proba, n_resamples)
    metric_df = pd.DataFrame(
        {
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, pos_label=pos_label),
            "Recall": recall_score(y_true, y_pred, pos_label=pos_label),
            "AP": average_precision_score(y_true, y_pred_proba),
            "F2 Score": fbeta_score(y_true, y_pred, beta=2, pos_label=pos_label),
            "AUC": roc_auc_score(y_true, y_pred_proba),
            "AUC_CI": f"{roc_auc_score(y_true, y_pred_proba):.3f} ({lower:.3f}-{upper:.3f})",
        },
        index=["Model"],
    )

    display(metric_df)

    # display(confusion_matrix(y_true, y_pred))

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    roc_fig = RocCurveDisplay.from_predictions(
        y_true, y_pred_proba, ax=ax[0],  # pos_label=pos_label
    )
    pr_fig = PrecisionRecallDisplay.from_predictions(
        y_true, y_pred_proba, ax=ax[1], pos_label=pos_label
    )

    if (-1) in np.array(y_true):
        get = {1: True, -1: False}.get
        y_true, y_pred = (list(map(get, y_true)), list(map(get, y_pred)))

    # cm_fig = ConfusionMatrixDisplay.from_predictions(
    #     y_true, y_pred, ax=ax[1], normalize="true", values_format=".2%"
    # )
    cm_fig = plot_confusion_matrix(y_true, y_pred, ax[2])

    plt.suptitle(plot_title)

    if save:
        plt.savefig(save, bbox_inches="tight", dpi=100)

    return metric_df, roc_fig, pr_fig, cm_fig

def load_sal(re_derive=False, data_dir='data/Salford'):
    data_dir = Path(data_dir)
    if re_derive:
        logging.info('Deriving from raw dataset')
        sal = SalfordData.from_raw(
            pd.read_hdf(data_dir/'raw_v2.h5', 'table')
        ).augment_derive_all().expand_icd10_definitions().sort_values('AdmissionDate')
        sal.to_hdf(data_dir/'sal_processed_transformers.h5', 'table')
    else:
        logging.info('Loading processed dataset')
        sal = SalfordData(pd.read_hdf(data_dir/'sal_processed_transformers.h5', 'table'))

    return sal

def get_sci_indexes():
    scii = (
        SCIData(
            SCIData.quickload("data/SCI/sci_processed.h5").sort_values(
                "AdmissionDateTime"
            )
        )   
        .mandate(SCICols.news_data_raw)
        .derive_ae_diagnosis_stems(onehot=False)
        .derive_critical_event(return_subcols=True, within=1)
        .set_index('SpellSerial')
    )

    scii = SCIData(scii[
        (~((scii.BreathingDevice=='NIV - NIV')&(scii.AdmitWard=='HH1M'))) #&
    # (~scii.AdmissionMethodDescription.isin(['BOOKED ADMISSION ', 'ELECTIVE PLANNED']))
    ].copy())

    sci_train, sci_test = train_test_split(
        scii,
        test_size=0.33,
        shuffle=False,
    )

    sci_test = sci_test[(
    #  (~((scii.BreathingDevice=='NIV - NIV')&(scii.AdmitWard=='HH1M'))) &
        (~sci_test.AdmissionMethodDescription.isin(['BOOKED ADMISSION ', 'ELECTIVE PLANNED']))
    ) & (sci_test.Age >= 18)]

    unseen_idx = sci_test[~sci_test.PatientNumber.isin(sci_train.PatientNumber)].index
    sci_test_precovid = sci_test[sci_test.AdmissionDateTime < '2020-03-01']

    return sci_train.index, sci_test.index, unseen_idx, sci_test_precovid.index


def plot_shap_features_joint(
    shap_values,
    title=None,
    max_display=20,
    figsize=(16, 8),
    bar_aspect=0.045,
    wspace=-0.3,
    topadjust=0.93,
    save=None,
    save_format="png",
):
    sns.set_style("whitegrid")
    fig = plt.figure(figsize=figsize)

    ax1 = fig.add_subplot(122, aspect="auto")
    shap.summary_plot(
        shap_values,
        max_display=max_display,
        show=False,
        plot_size=None,
        cmap=plt.get_cmap("coolwarm"),
    )
    ax1.set_yticklabels([])
    ax1.tick_params(axis="both", which="major", labelsize=16)
    ax1.tick_params(axis="both", which="minor", labelsize=14)
    ax1.set_xlabel("(b) Episode-individual SHAP value", fontsize=16)
    ax1.set_xlim((-4, 4))

    ax2 = fig.add_subplot(121, aspect=bar_aspect)
    shap.summary_plot(
        shap_values,
        plot_type="bar",
        plot_size=None,
        max_display=max_display,
        show=False,
        color="purple",
    )
    ax2.set_xlabel("(a) Mean absolute SHAP value", fontsize=16)
    ax2.tick_params(axis="both", which="major", labelsize=16)
    ax2.tick_params(axis="both", which="minor", labelsize=14)
    plt.tight_layout()
    plt.subplots_adjust(wspace=wspace)
    plt.suptitle(title, fontsize=20)
    plt.subplots_adjust(top=topadjust)
    if save:
        plt.savefig(
            save,
            bbox_inches="tight",
            dpi=200 if save_format != "svg" else None,
            format=save_format,
        )