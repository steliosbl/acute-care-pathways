import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import argparse
import torch
import torch.nn as nn
import os

import matplotlib.pyplot as plt
import seaborn as sns

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
    fbeta_score,
    roc_curve,
    average_precision_score,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    confusion_matrix
)

from .whole_nlp import get_text_dataset
from acd_experiment.utils.systematic_comparison import get_metrics


def get_multiclass_metrics(y_true, y_pred, y_pred_proba):
    """
    Same as get_metrics in acd_experiment.utils.systematic_comparison, but for multiclass targets
    Not modified get_metrics to keep compatability. Using best options for imbalanced data
    """
    return dict(
        AUC=roc_auc_score(y_true, y_pred_proba, multi_class='ovo', average='macro'),
        AP=average_precision_score(y_true, y_pred_proba, average='weighted'),
        Accuracy=accuracy_score(y_true, y_pred),
        Precision=precision_score(y_true, y_pred, average='macro'),
        Recall=recall_score(y_true, y_pred, average='macro'),
        F2=fbeta_score(y_true, y_pred, beta=2, average='macro'),
    )


def evaluate_from_pred(y_true, y_pred, y_pred_proba, save=None):
    metrics = get_metrics(y_true, y_pred, y_pred_proba)

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    roc_fig = RocCurveDisplay.from_predictions(
        y_true, y_pred_proba, ax=ax[0],  # pos_label=pos_label
    )
    pr_fig = PrecisionRecallDisplay.from_predictions(
        y_true, y_pred_proba, ax=ax[1],
    )

    cm = confusion_matrix(y_true, y_pred)
    cm_g = sns.heatmap(cm, square=True, cmap='Blues', annot=True, fmt='.0f', ax=ax[2])
    cm_g.set(xlabel='Predicted Label', ylabel='True Label')

    if save:
        plt.savefig(save, bbox_inches="tight", dpi=100)

    return metrics, roc_fig, pr_fig, cm_g


def main():
    """
        Evaluate trained HF models
    """
    parser = argparse.ArgumentParser(description='Evaluate trained HF models')

    parser.add_argument('--data-path', type=str, help='Path to the raw dataset HDF5 file', required=True)
    parser.add_argument("--old-data-path", type=str, help="Path to the old dataset HDF5")
    parser.add_argument("--select-features", help="Limit feature groups",
                        choices=['all', 'sci', 'sci_no_adm', 'new', 'new_no_adm', 'new_triagenotes', 'sci_triagenotes',
                                 'new_diag', 'sci_diag', 'new_no_adm_triagenotes'], default='all')
    parser.add_argument("--outcome", help="Outcome to predict", choices=['strict', 'h1', 'direct', 'sci', 'Readmitted',
                                                                         'ReadmittedPneumonia'], default='sci')
    parser.add_argument("--old-only", help="Use only patients in the original dataset", action="store_true")
    parser.add_argument("--column-name", help="Prepend column name before value in text representation",
                        action="store_true")
    parser.add_argument("--extracted-entity-path", type=str, default=None,
                        help="Path to extracted entities from triage notes. If None, don't use any")

    parser.add_argument('--model-name', type=str,
                        default='ml4pubmed/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext_pub_section',
                        help='Huggingface model tag, or path to a local model')
    parser.add_argument('--batch-size', type=int, default=16, help='Training batch size')
    parser.add_argument('--save-path', type=str, default='./', help='Directory to save results to')

    args = parser.parse_args()

    dataset = get_text_dataset(args.data_path, args.select_features, args.outcome, args.old_only, args.old_data_path,
                               args.column_name, extracted_entity_path=args.extracted_entity_path, split='val')

    # Get model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, device=0)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, output_hidden_states=False,
                                                               num_labels=2, ignore_mismatched_sizes=True)

    encoded_dataset = Dataset.from_pandas(dataset)
    encoded_dataset = encoded_dataset.class_encode_column('label')
    #encoded_dataset = encoded_dataset.train_test_split(test_size=0.2, shuffle=True, stratify_by_column='label')

    clf_pipe = pipeline('text-classification', model=model, tokenizer=tokenizer, device=0, max_length=512,
                        truncation=True)
    pred_labels = []
    pred_proba = []

    for out in tqdm(clf_pipe(KeyDataset(encoded_dataset, "text"), batch_size=args.batch_size),
                    total=len(encoded_dataset)):
        pred_labels.append(int(out['label'][-1]))

        # If output label is 0, get the opposite score for binaery-classification like eval
        if int(out['label'][-1]) == 0:
            pred_proba.append(1 - out['score'])
        else:
            pred_proba.append(out['score'])

    metrics = get_metrics(pd.DataFrame(encoded_dataset['test']['label']), pred_labels, pred_proba)

    # Add sensitivity, which isn't included in get_metrics
    # (Not added it there to keep compatability with old code)
    # Remember that specificity is the sensitivity of the negative class
    spec = recall_score(encoded_dataset['test']['label'], pred_labels, pos_label=0)
    metrics['specificity'] = spec

    text = f"-------- {args.model_name} EVALn ---------\n\n{metrics}"

    print(text)

    save_path = os.path.join(args.save_path, 'results.txt')
    with open(save_path, 'w') as f:
        f.write(text)

    print(f"--- Saved results to {save_path}")

    save_path = os.path.join(args.save_path, 'fig.png')
    evaluate_from_pred(pd.DataFrame(encoded_dataset['test']['label']), pred_labels, pred_proba,
                       save=save_path)
    print(f"--- Saved figure to {save_path}")


if __name__ == '__main__':
    main()
