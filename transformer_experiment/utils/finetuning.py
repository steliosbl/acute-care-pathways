import argparse

import numpy as np

import torch
from torch.nn import functional as F

from salford_datasets.utils import DotDict

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
    fbeta_score,
    average_precision_score,
)

_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

BERTModels = DotDict(
    BioClinicalBert="emilyalsentzer/Bio_ClinicalBERT",
    Bert="distilbert-base-uncased",
    PubMedBert="ml4pubmed/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext_pub_section"
)

def split_into_batches(Xt, batch_size):
    """ Given a tensor/ndarray and a batch size, splits it into batches of size up to batch_size along the first dimension """
    return np.array_split(
        Xt, np.ceil(len(Xt)/batch_size)
    )
    
def split_dict_into_batches(d, batch_size):
    contents = {
        k: split_into_batches(v, batch_size) for k, v in d.items()
    }
    return [dict(zip(contents,t)) for t in zip(*contents.values())]

def load_dict_to_device(d, device=_DEVICE):
    return {
        k: v.to(device) for k, v in d.items()
    }


def bert_finetuning_metrics(eval_pred):
    predictions, y_true = eval_pred
    y_pred = np.argmax(predictions, axis=1)
    y_pred_proba = F.softmax(torch.from_numpy(predictions), dim=1)[:,1]

    #print(classification_report(y_true, y_pred, target_names=list(RelatedArticleDataset.STANCE_LABELS.keys())))
    return dict(
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred),
        recall=recall_score(y_true, y_pred),
        f2=fbeta_score(y_true, y_pred, beta=2),
        AUC=roc_auc_score(y_true, y_pred_proba),
        AP=average_precision_score(y_true, y_pred_proba)
    )


def construct_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", help="Can be 'BioClinicalBert', 'Bert', or 'PubMedBert'", type=str, default="BioClinicalBert"
    )
    parser.add_argument(
        '-b', '--batch_size', help="The batch size to use. Default=56", type=int, default=56
    )
    parser.add_argument(
        '-e', '--experiment', help="The experiment number corresponding to a feature set. Can be '41', '42',...", type=str, default="41"
    )
    
    parser.add_argument(
        "-d",
        "--debug",
        help="Whether to only use a small subset of data for debugging",
        action="store_true",
    )

    return parser