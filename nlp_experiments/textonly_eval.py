import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import argparse
import torch
import torch.nn as nn
import os

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset

from .nlp_textonly import get_textonly_dataset
from acd_experiment.utils.systematic_comparison import get_metrics
from .binary_eval import evaluate_from_pred


def main():
    """
        Evaluate trained HF models
    """
    parser = argparse.ArgumentParser(description='Evaluate trained HF models on text only')

    parser.add_argument('--data-path', type=str, help='Path to the raw dataset HDF5 file', required=True)
    parser.add_argument('--text-features', choices=['triage', 'triage_diagnosis', 'triage_complaint'],
                        default='triage', help='The combination of text features to include')
    parser.add_argument('--demographics', action='store_true', help='Include (text-encoded) patient demographics')
    parser.add_argument("--target", choices=["CriticalEvent", "Ethnicity", "SentToSDEC", "LOSBand", "Readmission",
                                             "ReadmissionBand", "ReadmissionPneumonia", "EthnicityHA",
                                             "Readmitted", "ReadmittedPneumonia"],
                        default="CriticalEvent", help="Target variable to predict")

    parser.add_argument('--model-name', type=str,
                        default='ml4pubmed/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext_pub_section',
                        help='Huggingface model tag, or path to a local model')
    parser.add_argument('--batch-size', type=int, default=16, help='Training batch size')
    parser.add_argument('--save-path', type=str, default='./', help='Directory to save results to')

    args = parser.parse_args()

    multiclass = args.target not in ["CriticalEvent", "SentToSDEC", "Readmission", "ReadmissionPneumonia", "Readmitted",
                                     "ReadmittedPneumonia"]
    dataset, _, _, num_labels = get_textonly_dataset(args.data_path, args.target, args.text_features, args.demographics)

    # Get model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, device=0)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, output_hidden_states=False,
                                                               num_labels=num_labels, ignore_mismatched_sizes=True)

    # Tokenize the text
    encoded_dataset = Dataset.from_pandas(dataset)
    encoded_dataset = encoded_dataset.remove_columns('SpellSerial')
    encoded_dataset = encoded_dataset.class_encode_column('label', include_nulls=True)
    encoded_dataset = encoded_dataset.train_test_split(test_size=0.2, shuffle=True, stratify_by_column='label')

    clf_pipe = pipeline('text-classification', model=model, tokenizer=tokenizer, device=0)
    pred_labels = []
    pred_proba = []

    for out in tqdm(clf_pipe(KeyDataset(encoded_dataset['test'], "text"), batch_size=args.batch_size),
                    total=len(encoded_dataset['test'])):
        pred_labels.append(int(out['label'][-1]))

        # If output label is 0, get the opposite score for binary-classification like eval
        if num_labels == 2:
            if int(out['label'][-1]) == 0:
                pred_proba.append(1 - out['score'])
            else:
                pred_proba.append(out['score'])
        else:
            pred_proba.append(out['score'])

    if multiclass:
        metrics = []
    else:
        metrics = get_metrics(pd.DataFrame(encoded_dataset['test']['label']), pred_labels, pred_proba)

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
