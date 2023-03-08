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

from .whole_nlp import get_text_dataset
from acd_experiment.utils.systematic_comparison import get_metrics


def main():
    """
        Evaluate trained HF models
    """
    parser = argparse.ArgumentParser(description='Evaluate trained HF models')

    parser.add_argument('--data-path', type=str, help='Path to the raw dataset HDF5 file', required=True)
    parser.add_argument("--old-data-path", type=str, help="Path to the old dataset HDF5")
    parser.add_argument("--select-features", help="Limit feature groups",
                        choices=['all', 'sci', 'sci_no_adm', 'new', 'new_no_adm'], default='all')
    parser.add_argument("--outcome", help="Outcome to predict", choices=['strict', 'h1', 'direct', 'sci'],
                        default='sci')
    parser.add_argument("--old-only", help="Use only patients in the original dataset", action="store_true")
    parser.add_argument("--column-name", help="Prepend column name before value in text representation",
                        action="store_true")

    parser.add_argument('--model-name', type=str,
                        default='ml4pubmed/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext_pub_section',
                        help='Huggingface model tag, or path to a local model')
    parser.add_argument('--batch-size', type=int, default=16, help='Training batch size')
    parser.add_argument('--save-path', type=str, default='./', help='Directory to save results to')

    args = parser.parse_args()

    dataset = get_text_dataset(args.data_path, args.select_features, args.outcome, args.old_only, args.old_data_path,
                               args.column_name)

    # Get model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, device=0)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, output_hidden_states=False,
                                                               num_labels=2, ignore_mismatched_sizes=True)

    encoded_dataset = Dataset.from_pandas(dataset)
    encoded_dataset = encoded_dataset.class_encode_column('label')
    encoded_dataset = encoded_dataset.train_test_split(test_size=0.2, shuffle=True, stratify_by_column='label')

    clf_pipe = pipeline('text-classification', model=model, tokenizer=tokenizer)
    pred_labels = []
    pred_proba = []

    for out in tqdm(clf_pipe(KeyDataset(encoded_dataset['test'], "text"), batch_size=args.batch_size),
                    total=len(encoded_dataset['test'])):
        pred_labels.append(int(out['label'][-1]))
        pred_proba.append(out['score'])

    metrics = get_metrics(pd.DataFrame(encoded_dataset['test']['label']), pred_labels, pred_proba)

    text = f"-------- {args.model_name} EVALn ---------\n\n{metrics}"

    print(text)

    save_path = os.path.join(args.save_path, 'results.txt')
    with open(save_path, 'w') as f:
        f.write(text)

    print(f"--- Saved results to {save_path}")


if __name__ == '__main__':
    main()
