import numpy as np
import pandas as pd

import argparse
import torch
import os

from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset, interleave_datasets
import evaluate

from salford_datasets.salford import SalfordData, SalfordFeatures
from acd_experiment.sci import SCIData, SCICols
from acd_experiment.salford_adapter import SalfordAdapter
from .whole_nlp import compute_metric, CustomTrainer


def main():
    """
    Run NLP classifiers on text fields only (+ maybe demographics)
    """
    parser = argparse.ArgumentParser(description='Train a transformer on text fields in SalfordData')

    parser.add_argument('--data-path', type=str, help='Path to the raw dataset HDF5 file', required=True)
    parser.add_argument('--text-features', choices=['triage', 'triage_diagnosis', 'triage_complaint'],
                        default='triage', help='The combination of text features to include')
    parser.add_argument('--demographics', action='store_true', help='Include (text-encoded) patient demographics')
    parser.add_argument("--sampling", help="Use under/oversampling", choices=['under', 'over', None], default=None)
    parser.add_argument("--column-name", help="Prepend column name before value in text representation",
                        action="store_true")


    parser.add_argument('--model-name', type=str,
                        default='ml4pubmed/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext_pub_section',
                        help='Huggingface model tag, or path to a local model')
    parser.add_argument('--batch-size', type=int, default=16, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate (default: 2e-5)')
    parser.add_argument('--save-path', type=str, default='./', help='Directory to save model to')

    args = parser.parse_args()

    dataset = SalfordData.from_raw(pd.read_hdf(args.data_path, key='table'))
    dataset = SalfordData(dataset[dataset['Age'] >= 18])
    dataset = dataset.augment_derive_all()

    # Get the columns to keep
    if args.text_features == 'triage':
        columns = ['AE_TriageNote']
    elif args.text_features == 'triage_diagnosis':
        columns = ['AE_TriageNote', 'AE_MainDiagnosis']
    else:
        columns = ['AE_TriageNote', 'AE_PresentingComplaint']

    if args.demographics:
        columns += SalfordFeatures.Demographics

    # Derive the required CriticalEvent
    dataset = dataset.derive_critical_event(wards=["CCU", "HH1M"], ignore_admit_ward=False)

    dataset = dataset.to_text(columns=columns, column_name=args.column_name)

    # Get model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, device=0)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, output_hidden_states=False,
                                                                num_labels=2, ignore_mismatched_sizes=True)

    # Tokenize the text
    encoded_dataset = Dataset.from_pandas(dataset)
    encoded_dataset = encoded_dataset.remove_columns('SpellSerial')
    encoded_dataset = encoded_dataset.map(lambda x: tokenizer(x['text'], padding='max_length', max_length=512,
                                                              truncation=True, return_tensors='pt'),
                                          batched=True)

    encoded_dataset = encoded_dataset.class_encode_column('label')

    # Split into train test splits
    encoded_dataset = encoded_dataset.train_test_split(test_size=0.2, shuffle=True, stratify_by_column='label')

    if args.sampling:
        stopping_strat = "all_exhausted" if args.sampling == 'over' else "first_exhausted"
        probs = [0.4, 0.6] if args.sampling == "over" else [0.5, 0.5]
        encoded_dataset_neg = encoded_dataset['train'].filter(lambda x: x['label'] == 0)
        encoded_dataset_pos = encoded_dataset['train'].filter(lambda x: x['label'] == 1)

        encoded_dataset['train'] = interleave_datasets([encoded_dataset_neg, encoded_dataset_pos], probabilities=probs,
                                                       stopping_strategy=stopping_strat)

    # Set up training
    training_args = TrainingArguments(
        f"{args.model_name}-finetuned-salford-textonly-{args.text_features}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        push_to_hub=False,
        report_to='tensorboard'
    )

    if args.sampling:
        trainer = Trainer(
            model,
            training_args,
            train_dataset=encoded_dataset['train'],
            eval_dataset=encoded_dataset['test'],
            tokenizer=tokenizer,
            compute_metrics=compute_metric
        )
    else:
        trainer = CustomTrainer(
            model,
            training_args,
            train_dataset=encoded_dataset['train'],
            eval_dataset=encoded_dataset['test'],
            tokenizer=tokenizer,
            compute_metrics=compute_metric
        )

    trainer.train()

    save_path = os.path.join(args.save_path, f"{args.model_name}-finetuned-salford-textonly-{args.text_features}")
    trainer.save_model(save_path)

    print(f'----------- Model saved to {save_path}')


if __name__ == '__main__':
    main()
