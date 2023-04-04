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


def get_textonly_dataset(data_path, target, text_features, demographics=True, column_name=True):
    """
    Load SalfordDataset with deired text features
    :param data_path: str, Path to HDF5 file
    :param target: str, Chosen target variable. Must be one of ["CriticalEvent", "Ethnicity", "SentToSDEC", "LOSBand",
    "Readmission"]
    :param text_features: str, Features to get. Must be one of ['triage', 'triage_diagnosis', 'triage_complaint']
    :param demographics: bool, If True append patient demographics as text
    :param column_name: bool, If True prepend column names before each value
    :return: pd.DataFrame of desired features, str of the metric to use when choosing the best model,
    int of number of distinct labels in the dataset
    """
    dataset = SalfordData.from_raw(pd.read_hdf(data_path, key='table'))
    dataset = dataset.augment_derive_all()

    # If doing binary classification, we can use the default metrics
    metrics = None
    best_metric = 'f1'
    num_labels = 2
    if target == "Ethnicity":
        dataset.group_ethnicity(5)
        metrics = ['accuracy']
        best_metric = metrics[0]
        num_labels = len(dataset[target].unique())
    elif target == "EthnicityHA":
        # Only keep patients who were diagnoses with a heart attack, contains accepts regex
        dataset = SalfordData(dataset[dataset["MainICD10"].str.contains('I20|I21|I22|I23|I24', na=False)])

        # Rename to actual target column
        target = "Ethnicity"

        dataset = SalfordData(dataset.group_ethnicity(100))
        metrics = ['accuracy']
        best_metric = 'loss'
        num_labels = len(dataset[target].unique())
    elif target == "LOSBand":
        metrics = ['accuracy']
        best_metric = metrics[0]
        num_labels = len(dataset[target].unique())
    elif target == "ReadmissionBand":
        # the .fillna call in derive_readmission_band means a SalfordData instance isn't returned
        dataset = SalfordData(dataset.derive_readmission_band())

        metrics = ['accuracy']
        best_metric = 'loss'
        num_labels = len(dataset[target].unique())

        # dataset['ReadmissionBand'] is a pd.Categorical, which HF doesn't support
        dataset['ReadmissionBand'] = dataset['ReadmissionBand'].cat.codes
    elif target == "ReadmissionPneumonia":
        dataset = SalfordData(dataset.derive_readmission_reason(col_name="ReadmissionPneumonia"))
    elif target == "Readmitted":
        dataset = SalfordData(dataset.derive_is_readmitted())
    elif target == "ReadmittedPneumonia":
        dataset = SalfordData(dataset.derive_is_readmitted_reason(col_name="ReadmittedPneumonia"))
    elif target == "CriticalEvent":
        dataset = dataset.derive_critical_event(wards=["CCU", "HH1M"], ignore_admit_ward=False)

    # Get the columns to keep
    if text_features == 'triage':
        columns = ['AE_TriageNote']
    elif text_features == 'triage_diagnosis':
        columns = ['AE_TriageNote', 'AE_MainDiagnosis']
    else:
        columns = ['AE_TriageNote', 'AE_PresentingComplaint']

    if demographics:
        demo_cols = SalfordFeatures.Demographics

        # Don't include ethnicity if that's our target
        if target == "Ethnicity":
            demo_cols.remove("Ethnicity")

        # Convert to text in this format: FREETEXT [SEP] Column1: Data1, Column2: Data2 ...
        text_series = dataset.to_text(columns, column_name=column_name, inplace=False)['text']
        dataset = dataset.to_text(demo_cols, column_name=column_name, target_col=target, sep_token=', ')
        dataset['text'] = text_series + dataset['text']
    else:
        dataset = dataset.to_text(columns=columns, column_name=column_name, target_col=target, sep_token=', ')

    return dataset, best_metric, metrics, num_labels


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
    parser.add_argument("--target", choices=["CriticalEvent", "Ethnicity", "SentToSDEC", "LOSBand", "Readmission",
                                             "ReadmissionBand", "ReadmissionPneumonia", "EthnicityHA",
                                             "Readmitted", "ReadmittedPneumonia"],
                        default="CriticalEvent", help="Target variable to predict")

    parser.add_argument('--model-name', type=str,
                        default='ml4pubmed/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext_pub_section',
                        help='Huggingface model tag, or path to a local model')
    parser.add_argument('--batch-size', type=int, default=16, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate (default: 2e-5)')
    parser.add_argument('--save-path', type=str, default='./', help='Directory to save model to')

    args = parser.parse_args()

    multiclass = args.target not in ["CriticalEvent", "SentToSDEC", "Readmission", "ReadmissionPneumonia", "Readmitted",
                                     "ReadmittedPneumonia"]
    dataset, best_metric, metrics, num_labels = get_textonly_dataset(args.data_path, args.target, args.text_features,
                                                                     args.demographics, args.column_name)

    # Get model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, device=0)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, output_hidden_states=False,
                                                               num_labels=num_labels, ignore_mismatched_sizes=True)

    # Tokenize the text
    encoded_dataset = Dataset.from_pandas(dataset)
    encoded_dataset = encoded_dataset.remove_columns('SpellSerial')
    encoded_dataset = encoded_dataset.map(lambda x: tokenizer(x['text'], padding='max_length', max_length=512,
                                                              truncation=True, return_tensors='pt'),
                                          batched=True)

    encoded_dataset = encoded_dataset.class_encode_column('label', include_nulls=True)

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
        metric_for_best_model=best_metric,
        push_to_hub=False,
        report_to='tensorboard'
    )

    if args.sampling or args.target != "CriticalEvent":
        trainer = Trainer(
            model,
            training_args,
            train_dataset=encoded_dataset['train'],
            eval_dataset=encoded_dataset['test'],
            tokenizer=tokenizer,
            compute_metrics=lambda x: compute_metric(x, metrics, multiclass=multiclass)
        )
    else:
        trainer = CustomTrainer(
            model,
            training_args,
            train_dataset=encoded_dataset['train'],
            eval_dataset=encoded_dataset['test'],
            tokenizer=tokenizer,
            compute_metrics=lambda x: compute_metric(x, multiclass=multiclass)
        )

    trainer.train()

    save_path = os.path.join(args.save_path, f"{args.model_name}-finetuned-salford-textonly-{args.text_features}")
    trainer.save_model(save_path)

    print(f'----------- Model saved to {save_path}')


if __name__ == '__main__':
    main()
