import numpy as np
import pandas as pd

import argparse
import torch
import os

from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
import evaluate

from salford_datasets.salford import SalfordData, SalfordFeatures
from acd_experiment.sci import SCIData, SCICols
from acd_experiment.salford_adapter import SalfordAdapter


def compute_metric(eval_pred):
    metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, 1)

    return metric.compute(predictions=predictions, references=labels)


def main():
    """
    Replicate the old ACD experiments, but use an NLP model instead of LGBMs etc.
    """
    parser = argparse.ArgumentParser(description='Train a transformer on some combination of the SalfordData dataset')

    parser.add_argument('--data-path', type=str, help='Path to the raw dataset HDF5 file', required=True)
    parser.add_argument("--old-data-path", type=str, help="Path to the old dataset HDF5")
    parser.add_argument("--select-features", help="Limit feature groups",
                        choices=['all', 'sci', 'sci_no_adm', 'new', 'new_no_adm'], default='all')
    parser.add_argument("--outcome", help="Outcome to predict", choices=['strict', 'h1', 'direct', 'sci'],
                        default='sci')
    parser.add_argument("--old-only", help="Use only patients in the original dataset", action="store_true")

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

    if args.old_only:
        print("----- Using only patients in both datasets")
        # Get original data
        scii = (
            SCIData(
                SCIData.quickload(args.old_data_path).sort_values(
                    "AdmissionDateTime"
                )
            )
                .mandate(SCICols.news_data_raw)
                .derive_ae_diagnosis_stems(onehot=False)
        )

        dataset = SalfordAdapter(dataset.loc[np.intersect1d(dataset.index, scii.SpellSerial)])

    # Get the columns to keep
    # TODO: Is there a better way to do this with SalfordFeatures?
    scores_with_notes_labs_and_hospital = ['NEWS_RespiratoryRate_Admission', 'NEWS_O2Sat_Admission',
                                           'NEWS_Temperature_Admission', 'NEWS_BP_Admission',
                                           'NEWS_HeartRate_Admission', 'NEWS_AVCPU_Admission',
                                           'NEWS_BreathingDevice_Admission', 'Female', 'Age',
                                           'Blood_Haemoglobin_Admission', 'Blood_Urea_Admission',
                                           'Blood_Sodium_Admission', 'Blood_Potassium_Admission',
                                           'Blood_Creatinine_Admission', 'AE_PresentingComplaint', 'AE_MainDiagnosis',
                                           'SentToSDEC', 'Readmission', 'AdmitMethod', 'AdmissionSpecialty']
    new_features = ['Blood_DDimer_Admission', 'Blood_CRP_Admission', 'Blood_Albumin_Admission',
                    'Blood_WhiteCount_Admission', 'Waterlow_Score', 'CFS_Score', 'CharlsonIndex']

    if args.select_features == 'sci':
        columns = scores_with_notes_labs_and_hospital
    elif args.select_features == 'sci_no_adm':
        columns = scores_with_notes_labs_and_hospital[:-1]
    elif args.select_features == 'new':
        columns = scores_with_notes_labs_and_hospital + new_features
    elif args.select_features == 'new_no_adm':
        columns = scores_with_notes_labs_and_hospital[:-1] + new_features
    else:
        columns = None

    # Derive the required CriticalEvent
    if args.outcome == 'strict':
        dataset = dataset.derive_critical_event(wards=["CCU"], ignore_admit_ward=True)
    elif args.outcome == 'h1':
        dataset = dataset.derive_critical_event(wards=["CCU", "HH1M"], ignore_admit_ward=True)
    elif args.outcome == 'direct':
        dataset = dataset.derive_critical_event(wards=["CCU"], ignore_admit_ward=False)
    else:
        dataset = dataset.derive_critical_event(wards=["CCU", "HH1M"], ignore_admit_ward=False)

    dataset = dataset.to_text(columns=columns)

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

    # Set up training
    training_args = TrainingArguments(
        f"{args.model_name}-finetuned-salford-alltext-{args.select_features}-{args.outcome}",
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

    trainer = Trainer(
        model,
        training_args,
        train_dataset=encoded_dataset['train'],
        eval_dataset=encoded_dataset['test'],
        tokenizer=tokenizer,
        compute_metrics=compute_metric
    )

    trainer.train()

    save_path = os.path.join(args.save_path, f'{args.model_name}-finetuned-salford-alltext')
    trainer.save_model(save_path)

    print(f'----------- Model saved to {save_path}')


if __name__ == '__main__':
    main()
