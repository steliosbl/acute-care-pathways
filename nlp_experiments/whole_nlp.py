import numpy as np
import pandas as pd

import argparse
import torch
import torch.nn as nn
import os
import pickle

from sklearn.impute import SimpleImputer

from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset, interleave_datasets
import evaluate

from salford_datasets.salford import SalfordData, SalfordFeatures
from acd_experiment.sci import SCIData, SCICols
from acd_experiment.salford_adapter import SalfordAdapter


def get_text_dataset(data_path, select_features, outcome, old_only=False, old_data_path=None, column_name=False,
                     impute=False, extracted_entity_path=None, split='whole'):
    """ Load SalfordData and convert tabular data to a single text string
    :param data_path: str, path to Salford HDF5 file
    :param select_features: Type of features to use
    :param outcome: Type of outcome to select as CriticalEvent
    :param old_only: if True, use only records present in original data
    :param old_data_path: Path to the original data, must be passed if old_only is True
    :param column_name: if True, prepend column name before value in text representation
    :param impute: if True, replace missing values with median
    :param extracted_entity_path: str, if not None, load extracted entities from path and append to data
    :param split: str, 'whole', 'val', 'train'
    :return SalfordData:
    """
    dataset = SalfordData.from_raw(pd.read_hdf(data_path, key='table'))
    dataset = dataset.augment_derive_all()
    dataset = dataset.inclusion_exclusion_criteria()

    if split == 'train':
        dataset = SalfordData(dataset[dataset['AdmissionDate'] < pd.to_datetime('2020-02-01')])
    elif split == 'val':
        # Perhaps we want to change this? Has the issue that all val set patients are during covid...
        dataset = SalfordData(dataset[dataset['AdmissionDate'] >= pd.to_datetime('2020-02-01')])

    if old_only:
        print("----- Using only patients in both datasets")
        # Get original data
        scii = (
            SCIData(
                SCIData.quickload(old_data_path).sort_values(
                    "AdmissionDateTime"
                )
            )
            .mandate(SCICols.news_data_raw)
            .derive_ae_diagnosis_stems(onehot=False)
        )

        dataset = SalfordAdapter(dataset.loc[np.intersect1d(dataset.index, scii.SpellSerial)])

    # Get the columns to keep
    # TODO: Is there a better way to do this with SalfordFeatures?
    scores_with_notes_labs_and_hospital = ['Obs_RespiratoryRate_Admission', 'Obs_BreathingDevice_Admission',
                                           'Obs_O2Sats_Admission', 'Obs_Temperature_Admission',
                                           'Obs_SystolicBP_Admission', 'Obs_DiastolicBP_Admission',
                                           'Obs_HeartRate_Admission', 'Obs_AVCPU_Admission',
                                           'Obs_Pain_Admission', 'Obs_Nausea_Admission', 'Obs_Vomiting_Admission',
                                           'Female', 'Age', 'Blood_Haemoglobin_Admission', 'Blood_Urea_Admission',
                                           'Blood_Sodium_Admission', 'Blood_Potassium_Admission',
                                           'Blood_Creatinine_Admission', 'AE_PresentingComplaint',
                                           'AE_MainDiagnosis', 'SentToSDEC', 'Readmission', 'AdmitMethod',
                                           'AdmissionSpecialty']
    new_features = ['Blood_DDimer_Admission', 'Blood_CRP_Admission', 'Blood_Albumin_Admission',
                    'Blood_WhiteCount_Admission', 'Waterlow_Score', 'CFS_Score', 'CharlsonIndex']
    if select_features == 'sci':
        columns = scores_with_notes_labs_and_hospital
    elif select_features == 'sci_no_adm':
        columns = scores_with_notes_labs_and_hospital[:-1]
    elif select_features == 'new':
        columns = scores_with_notes_labs_and_hospital + new_features
    elif select_features == 'new_no_adm':
        columns = scores_with_notes_labs_and_hospital[:-1] + new_features
    elif select_features == 'new_no_adm_triagenotes':
        columns = scores_with_notes_labs_and_hospital[:-1] + new_features
    elif select_features == 'new_triagenotes':
        columns = scores_with_notes_labs_and_hospital + new_features
    elif select_features == 'sci_triagenotes':
        columns = scores_with_notes_labs_and_hospital
    elif select_features == 'new_diag':
        columns = scores_with_notes_labs_and_hospital + new_features + SalfordFeatures.Diagnoses
        dataset = dataset.expand_icd10_definitions()
    else:
        columns = None

    # Derive the required CriticalEvent (if los, this isn't needed)
    outcome_col = "CriticalEvent"
    if outcome == 'strict':
        dataset = dataset.derive_critical_event(wards=["CCU"], ignore_admit_ward=True)
    elif outcome == 'h1':
        dataset = dataset.derive_critical_event(wards=["CCU", "HH1M"], ignore_admit_ward=True)
    elif outcome == 'direct':
        dataset = dataset.derive_critical_event(wards=["CCU"], ignore_admit_ward=False)
    elif outcome == 'sci':
        dataset = dataset.derive_critical_event(wards=["CCU", "HH1M"], ignore_admit_ward=False)
    elif outcome == 'los':
        outcome_col = 'LOSBand'
    elif outcome == 'readm':
        outcome_col = 'Readmission'
    elif outcome == "Readmitted":
        dataset = SalfordData(dataset.derive_is_readmitted())
        outcome_col = "Readmitted"
    elif outcome == "ReadmittedPneumonia":
        dataset = SalfordData(dataset.derive_is_readmitted_reason(col_name="ReadmittedPneumonia"))
        outcome_col = "ReadmittedPneumonia"

    if impute:
        number_columns = dataset.select_dtypes(include=[float, int]).columns

        imp = SimpleImputer(strategy='median', missing_values=np.nan)
        dataset[number_columns] = imp.fit_transform(dataset[number_columns])

    if extracted_entity_path:
        # Place this at the start so it doesn't get cut off
        columns.append(columns[0])
        columns[0] = 'extracted entities'

        with open(extracted_entity_path, 'rb') as f:
            extracted_entities = pickle.load(f)

        if len(extracted_entities) != len(dataset):
            raise RuntimeError("extracted_entities must be of the same lengths as the Salford dataset!")

        # Currently only supports entities extracted with MedCAT. Get entities which are either disorders or findings
        # Just place them in one column as text
        extracted_disorders = {}
        for index, e in enumerate(extracted_entities):
            index = dataset.index[index]
            extracted_disorders[index] = ""
            for i in e['entities']:
                if 'disorder' in e['entities'][i]['types'] or 'finding' in e['entities'][i]['types']:
                    extracted_disorders[index] += e['entities'][i]['pretty_name'] + '; '

        dataset['extracted entities'] = list(extracted_disorders.values())

    if '_triagenotes' in select_features:
        # Separate TriageNote from tabular data with [SEP], then commas for tabular columns
        text_series = dataset.to_text(['AE_TriageNote'], column_name=column_name, inplace=False)['text']
        dataset = dataset.to_text(columns, column_name=column_name, target_col=outcome_col, sep_token=', ')
        dataset['text'] = text_series + dataset['text']
    else:
        dataset = dataset.to_text(columns=columns, column_name=column_name, target_col=outcome_col, sep_token=', ')


    return dataset


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss
        loss_fct = nn.CrossEntropyLoss(weight=torch.tensor([0.5026853973129243, 93.59609375]).cuda())
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def compute_metric(eval_pred, metrics=None, multiclass=False):
    """ Compute given metrics given outputs from a HF model
    :param eval_pred: Output from a HF model
    :param metrics: List[str] of metrics to compute, if None use a set of defaults for binary classification
    :param multiclass: bool, if True then use multiclass metrics (notably F1 score)
    :return: metrics
    """

    if metrics is None:
        metrics = ["accuracy", "f1", "precision", "recall"]

    average = 'macro' if multiclass else 'binary'

    metric = evaluate.combine(metrics)
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, 1)

    return metric.compute(predictions=predictions, references=labels, average=average)


def main():
    """
    Replicate the old ACD experiments, but use an NLP model instead of LGBMs etc.
    """
    parser = argparse.ArgumentParser(description='Train a transformer on some combination of the SalfordData dataset')

    parser.add_argument('--data-path', type=str, help='Path to the raw dataset HDF5 file', required=True)
    parser.add_argument("--old-data-path", type=str, help="Path to the old dataset HDF5")
    parser.add_argument("--select-features", help="Limit feature groups",
                        choices=['all', 'sci', 'sci_no_adm', 'new', 'new_no_adm', 'new_triagenotes', 'sci_triagenotes',
                                 'new_diag', 'sci_diag', 'new_no_adm_triagenotes'], default='all')
    parser.add_argument("--outcome", help="Outcome to predict", choices=['strict', 'h1', 'direct', 'sci', 'Readmitted',
                                                                         'ReadmittedPneumonia'],
                        default='sci')
    parser.add_argument("--old-only", help="Use only patients in the original dataset", action="store_true")
    parser.add_argument("--sampling", help="Use under/oversampling", choices=['under', 'over', None], default=None)
    parser.add_argument("--column-name", help="Prepend column name before value in text representation",
                        action="store_true")
    parser.add_argument("--impute", action="store_true", help="Impute missing data")
    parser.add_argument("--extracted-entity-path", type=str, default=None,
                        help="Path to extracted entities from triage notes. If None, don't use any")

    parser.add_argument('--model-name', type=str,
                        default='ml4pubmed/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext_pub_section',
                        help='Huggingface model tag, or path to a local model')
    parser.add_argument('--batch-size', type=int, default=16, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=2e-5, help='learning rate (default: 2e-5)')
    parser.add_argument('--save-path', type=str, default='./', help='Directory to save model to')

    args = parser.parse_args()

    dataset = get_text_dataset(args.data_path, args.select_features, args.outcome, args.old_only, args.old_data_path,
                               args.column_name, args.impute, args.extracted_entity_path, split='train')
    num_labels = len(dataset['label'].unique())

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

    # If we're under/oversampling, we don't want to use weighted loss
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

    save_path = os.path.join(args.save_path, f'{args.model_name}-finetuned-salford-alltext')
    trainer.save_model(save_path)

    print(f'----------- Model saved to {save_path}')


if __name__ == '__main__':
    main()
