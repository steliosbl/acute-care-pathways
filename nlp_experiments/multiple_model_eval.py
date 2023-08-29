from salford_datasets.salford import SalfordData, SalfordFeatures
from acd_experiment.sci import SCIData, SCICols
from acd_experiment.salford_adapter import SalfordAdapter

import torch
import numpy as np
import pandas as pd

import os
import argparse

import matplotlib.pyplot as plt
import seaborn as sns

from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer, pipeline
from transformers.pipelines.pt_utils import KeyDataset
from datasets import Dataset
import evaluate

from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay, confusion_matrix, precision_recall_curve
from sklearn.metrics import recall_score, precision_score, fbeta_score, average_precision_score, roc_auc_score

from aif360.sklearn.metrics import generalized_entropy_error, between_group_generalized_entropy_error


from tqdm.auto import tqdm


def get_tabular_dataset(select_features, outcome, dataset, triage_note=True):
    # Get the columns to keep
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
    elif select_features == 'new_triagenotes':
        columns = scores_with_notes_labs_and_hospital + new_features
    elif select_features == 'sci_triagenotes':
        columns = scores_with_notes_labs_and_hospital
    elif select_features == 'new_diag':
        columns = scores_with_notes_labs_and_hospital + new_features + SalfordFeatures.Diagnoses
        dataset = dataset.expand_icd10_definitions()
    else:
        columns = None

    # Derive the required CriticalEvent
    if outcome == 'strict':
        dataset = dataset.derive_critical_event(wards=["CCU"], ignore_admit_ward=True)
    elif outcome == 'h1':
        dataset = dataset.derive_critical_event(wards=["CCU", "HH1M"], ignore_admit_ward=True)
    elif outcome == 'direct':
        dataset = dataset.derive_critical_event(wards=["CCU"], ignore_admit_ward=False)
    elif outcome in ['Readmitted', 'ReadmittedPneumonia']:
        pass
    else:
        dataset = dataset.derive_critical_event(wards=["CCU", "HH1M"], ignore_admit_ward=False)

    if triage_note:
        # Separate TriageNote from tabular data with [SEP], then commas for tabular columns
        text_series = dataset.to_text(['AE_TriageNote'], column_name=True, inplace=False)['text']
        dataset_tab_text = dataset.to_text(columns, column_name=True, sep_token=', ')
        dataset_tab_text['text'] = text_series + dataset_tab_text['text']
    else:
        dataset_tab_text = dataset.to_text(columns=columns, column_name=True, sep_token=', ')

    # Convert to HF dataset
    encoded_dataset = Dataset.from_pandas(dataset_tab_text)
    encoded_dataset = encoded_dataset.remove_columns('SpellSerial')
    encoded_dataset = encoded_dataset.class_encode_column('label')

    # Split into train test splits
    #dataset_tab_text_eval = encoded_dataset.train_test_split(test_size=0.2, shuffle=True, stratify_by_column='label')[
    #    'test']

    return encoded_dataset


def main():
    sns.set(style='white', palette='colorblind', rc={'figure.dpi': 150, 'savefig.dpi': 150})

    parser = argparse.ArgumentParser(description='Evaluate trained HF models')

    parser.add_argument('--data-path', type=str, help='Path to the raw dataset HDF5 file', required=True)
    parser.add_argument("--outcome", help="Outcome to predict", choices=['CriticalEvent', 'Readmitted',
                                                                         'ReadmittedPneumonia'], default='sci')

    parser.add_argument("--model-basepath", type=str, default='/home/matthew/phd/acute-care-pathways/runs/',
                        help="Basepath to model locations")
    parser.add_argument("--model-list", type=str, required=True, help="Path to CSV containing list of models to eval")
    parser.add_argument('--batch-size', type=int, default=16, help='Eval batch size')
    parser.add_argument('--save-path', type=str, default='./', help='Directory to save results to')

    args = parser.parse_args()

    # Load dataset
    dataset = SalfordData.from_raw(pd.read_hdf(args.data_path, key='table'))
    dataset = dataset.augment_derive_all()
    dataset = dataset.inclusion_exclusion_criteria()

    dataset = SalfordData(dataset[dataset['AdmissionDate'] >= pd.to_datetime('2020-02-01')])
    wholeset_n_days = (dataset.AdmissionDate.max() - dataset.AdmissionDate.min()).days

    # Get text-only eval dataset
    # Get the text-only dataset
    DEMOGRAPHICS = True

    if args.outcome == "ReadmissionBand":
        dataset = SalfordData(dataset.derive_readmission_band())
        dataset['ReadmissionBand'] = dataset['ReadmissionBand'].cat.codes
    elif args.outcome == "Readmitted":
        dataset = SalfordData(dataset.derive_is_readmitted())
    elif args.outcome == "ReadmittedPneumonia":
        dataset = SalfordData(dataset.derive_is_readmitted_reason(col_name="ReadmittedPneumonia"))
    elif args.outcome == "CriticalEvent":
        dataset = dataset.derive_critical_event(wards=["CCU", "HH1M"], ignore_admit_ward=False)

    if DEMOGRAPHICS:
        text_series = dataset.to_text(['AE_TriageNote'], column_name=True, inplace=False)['text']
        dataset_text = dataset.to_text(SalfordFeatures.Demographics, column_name=True, target_col=args.outcome,
                                       sep_token=', ')
        dataset_text['text'] = text_series + dataset_text['text']
    else:
        columns = ['AE_TriageNote', 'AE_PresentingComplaint']

        dataset_text = dataset.to_text(columns=columns, target_col=args.outcome, column_name=True, sep_token=', ')

    # Convert to HF dataset
    encoded_dataset = Dataset.from_pandas(dataset_text)
    encoded_dataset = encoded_dataset.remove_columns('SpellSerial')
    encoded_dataset = encoded_dataset.class_encode_column('label')
    num_labels = len(np.unique(encoded_dataset['label']))

    # Split into train test splits
    #dataset_text_eval = encoded_dataset.train_test_split(test_size=0.2, shuffle=True, stratify_by_column='label')[
    #    'test']
    dataset_text_eval = encoded_dataset

    # Load dataframe with models to evaluate
    model_list = pd.read_csv(args.model_list)

    # Go through and load the necessary datasets
    models = {}
    for i in range(len(model_list)):
        row = model_list.iloc[i]

        if row['Model Type'] == 'textonly':
            current_dataset = dataset_text_eval
        elif row['Model Type'] == 'tabonly':
            current_dataset = get_tabular_dataset(row['Select Features'], args.outcome, dataset, False)
        elif row['Model Type'] == 'tabandtext':
            current_dataset = get_tabular_dataset(row['Select Features'], args.outcome, dataset, True)
        else:
            raise NotImplementedError(f"Model Type in model list CSV must be one of: textonly, tabonly, tabandtext. "
                                      f"Not {row['Model Type']}")

        models[row['Model Name']] = [row['Model Path'], current_dataset]

    # Go through and evaluate all models
    all_pred_labels = {}
    all_pred_proba = {}

    for model_name in models:
        model_path = os.path.join(args.model_basepath, models[model_name][0])
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, device=0)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, output_hidden_states=False,
                                                                   num_labels=num_labels, ignore_mismatched_sizes=True)

        clf_pipe = pipeline('text-classification', model=model, tokenizer=tokenizer, device=0, max_length=512,
                            truncation=True)
        pred_labels = []
        pred_proba = []

        for out in tqdm(clf_pipe(KeyDataset(models[model_name][1], "text"), batch_size=16),
                        total=len(models[model_name][1])):
            pred_labels.append(int(out['label'][-1]))

            # If output label is 0, get the opposite score for binary-classification like eval
            if int(out['label'][-1]) == 0:
                pred_proba.append(1 - out['score'])
            else:
                pred_proba.append(out['score'])

        all_pred_labels[model_name] = pred_labels
        all_pred_proba[model_name] = pred_proba

    # Add results for NEWS2
    dataset_news = dataset.copy()
    dataset_news['label'] = dataset_news['CriticalEvent']
    dataset_news = dataset_news.dropna(subset=['label', 'NEWS_Score_Admission'])
    models['Reference (NEWS2)'] = [None, dataset_news]

    all_pred_labels['Reference (NEWS2)'] = dataset_news.NEWS_Score_Admission >= 7
    all_pred_proba['Reference (NEWS2)'] = dataset_news.NEWS_Score_Admission

    print("---- NEWS2 results")
    print(recall_score(models['Reference (NEWS2)'][1]['label'], all_pred_labels['Reference (NEWS2)']))
    print(precision_score(models['Reference (NEWS2)'][1]['label'], all_pred_labels['Reference (NEWS2)']))
    print(fbeta_score(models['Reference (NEWS2)'][1]['label'], all_pred_labels['Reference (NEWS2)'], beta=2))
    print(average_precision_score(models['Reference (NEWS2)'][1]['label'], all_pred_proba['Reference (NEWS2)']))
    print(roc_auc_score(models['Reference (NEWS2)'][1]['label'], all_pred_proba['Reference (NEWS2)']))

    # Plot ROC and PR curves
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))

    colours = sns.color_palette()[:len(models) - 1]

    # Manually change colour of reference
    colours.append('r')

    for model_name, colour in zip(models, colours):
        RocCurveDisplay.from_predictions(
            models[model_name][1]['label'],
            all_pred_proba[model_name],
            name=f"Model {model_name}",
            color=colour,
            linestyle='--' if 'NEWS' in model_name else '-',
            ax=ax[0],
        )

        PrecisionRecallDisplay.from_predictions(
            models[model_name][1]['label'],
            all_pred_proba[model_name],
            name=f"Model {model_name}",
            color=colour,
            linestyle='--' if 'NEWS' in model_name else '-',
            ax=ax[1],
        )

        ax[0].set_title('ROC Curve')
        ax[1].set_title('Precision-Recall Curve')

    # Add random for ROC curve
    x = np.linspace(0, 1)
    y = x
    ax[0].plot(x, y, linestyle='--', color='black', label='Random classifier')
    ax[0].legend()

    savepath = os.path.join(args.save_path, 'auroc_pr.png')
    plt.savefig(savepath, bbox_inches='tight')

    print(f"----- Saved ROC and PR figs to {savepath}")

    # Plot the alert rate curve
    plt.clf()
    fig, ax = plt.subplots(figsize=(8, 8))

    for model_name, colour in zip(models, colours):
        precision, recall, thresholds = precision_recall_curve(models[model_name][1]['label'],
                                                               all_pred_proba[model_name])
        alert_rate = np.array(
            [np.where(all_pred_proba[model_name] > threshold, 1, 0).sum() for threshold in thresholds]
        ) / wholeset_n_days

        if "NEWS" not in model_name:
            sns.lineplot(
                x=recall[np.round(np.linspace(0, len(recall) - 1, 200)).astype(int)],
                y=alert_rate[np.round(np.linspace(0, len(alert_rate) - 1, 200)).astype(int)],
                label=model_name,
                linewidth=2,
                ax=ax,
                color=colour,
                linestyle='--' if 'NEWS' in model_name else '-'
            )
        else:
            sns.lineplot(
                x=recall[:-1],
                y=alert_rate,
                label=model_name,
                linewidth=2,
                ax=ax,
                color=colour,
                linestyle='--' if 'NEWS' in model_name else '-'
            )

    ax.set_title('Alert Rate vs. Sensitivity')
    ax.set_xlabel('Sensitivity')
    ax.set_ylabel('Mean alerts per day')

    savepath = os.path.join(args.save_path, 'alerts.png')
    plt.savefig(savepath, bbox_inches='tight')

    print(f"----- Saved Alert Rate fig to {savepath}")

    plt.clf()
    fig, ax = plt.subplots(figsize=(8, 8))

    for model_name, colour in zip(models, colours):
        precision, recall, thresholds = precision_recall_curve(models[model_name][1]['label'],
                                                               all_pred_proba[model_name])
        nne = 1 / precision

        sns.lineplot(
            x=recall[:-1],
            y=nne[:-1],
            label=model_name,
            linewidth=2,
            ax=ax,
            color=colour,
            linestyle='--' if 'NEWS' in model_name else '-'
        )

    ax.set_title('Numbers Needed to Evaluate vs. Sensitivity')
    ax.set_xlabel('Sensitivity')
    ax.set_ylabel('Numbers Needed to Evaluate')

    savepath = os.path.join(args.save_path, 'nne.png')
    plt.savefig(savepath, bbox_inches='tight')

    print(f"----- Saved NNE fig to {savepath}")

    plt.clf()
    fig, ax = plt.subplots(figsize=(8, 8))

    for model_name, colour in zip(models, colours):
        precision, recall, thresholds = precision_recall_curve(models[model_name][1]['label'],
                                                               all_pred_proba[model_name])
        gee = [generalized_entropy_error(models[model_name][1]['label'],
                                         np.where(all_pred_proba[model_name] > _, 1, 0)) for _ in thresholds]

        sns.lineplot(
            x=recall[:-1],
            y=gee,
            label=model_name,
            linewidth=2,
            ax=ax,
            color=colour,
            linestyle='--' if 'NEWS' in model_name else '-'
        )

    ax.set_title('Generalised Entropy Index vs. Sensitivity')
    ax.set_xlabel('Sensitivity')
    ax.set_ylabel('Generalised Entropy Index')

    savepath = os.path.join(args.save_path, 'gee.png')
    plt.savefig(savepath, bbox_inches='tight')

    print(f"----- Saved NNE fig to {savepath}")


if __name__ == '__main__':
    main()
