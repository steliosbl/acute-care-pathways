from salford_datasets.salford import SalfordData, SalfordFeatures
from acd_experiment.sci import SCIData, SCICols
from acd_experiment.salford_adapter import SalfordAdapter
from .whole_nlp import get_text_dataset

import argparse
import pickle
import torch
import numpy as np
import pandas as pd
from tqdm import trange

import matplotlib.pyplot as plt
import seaborn as sns

from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
import evaluate

from captum.attr import LayerIntegratedGradients, TokenReferenceBase


def get_column_attributions(dataset, tokenizer, lig, columns, sample_idx, absolute_values=False):
    """
    Get columnar attributions using Layer Integrated Gradients for a single sample
    :param dataset: HF dataset
    :param tokenizer: HF tokenizer
    :param lig: captum.attr.LayerIntegratedGradients instance
    :param columns: List[str] of column names
    :param sample_idx: int, sample index to get atrributions for
    :param absolute_values: bool, if True the return absolute attributions
    :return: dict of {column_name: attribution_value} pairs
    """
    sample_text = dataset['text'][sample_idx]
    sample_label = dataset['label'][sample_idx]
    sample_ids = tokenizer.encode(sample_text, return_tensors='pt').cuda()

    # Baseline is a string of padding tokens the same length as the input
    baseline = torch.tensor([tokenizer.cls_token_id] + [tokenizer.pad_token_id] * (len(sample_ids[0]) - 2) + [
        tokenizer.sep_token_id]).cuda()
    baseline = baseline.unsqueeze(dim=0)

    attributions = lig.attribute(inputs=sample_ids,
                                 baselines=baseline,
                                 target=sample_label,
                                 n_steps=50
                                 )

    attributions = attributions.sum(dim=-1)
    attributions = attributions / torch.norm(attributions)

    if absolute_values:
        attributions = torch.abs(attributions)

    # ID 16 is , - this can be used to split data into columns
    comma_token = tokenizer(',')['input_ids'][1]
    ids_split = torch.tensor_split(sample_ids, [i for i, x in enumerate(sample_ids[0]) if x == comma_token], dim=1)
    attr_split = torch.tensor_split(attributions, [i for i, x in enumerate(sample_ids[0]) if x == comma_token], dim=1)

    # Last columns are all the structured ones
    column_attributions = {c: 0 for c in columns}
    if len(columns) <= len(ids_split):
        for i, c in enumerate(column_attributions):
            column_attributions[c] = attr_split[i].sum()

    return column_attributions


def main():
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

    parser.add_argument("--abs", action="store_true", help="Calculate absolute attributions")

    parser.add_argument('--model-name', type=str,
                        default='ml4pubmed/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext_pub_section',
                        help='Huggingface model tag, or path to a local model')
    parser.add_argument('--save-path', type=str, default='./', help='Directory to save results to')

    args = parser.parse_args()

    # Get the correct set of column names
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

    if args.select_features == 'sci':
        columns = scores_with_notes_labs_and_hospital
    elif args.select_features == 'sci_no_adm':
        columns = scores_with_notes_labs_and_hospital[:-1]
    elif args.select_features == 'new':
        columns = scores_with_notes_labs_and_hospital + new_features
    elif args.select_features == 'new_no_adm':
        columns = scores_with_notes_labs_and_hospital[:-1] + new_features
    elif args.select_features == 'new_triagenotes':
        columns = scores_with_notes_labs_and_hospital + new_features
    elif args.select_features == 'sci_triagenotes':
        columns = scores_with_notes_labs_and_hospital
    elif args.elect_features == 'new_diag':
        columns = scores_with_notes_labs_and_hospital + new_features + SalfordFeatures.Diagnoses
    else:
        columns = None

    dataset = get_text_dataset(args.data_path, args.select_features, args.outcome, args.old_only, args.old_data_path,
                               args.column_name, split='val')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, device=0)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, output_hidden_states=False,
                                                               num_labels=2, ignore_mismatched_sizes=True)
    model = model.to('cuda:0')

    encoded_dataset = Dataset.from_pandas(dataset)
    encoded_dataset = encoded_dataset.class_encode_column('label')
    #encoded_dataset = encoded_dataset.train_test_split(test_size=0.2, shuffle=True, stratify_by_column='label')

    def forward_func(x):
        # Captum expects just the log probabilities
        return model(x, attention_mask=torch.ones_like(x))[0]

    lig = LayerIntegratedGradients(forward_func, model.bert.embeddings)

    if '_triagenote' in args.select_features:
        columns = ['Triage Note'] +  columns

    # label, attributions
    all_column_attributions = []
    for sample_idx in trange(len(encoded_dataset)):
        sample_column_attributions = get_column_attributions(encoded_dataset,
                                                             tokenizer,
                                                             lig,
                                                             columns,
                                                             sample_idx,
                                                             args.abs
                                                             )

        all_column_attributions.append([encoded_dataset['label'][sample_idx], sample_column_attributions])

    with open(args.save_path, 'wb') as f:
        pickle.dump(all_column_attributions, f)

    print(f"----- Saved attributions to {args.save_path}")


if __name__ == '__main__':
    main()
