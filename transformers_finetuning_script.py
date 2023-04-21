# %%
import os, pickle, warnings, dataclasses, itertools, argparse
from pathlib import Path
from functools import partial 
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats as st

import torch
from torch import nn
from torch.nn import functional as F

from tqdm import tqdm
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

from typing import Iterable, Tuple

import logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

import transformers
transformers.logging.set_verbosity_error()

from salford_datasets.salford import SalfordData, SalfordFeatures, SalfordPrettyPrint, SalfordCombinations
from salford_datasets.utils import DotDict

# %%
class Notebook:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DATA_DIR = Path('data/Salford/')
    CACHE_DIR = Path('models/')
    RE_DERIVE = False

# %%
BERTModels = DotDict(
    BioClinicalBert="emilyalsentzer/Bio_ClinicalBERT",
    Bert="distilbert-base-uncased",
    PubMedBert="ml4pubmed/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext_pub_section"
)

# %%
from transformers import AutoTokenizer

@dataclasses.dataclass
class SalfordTransformerDataset(torch.utils.data.Dataset):
    _text: Iterable[str]
    _labels: Iterable[str]
    _avail_idx: Iterable[bool]
    _text_tz: Iterable[str] = None

    @classmethod
    def from_SalfordData(cls, sal, model_uri, columns=SalfordCombinations.with_services):
        _avail_idx = sal[columns].notna().any(axis=1)
        _text = SalfordData(sal).tabular_to_text(columns)
        _labels = sal.CriticalEvent.copy().astype(int).values

        return cls(_text, _labels, _avail_idx).tokenise(model_uri)

    def tokenise(self, model_uri):
        tz =  AutoTokenizer.from_pretrained(model_uri)
        tz_kwargs = dict(truncation=True, padding=True, max_length=512)

        self._text_tz = tz(list(self._text), **tz_kwargs)
        return self
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return SalfordTransformerDataset(
                _text = self._text[idx],
                _labels = self._labels[idx],
                _avail_idx = self._avail_idx.iloc[idx],
                _text_tz = dict(
                    input_ids = self._text_tz['input_ids'][idx],
                    attention_mask = self._text_tz['attention_mask'][idx]
                )
            )
            
        return dict(
            input_ids = self._text_tz['input_ids'][idx],
            attention_mask = self._text_tz['attention_mask'][idx],
            labels = self._labels[idx]
        )

    def __len__(self):
        return len(self._text)

    @property
    def tensors(self):
        return dict(
            input_ids = torch.tensor(self._text_tz['input_ids']),
            attention_mask = torch.tensor(self._text_tz['attention_mask'])
        )

# %% [markdown]
# ## 4. Fine-Tuned Transformer
# 
#  - 4.1. Clinical notes on their own
#  - 4.2. `with_services` on its own (text-ified)
#  - 4.3. Expanded diagnoses on their own
#  - 4.4. `with_services` and clinical notes
#  - 4.5. All together

# %%
EXPERIMENT_FEATURE_SETS = {
    '41': SalfordFeatures.Text[:-2],
    '42': SalfordCombinations.with_services,
    '43': SalfordFeatures.Diagnoses,
    '44': SalfordCombinations.with_services + SalfordFeatures.Text[:-2],
    '45': SalfordCombinations.with_services + SalfordFeatures.Text[:-2] + SalfordFeatures.Diagnoses
}

# %%
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformer_experiment.utils.finetuning import bert_finetuning_metrics
from transformer_experiment.utils.finetuning import split_dict_into_batches, load_dict_to_device

def finetune_note_transformer(sal_tz, model_uri, save_directory="bert-finetuned-notes_fake_delete", batch_size=56):
    bert_args = TrainingArguments(
        Notebook.CACHE_DIR/save_directory,
        evaluation_strategy = "epoch",
        save_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=10,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model='AP',
        report_to='none',
        optim="adamw_torch",
        disable_tqdm=False
    )

    bert_kwargs = dict(
        num_labels=2, output_attentions=False, output_hidden_states=False, ignore_mismatched_sizes=True
    )

    X_train, X_val = train_test_split(sal_tz, test_size=0.15, random_state=123, stratify=sal_tz._labels)

    model = AutoModelForSequenceClassification.from_pretrained(model_uri, **bert_kwargs)

    trainer = Trainer(
        model,
        bert_args,
        train_dataset=X_train,
        eval_dataset=X_val,
        compute_metrics=bert_finetuning_metrics
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer.train()

    trainer.save_model(Notebook.CACHE_DIR/save_directory/'best_model')

    return model.eval()

def finetuned_inference(model, dataset, batch_size):
    with torch.no_grad():
        X = split_dict_into_batches(dataset.tensors, batch_size)
        y_pred_logit = torch.concat([
            model(**load_dict_to_device(x)).logits for x in tqdm(X)
        ])

        y_pred_proba = F.softmax(y_pred_logit, dim=1)[:,1]
        
    return y_pred_proba.cpu().detach().numpy()

# %%
from sklearn.model_selection import train_test_split

def load_sal(re_derive=False):
    if re_derive:
        logging.info('Deriving from raw dataset')
        sal = SalfordData.from_raw(
            pd.read_hdf(Notebook.DATA_DIR/'raw_v2.h5', 'table')
        ).augment_derive_all().expand_icd10_definitions().sort_values('AdmissionDate')
        sal.to_hdf(Notebook.DATA_DIR/'sal_processed_transformers.h5', 'table')
    else:
        logging.info('Loading processed dataset')
        sal = SalfordData(pd.read_hdf(Notebook.DATA_DIR/'sal_processed_transformers.h5', 'table'))

    return sal

def tokenise_dataset(model_uri, feature_set, re_derive=False, debug=False):
    sal = load_sal(re_derive)
    if debug:
        sal = sal.sample(100)
        sal.loc[sal.sample(20).index, 'CriticalEvent'] = True
    
    sal_train, sal_test = train_test_split(sal, test_size=0.33, shuffle=False)

    logging.info('Tokenising feature set')
    sal_bert_train = SalfordTransformerDataset.from_SalfordData(sal_train, model_uri, feature_set)
    sal_bert_test = SalfordTransformerDataset.from_SalfordData(sal_test, model_uri, feature_set)

    return sal_bert_train, sal_bert_test

def load_tokenised_dataset_cached(bert_variant, experiment_num, feature_set):
    cache_filepath = Notebook.CACHE_DIR/f'sal_bert_{bert_variant}_{experiment_num}.bin'
    if os.path.isfile(cache_filepath):
        logging.info('Loading tokenised data from cache')
        with open(cache_filepath, 'rb') as file:
            sal_bert_train, sal_bert_test = pickle.load(file)
    else:
        sal_bert_train, sal_bert_test = tokenise_dataset(BERTModels[bert_variant], feature_set)
        with open(cache_filepath, 'wb') as file:
            pickle.dump((sal_bert_train, sal_bert_test), file)
    
    return sal_bert_train, sal_bert_test

# %%
from transformers import AutoModelForSequenceClassification
import shutil

def get_checkpoint_directory(experiment_num='41', bert_variant='BioClinicalBert'):
    model_directory = f'bert_{bert_variant}_{experiment_num}' 
    checkpoint_dir = [_ for _ in os.listdir(Notebook.CACHE_DIR/model_directory) if 'checkpoint-' in _]
    checkpoint_dir = sorted(checkpoint_dir, key=lambda _: int(_.split('-')[1]))
    checkpoint_dir = Notebook.CACHE_DIR/model_directory/(checkpoint_dir[-1])

    return model_directory, checkpoint_dir

def run_finetuning_4(experiment_num='41', bert_variant='BioClinicalBert', batch_size=56, debug=False):
    feature_set = EXPERIMENT_FEATURE_SETS[experiment_num]
    model_uri = BERTModels[bert_variant]

    if debug:
        model_directory = "bert-finetuned-notes_fake_delete"
        sal_bert_train, sal_bert_test = tokenise_dataset(model_uri, feature_set, debug=True)
    else:
        model_directory = f'bert_{bert_variant}_{experiment_num}' 
        sal_bert_train, sal_bert_test = load_tokenised_dataset_cached(bert_variant, experiment_num, feature_set)

    model = finetune_note_transformer(
        sal_bert_train, model_uri, model_directory, batch_size
    )

    checkpoint_dir = [Notebook.CACHE_DIR/model_directory/_ for _ in os.listdir(Notebook.CACHE_DIR/model_directory) if 'checkpoint-' in _]
    for directory in checkpoint_dir:
        shutil.rmtree(directory)

    y_pred_proba = finetuned_inference(model, sal_bert_test, batch_size)
    
    with open(Notebook.CACHE_DIR/model_directory/'test_pred_proba.bin', 'wb') as file:
        pickle.dump(y_pred_proba, file)

from transformer_experiment.utils.shallow_classifiers import load_salford_dataset, get_train_test_indexes

def run_inference_4(experiment_num='41', bert_variant='BioClinicalBert', batch_size=56):
    feature_set = EXPERIMENT_FEATURE_SETS[experiment_num]
    model_uri = BERTModels[bert_variant]
    
    #model_directory, checkpoint_dir = get_checkpoint_directory(experiment_num, bert_variant)
    model_directory = f'bert_{bert_variant}_{experiment_num}' 

    _, sal_bert_test = load_tokenised_dataset_cached(bert_variant, experiment_num, feature_set)

    model = AutoModelForSequenceClassification.from_pretrained(Notebook.CACHE_DIR/model_directory/'best_model').to(Notebook.DEVICE).eval()

    _, sal_test_idx, _, _ = get_train_test_indexes(load_salford_dataset(Notebook.RE_DERIVE, Notebook.DATA_DIR))
    idx_mask = sal_bert_test._avail_idx & sal_bert_test._avail_idx.index.isin(sal_test_idx)

    y_pred_proba = finetuned_inference(model, sal_bert_test, batch_size)
    y_pred_proba = pd.Series(y_pred_proba[idx_mask], index=idx_mask[idx_mask].index)

    with open(Notebook.CACHE_DIR/model_directory/'test_pred_proba_indexed.bin', 'wb') as file:
        pickle.dump(y_pred_proba, file)


# %%
from transformer_experiment.utils.finetuning import construct_parser

if __name__ == '__main__':
    parser = construct_parser()
    args = parser.parse_args()

    #run_finetuning_4(args.experiment, args.model, args.batch_size, args.debug)
    run_inference_4(args.experiment, args.model, args.batch_size)


