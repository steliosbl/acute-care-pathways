from salford_datasets.salford import SalfordData, SalfordFeatures
from acd_experiment.sci import SCIData, SCICols
from acd_experiment.salford_adapter import SalfordAdapter

import numpy as np
import pandas as pd

import argparse
import pickle

from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import cross_validate
from sklearn.metrics import precision_score, recall_score, roc_auc_score, average_precision_score, fbeta_score, \
    make_scorer

import itertools
from lightgbm import LGBMClassifier
from tqdm import tqdm


def get_dataset(data_path, select_features, outcome, old_only=False, old_data_path=None, impute=False,
                text_embeddings=None):
    """ Load SalfordData and convert tabular data to a single text string
    :param data_path: str, path to Salford HDF5 file
    :param select_features: Type of features to use
    :param outcome: Type of outcome to select as CriticalEvent
    :param old_only: if True, use only records present in original data
    :param old_data_path: Path to the original data, must be passed if old_only is True
    :param impute: if True, replace missing values with median
    :param text_embeddings: str, path to pre-computed text embeddings to load. If None, don't load any
    :return SalfordData:
    """

    dataset = SalfordData.from_raw(pd.read_hdf(data_path, key='table'))
    dataset = SalfordData(dataset[dataset['Age'] >= 18]).augment_derive_all()

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
    elif select_features == 'new_triagenotes':
        columns = scores_with_notes_labs_and_hospital + new_features
    elif select_features == 'new_diag':
        columns = scores_with_notes_labs_and_hospital + new_features + SalfordFeatures.Diagnoses
        dataset = dataset.expand_icd10_definitions()
    elif select_features == 'none':
        # Only include text embeddings
        columns = []
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

    if impute:
        number_columns = dataset.select_dtypes(include=[float, int]).columns

        imp = SimpleImputer(strategy='median', missing_values=np.nan)
        dataset[number_columns] = imp.fit_transform(dataset[number_columns])

    columns_to_keep = columns + [outcome_col]
    dataset = SalfordData(dataset[columns_to_keep]).convert_str_to_categorical(inplace=False)

    if text_embeddings:
        dataset = SalfordData(dataset).load_text_embeddings(text_embeddings)

    Y = dataset[outcome_col]
    del dataset[outcome_col]
    X = dataset

    return X, Y


def main():
    """
        Train LGBMs with pre computed text embeddings
    """
    parser = argparse.ArgumentParser(description='Train LGBMs with pre computed text embeddings')

    parser.add_argument('--data-path', type=str, help='Path to the raw dataset HDF5 file', required=True)
    parser.add_argument("--old-data-path", type=str, help="Path to the old dataset HDF5")
    parser.add_argument("--select-features", help="Limit feature groups",
                        choices=['all', 'sci', 'sci_no_adm', 'new', 'new_no_adm', 'new_triagenotes', 'new_diag',
                                 'sci_diag', 'none'], default='all')
    parser.add_argument("--outcome", help="Outcome to predict", choices=['strict', 'h1', 'direct', 'sci'],
                        default='sci')
    parser.add_argument("--old-only", help="Use only patients in the original dataset", action="store_true")
    parser.add_argument("--impute", action="store_true", help="Impute missing data")
    parser.add_argument("--text-embeddings", type=str, default=None, help="Path to text embeddings. If None, don't use")

    parser.add_argument('--save-model', type=str, default='./', help='Directory to save model to')

    args = parser.parse_args()

    X, Y = get_dataset(args.data_path, args.select_features, args.outcome, args.old_only, args.old_data_path,
                       args.impute, args.text_embeddings)

    calibration_parameters = dict(
        ensemble=True,
        cv=3,
        method='isotonic',
        n_jobs=3
    )

    cross_validation_metrics = dict(
        Precision='precision',
        Recall='recall',
        AUC='roc_auc',
        AP='average_precision',
        F2=make_scorer(fbeta_score, beta=2),
        Spec=make_scorer(recall_score, pos_label=0)
    )

    lightgbm_parameters = dict(
        objective='binary',
        random_state=42,
        metrics=['l2', 'auc'],
        boosting_type='gbdt',
        n_jobs=-1,
        is_unbalance=True
    )

    results = cross_validate(CalibratedClassifierCV(
        LGBMClassifier(**lightgbm_parameters), **calibration_parameters
    ), X, Y, cv=5, n_jobs=1, scoring=cross_validation_metrics, return_estimator=True)

    print(results)

    if args.save_model:
        with open(args.save_model, 'wb') as f:
            pickle.dump(results['estimator'][-2], f)

        print('----- Saved model to', args.save_model)


if __name__ == '__main__':
    main()
