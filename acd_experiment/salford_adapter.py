from .base_dataset import BaseDataset
from salford_datasets.salford import SalfordData, SalfordFeatures

import pandas as pd


class SalfordAdapter(SalfordData, BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def omit_redundant(self):
        return SalfordAdapter(
            self.drop(
                SalfordFeatures.Admin
                + SalfordFeatures.Spell
                + SalfordFeatures.Discharge
                + SalfordFeatures.Wards
                + SalfordFeatures.WardLOS
                + SalfordFeatures.Diagnoses
                + SalfordFeatures.Operations
                + SalfordFeatures.VBG
                + ["AE_Arrival", "AE_Departure", "MainDiagnosis", "MainProcedure",],
                axis=1,
                errors="ignore",
            )
        )

    def categorize(self, categories=None):
        return SalfordAdapter(super().convert_str_to_categorical())

    @classmethod
    def feature_group_combinations(cls, scored=False):
        r = {}
        if scored:
            r["news"] = SalfordFeatures.First_NEWS
        else:
            r["news"] = SalfordACDFeatures["NEWS_Raw"]

        r["with_phenotype"] = (
            r["news"]
            + SalfordACDFeatures['NEWS_Extensions']
            + SalfordFeatures.Demographics
        )
        r["with_composites"] = r["with_phenotype"] + SalfordACDFeatures['Composite']
        r["with_labs"] = r["with_composites"] + SalfordFeatures.First_Blood
        r["with_notes"] = r["with_labs"] + SalfordACDFeatures["Notes"]
        r["with_services"] = r["with_notes"] + SalfordACDFeatures["Services"]

        return r

    def fill_na(self):
        return SalfordAdapter(super().fill_na())

    def onehot_encode_categories(self):
        r = self
        for col in self.select_dtypes(include="category").columns:
            r = r._encode_onehot(
                r[[col]], prefix=col.replace("Description", ""), return_df=False
            )
        return r

    def impute_news(self):
        SalfordAdapter(super().single_imputation_obs_and_news())
        return self

    def impute_blood(self):
        SalfordAdapter(super().single_imputation_bloods())
        return self

    def ordinal_encode_categories(self):
        r = self.copy()
        mask = r.select_dtypes(include="category")
        r[mask.columns] = mask.apply(lambda x: x.cat.codes)

        return SalfordAdapter(r)

    def xy(
        self,
        x=[],
        dropna: bool = False,
        fillna: bool = False,
        ordinal_encoding: bool = False,
        onehot_encoding: bool = False,
        imputation: bool = False,
        outcome: str = "CriticalEvent",
        outcome_within: int = 1,
    ):
        X = self
        if imputation:
            X = self.impute_news().impute_blood()

        critical_care = self.derive_critical_care(
            within=outcome_within,
            wards=["CCU", "HH1M"],
            ignore_admit_ward=False,
            return_series=True,
        )
        mortality = self.derive_mortality(within=outcome_within, return_series=True)
        if outcome == "CriticalCare":
            y = critical_care
        elif outcome == "Mortality":
            y = mortality
        else:
            y = critical_care | mortality

        if len(x):
            X = SalfordAdapter(X[x].copy())
        else:
            X = X.omit_redundant()

        if dropna:
            X = SalfordAdapter(X.dropna(how="any"))
            y = y[X.index]

        X = X.categorize()

        if fillna:
            X = X.fill_na()

        if ordinal_encoding:
            X = X.ordinal_encode_categories()
        elif onehot_encoding:
            cat_cols = X.select_dtypes(include="category").columns
            X = X.onehot_encode_categories()
            X = SalfordAdapter(X.drop(cat_cols, axis=1))

        return X, y


SalfordACDFeatures = dict(
    NEWS_Raw=[
        "Obs_RespiratoryRate_Admission",
        "Obs_O2Sats_Admission",
        "Obs_Temperature_Admission",
        "Obs_SystolicBP_Admission",
        "Obs_HeartRate_Admission",
        "Obs_AVCPU_Admission",
    ],
    NEWS_Extensions=[
        "Obs_BreathingDevice_Admission",
        "Obs_DiastolicBP_Admission",
        "Obs_Pain_Admission",
        "Obs_Nausea_Admission",
        "Obs_Vomiting_Admission",
    ],
    Composite=SalfordFeatures.CompositeScores + ["CharlsonIndex"],
    Notes=["AE_PresentingComplaint"],#, "AE_MainDiagnosis"],
    Services=["AdmitMethod", "AdmissionSpecialty", "SentToSDEC", "Readmission"],
)
