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
        return SalfordAdapter(super().categorize(categories))

    @property
    def feature_group_combinations(self):
        r = dict(news_scores=SalfordACDFeatures["NEWS"])
        r["news_scores_with_phenotype"] = (
            r["news_scores"] + SalfordACDFeatures["Phenotype"]
        )
        r["scores_with_labs"] = (
            r["news_scores_with_phenotype"] + SalfordACDFeatures["Labs"]
        )
        r["scores_with_notes_and_labs"] = (
            r["scores_with_labs"] + SalfordACDFeatures["Notes"]
        )
        r["scores_with_notes_labs_and_hospital"] = (
            r["scores_with_notes_and_labs"] + SalfordACDFeatures["Hospital"]
        )
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
        static_fills = {col: 0 for col in SalfordACDFeatures["NEWS"]}

        for col, val in static_fills.items():
            if col in self.columns:
                self[col] = self[col].fillna(val)

        return self

    def impute_blood(self):
        for _ in SalfordACDFeatures["Labs"]:
            ser = pd.to_numeric(self[_], errors="coerce")
            self[_] = ser.fillna(ser.median())

        return self

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

        critical_care = self.derive_critical_care(within=outcome_within, wards=['CCU', 'HH1M'], ignore_admit_ward=False, return_series=True)
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
            raise NotImplementedError()
        elif onehot_encoding:
            cat_cols = X.select_dtypes(include="category").columns
            X = X.onehot_encode_categories()
            X = SalfordAdapter(X.drop(cat_cols, axis=1))

        return X, y


SalfordACDFeatures = dict(
    NEWS=[
        "NEWS_RespiratoryRate_Admission",
        "NEWS_O2Sat_Admission",
        "NEWS_Temperature_Admission",
        "NEWS_BP_Admission",
        "NEWS_HeartRate_Admission",
        "NEWS_AVCPU_Admission",
        "NEWS_BreathingDevice_Admission",
    ],
    Phenotype=["Female", "Age"],
    Notes=["AE_PresentingComplaint", "AE_MainDiagnosis"],
    Labs=[
        "Blood_Haemoglobin_Admission",
        "Blood_Urea_Admission",
        "Blood_Sodium_Admission",
        "Blood_Potassium_Admission",
        "Blood_Creatinine_Admission",
    ],
    Hospital=["AdmitMethod", "AdmissionSpecialty", "SentToSDEC", "Readmission"],
)
