import pandas as pd
import numpy as np
import json

from .salford_raw import (
    SalfordTimeseries,
    RedundantColumns,
    RawTimeseries,
    AEDiagnosisStems,
    AEVaguePresentingComplaints,
)
from .icd10 import ICD10Table
from .ccs import CCSTable
from .utils import DotDict, row_to_text


class SalfordData(pd.DataFrame):
    """ Represents the Salford dataset and related methods to augment or filter it """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, name):
        """ Sets up direct access of feature groups, e.g., SalfordData(df).Wards """
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name in SalfordFeatures:
                return self[SalfordFeatures[name]]
            raise

    def _finalize_derived_feature(self, series, col_name, return_series):
        """ Given a derived feature in pd.Series form,
        concatenates it to the dataset under $col_name$ or reindexes to match the dataset """

        if return_series:
            return series.reindex_like(self).rename(col_name)

        r = self.copy()
        r[col_name] = series
        return SalfordData(r)

    def _finalize_derived_feature_wide(self, df, col_names, return_df):
        """ Given a derived feature in wide (pd.DataFrame) form,
        concatenates it to the dataset under $col_names$ or reindexes to match the dataset
        :param drop_cols: If not None, drop these cols from the returned SalfordData object
        raises ValueError if drop_cols is passed and return_df=True, as this makes no sense
        """
        if return_df:
            return df.reindex(index=self.index, columns=col_names)

        r = self.copy()
        r[col_names] = df

        return SalfordData(r)

    def _encode_onehot(self, df, prefix, prefix_sep="__", fillna=False, return_df=True):
        """ Given a set of columns, one-hot encodes them
        :param df: The dataframe to encode
        :param prefix: What to call the new columns
        """
        encoded = (
            pd.get_dummies(df.stack(), prefix=prefix, prefix_sep=prefix_sep)
            .groupby(level=0)
            .any()
        )

        return self._finalize_derived_feature_wide(
            encoded, encoded.columns, return_df=return_df
        )

    def derive_mortality(self, within=1, col_name="Mortality", return_series=False):
        """ Determines the patients' mortality outcome.
        :param within: Time since admission to consider a death. E.g., 1.0 means died within 24 hours, otherwise lived past 24 hours
        :returns: New SalfordData instance with the new feature added under $col_name
            if return_series: pd.Series instance of the new feature
            if return_threshold: pd.Series instance with the times the outcome occurred, if applicable
        """
        series = self.DiedDuringStay
        if within is not None:
            series = series & (self.TotalLOS <= within)

        return self._finalize_derived_feature(series, col_name, return_series)

    def derive_critical_care(
        self,
        within=1,
        wards=["CCU"],
        col_name="CriticalCare",
        ignore_admit_ward=True,
        return_series=False,
    ):
        """ Determines admission to critical care during the spell as indicated by admission to specified wards
        :param within: Threshold of maximum LOS to consider events for. Critical care admission that occurs after this value won't be counted.
        :param wards: The wards to search for. By default, ['CCU']
        :param ignore_admit_ward: Ignore patients admitted directly to critical care
        :returns: New SalfordData instance with the new features added
            if return_series: pd.Series instance of the new feature
        """
        wards_to_check = SalfordFeatures.Wards
        if ignore_admit_ward:
            wards_to_check = list(set(wards_to_check) - {"AdmitWard"})
        m = self[wards_to_check].isin(wards)
        column_where_critical_appeared = m.idxmax(axis=1).where(m.any(1)).dropna()
        stacked_los = self.WardLOS.stack()

        ward_los_correspondence = dict(
            zip(SalfordFeatures.Wards, SalfordFeatures.WardLOS)
        )

        cumsum_entry_where_critical_appeared = zip(
            column_where_critical_appeared.index,
            map(ward_los_correspondence.get, column_where_critical_appeared.values),
        )

        los_on_critical_admission = (
            stacked_los.groupby(level=0).cumsum() - stacked_los
        ).loc[cumsum_entry_where_critical_appeared]

        los_on_critical_admission.index = los_on_critical_admission.index.droplevel(1)
        series = (
            (los_on_critical_admission <= (within or 9999))
            .reindex_like(self)
            .fillna(False)
        )

        return self._finalize_derived_feature(series, col_name, return_series)

    def derive_critical_event(
        self,
        within=1,
        col_name="CriticalEvent",
        return_series=False,
        wards=["CCU"],
        ignore_admit_ward=True,
    ):
        """ Determines critical event occurrence, i.e., a composite of critical care admission or mortality
        :param within: Threshold of maximum LOS to consider events for. Critical care admission or mortality that occurs after this value won't be counted.
        :param wards: The wards to search for to obtain critical care admission. By default, ['CCU']
        :param ignore_admit_ward: Ignore patients admitted directly to critical care
        :returns: New SalfordData instance with the new features added
            if return_series: pd.Series instance of the new feature
        """

        mortality = self.derive_mortality(within=within, return_series=True)
        critical = self.derive_critical_care(
            within=within,
            wards=wards,
            ignore_admit_ward=ignore_admit_ward,
            return_series=True,
        )

        series = mortality | critical

        return self._finalize_derived_feature(series, col_name, return_series)

    def _readmission_time_spans(self):
        return (
            self.sort_values(["PatientNumber", "AdmissionDate"])
            .groupby("PatientNumber")
            .AdmissionDate.diff()
            .dropna()
        )

    def derive_readmission(
        self, within=30, col_name="Readmission", return_series=False
    ):
        """ Determines whether each record is a readmission,
        i.e. the same patient appeared in a previous record (chronologically) within the specified time window
        :param within: Maximum time between last discharge and latest admission to consider a readmission
        :returns: New SalfordData instance with the new features added
            if return_series: pd.Series instance of the new feature
        """
        series = self._readmission_time_spans() < pd.Timedelta(days=within)

        return self._finalize_derived_feature(series, col_name, return_series).fillna(
            False
        )

    def derive_readmission_band(
        self,
        bins=[0, 1, 2, 7, 14, 30, 60],
        labels=["24 Hrs", "48 Hrs", "1 Week", "2 Weeks", "1 Month", "2 Months"],
        col_name="ReadmissionBand",
        return_series=False,
    ):
        """ Bins all readmissions (re-appearances of patients) depending on the elapsed time between the last discharge and latest admission
        :param within: Maximum time between last discharge and latest admission to consider a readmission
        :returns: New SalfordData instance with the new features added
            if return_series: pd.Series instance of the new feature
        """
        bins = [pd.Timedelta(days=_) for _ in [-1] + bins]
        labels = labels + ["N/A"]

        series = pd.cut(
            self._readmission_time_spans(), bins, labels=labels, ordered=True
        )

        return self._finalize_derived_feature(series, col_name, return_series).fillna(
            "N/A"
        )

    def derive_sdec(
        self, sdec_wards=["AEC", "AAA"], col_name="SentToSDEC", return_series=False
    ):
        """ Determines whether the patient originally was admitted to SDEC
        :param sdec_wards: The wards to search for. By default, ['AEC', 'AAA']
        :returns: New SalfordData instance with the new features added
            if return_series: pd.Series instance of the new feature
        """
        series = self.AdmitWard.isin(sdec_wards)

        return self._finalize_derived_feature(series, col_name, return_series)

    def derive_ae_diagnosis_stems(
        self, stems=AEDiagnosisStems, col_name="AE_MainDiagnosis", return_series=False
    ):
        """ Derive stems from AE Diagnosis column
        :param stems: List of stems to derive
        :returns: New SalfordData instance with the new features in place of the old column
            if return_series: pd.Series instance of the new feature
        """
        # TODO: This actually doesn't catch multiple stems in the same entry
        stems = self.AE_MainDiagnosis.str.lower().str.extract(
            f'({"|".join(stems)})', expand=False
        )

        return self._finalize_derived_feature(stems, col_name, return_series)

    def derive_charlson_index(
        self, col_name="CharlsonIndex", return_series=False, comorbidities_lookup=None
    ):
        """ Derives the Charlson Comorbidity Index score based on comorbidity severity
        :param comorbidities_lookup: Lookup table (dict) identifying ICD-10 3-codes as relevant comorbidities
        :returns: New SalfordData instance with the new features added
            if return_series: pd.Series instance of the new feature
        """
        # Scores from: https://www.mdcalc.com/calc/3917/charlson-comorbidity-index-cci
        charlson_scores = {
            "myocardial_infarction": 1,
            "congestive_heart_failure": 1,
            "periphral_vascular_disease": 1,
            "cerebrovascular_disease": 1,
            "dementia": 1,
            "chronic_pulmonary_disease": 1,
            "connective_tissue_disease_rheumatic_disease": 1,
            "peptic_ulcer_disease": 1,
            "mild_liver_disease": 1,
            "diabetes_wo_complications": 1,
            "diabetes_w_complications": 2,
            "paraplegia_and_hemiplegia": 2,
            "renal_disease": 2,
            "cancer": 2,
            "moderate_or_sever_liver_disease": 3,
            "metastitic_carcinoma": 6,
            "aids_hiv": 6,
        }

        if comorbidities_lookup is None:
            comorbidities_lookup = json.load(
                open("data/icd10/charlson_elixhauser10.json")
            )

        # Get the comorbidity component of the Index first
        series = (
            (
                # Convert the ICD-10 coded diagnoses to 3-codes
                self.derive_icd10_3codes(return_df=True)
                # Identify the relevant comorbidities (set the rest to NaN)
                .applymap(comorbidities_lookup.get)
                # Substitute the comorbidities with their scores
                .applymap(charlson_scores.get)
            )
            .sum(axis=1)
            .fillna(0)
        )

        # Get the age component
        series[(50 < self.Age) & (self.Age <= 59)] += 1
        series[(60 < self.Age) & (self.Age <= 69)] += 2
        series[(70 < self.Age) & (self.Age <= 79)] += 3
        series[self.Age > 79] += 4

        return self._finalize_derived_feature(series, col_name, return_series)

    def clean_icd10(self, return_df=False):
        """ Standardises the ICD-10 diagnosis entries to match the lookup table
        :returns: New SalfordData instance with the entries altered
            if return_df: pd.DataFrame instance with the cleaned entries
        """
        df = ICD10Table.fuzzy_match(self.Diagnoses)

        return self._finalize_derived_feature_wide(
            df, SalfordFeatures.Diagnoses, return_df
        )

    def derive_icd10_3codes(self, return_df=False, clean_icd10=True):
        """ Converts the ICD-10 diagnosis entries to 3-codes (e.g., "X10.31" -> "X10")
        :param clean_icd10: Whether to standardise the ICD-10 codes first
        :returns: New SalfordData instance with the entries altered
            if return_df: pd.DataFrame instance with the cleaned entries
        """

        df = self.Diagnoses.stack().str.split(".").str[0].unstack()

        return self._finalize_derived_feature_wide(
            df, SalfordFeatures.Diagnoses, return_df
        )

    def derive_ccs(self, return_df=False, clean_icd10=True, grouping="HSMR"):
        """ Computes the CCS codes for the patients' ICD-10 diagnoses
        :param clean_icd10: Whether to standardise the ICD-10 codes first. REQUIRED if it has not already been done.
        :param grouping: CCS sub-grouping scheme to use. Must be one of ['SHMI', 'HSMR', None].
        :returns: New SalfordData instance with the CCS in place of the ICD-10
            if return_df: pd.DataFrame instance with the cleaned entries
        :raises ValueError: If the $grouping value is invalid.
        """
        if grouping not in ["SHMI", "HSMR", None]:
            raise ValueError('The `grouping` must be one of "SHMI", "HSMR", or None.')

        icd10 = self.clean_icd10(return_df=True) if clean_icd10 else self.Diagnoses
        df = CCSTable.fuzzy_match(icd10)

        if grouping == "HSMR":
            df = CCSTable.convert_hsmr(df)
        elif grouping == "SHMI":
            df = CCSTable.convert_shmi(df)

        return self._finalize_derived_feature_wide(
            df, SalfordFeatures.Diagnoses, return_df
        )

    def augment_derive_all(self, within=1):
        return SalfordData(
            pd.concat(
                [
                    self.drop("AE_MainDiagnosis", axis=1),
                    self.derive_mortality(within=within, return_series=True),
                    self.derive_critical_care(within=within, return_series=True),
                    self.derive_critical_event(within=within, return_series=True),
                    self.derive_readmission(return_series=True),
                    # self.derive_readmission_band(return_series=True),
                    self.derive_sdec(return_series=True),
                    self.derive_ae_diagnosis_stems(return_series=True),
                    self.derive_charlson_index(return_series=True),
                ],
                axis=1,
            )
        )

    def to_text(self, columns=None, target_col='CriticalEvent', inplace=True):
        """ Convert the given columns to a single text string for passing to an NLP model
        :columns: List of column names to be converted to a string. If None, convert all but the target column
        :target_col: str of the column to be used as the label
        :return: SalfordData instance with one text column and one label column
            if inplace: Reference to the existing, altered SalfordData instance
        """
        df = self if inplace else self.copy()

        if not columns:
            columns = list(df.columns)

        columns.append(target_col)

        df = df[columns]
        Y = df[target_col]
        df = df.drop(target_col, axis=1)

        # Drop patientnumber if still in df at this stage (though it shouldn't be, this is just a fail safe)
        df = df.drop('PatientNumber', axis=1, errors='ignore')

        df['text'] = df.apply(row_to_text, axis=1)
        df = df.drop(columns, axis=1, errors='ignore')

        # Use HuggingFace standard here, just for ease of use later on
        df['label'] = Y

        return df

    def convert_str_to_categorical(self, columns=None, inplace=True):
        """ Transforms all string columns into categorical types 
        :columns: List of columns to convert. If empty, selects all string/object type columns
        :returns SalfordData instance with the columns altered 
            if inplace: Reference to the existing, altered SalfordData instance
        """
        df = self if inplace else self.copy()

        selection = df[columns] if columns else df.select_dtypes("object")
        df[selection.columns] = selection.astype("category")

        return df

    @classmethod
    def from_raw(cls, raw):
        """ Applies initial pre-processing steps to the raw dataset, extracted from xlsx """
        # Get rid of various unnecessary columns, such as timeseries collection dates
        df = raw.drop(RedundantColumns, axis=1)

        # Identify these placeholder values as missing in their respective columns
        make_nan = {
            "Gender": "Other",
            "Ethnicity": "NOT STATED",
            "AE_MainDiagnosis": "UNKNOWN",
            "AE_Location": "System Generated",
        }
        for col_name, nan_value in make_nan.items():
            df[col_name] = df[col_name].replace(nan_value, np.nan)

        # Reindex by spellserial, as numbering from excel is inconsistent
        df = df.set_index("SpellSerial")

        # Remove patients who have not been discharged yet
        df = df[df.LOSBand != "Still In"]

        # Filter patients without an ID or age, at minimum, and make these columns integers
        df = df[~(df.PatientNumber.isna() | df.Age.isna())].copy()
        df.Age = df.Age.astype(int)
        df.PatientNumber = df.PatientNumber.astype(int)

        # Convert the following columns to bool
        binarize = [
            ("Gender", ["Female"]),
            ("DiedDuringStay", ["Yes"]),
            ("DiedWithin30Days", ["Yes"]),
            ("AdmissionType", ["Elective"]),
            ("CareHome", ["Yes"]),
            ("Resus_Status", ["DNA - CPR", "Unified DNA - CPR"]),
        ]

        for col_name, true_values in binarize:
            df[col_name] = (
                df[col_name]
                .apply(true_values.__contains__)
                .apply(lambda x: np.nan if x == NotImplemented else x)
            )

        # Rename some columns, including all time series
        renaming = {
            "Gender": "Female",
            "AdmissionType": "ElectiveAdmission",
            "Resus_Status": "HasDNAR",
            "AdmitWardLoS": "AdmitWardLOS",
            "CFS_score": "CFS_Score",
        }

        # Naming scheme to use for time series
        timeseries_labelling = ["Admission", "24HPostAdm", "24HPreDisch", "Discharge"]
        df.rename(
            columns=renaming
            | {
                col: f"{parent}_{timeseries_labelling[i]}"
                for parent, cols in RawTimeseries.items()
                for i, col in enumerate(cols)
            },
            inplace=True,
        )

        # Simplify A&E patient groups by combining all trauma admissions & making values lowercase
        AE_Patient_Group_Aggregation = {
            "FALLS": "TRAUMA",
            "ACCIDENT": "TRAUMA",
            "SELF HARM": "TRAUMA",
            "ROAD TRAFFIC ACCIDENT": "TRAUMA",
            "ASSAULT": "TRAUMA",
            "SPORTS INJURY": "TRAUMA",
            "KNIFE INJURIES INFLICTED": "TRAUMA",
            "FIREWORK INJURY": "TRAUMA",
        }
        df["AE_PatientGroup"] = df.AE_PatientGroup.replace(
            AE_Patient_Group_Aggregation
        ).str.lower()

        # Remove vague values from the A&E presenting complaint and diagnosis
        for col in ["AE_PresentingComplaint", "AE_MainDiagnosis"]:
            df[col] = df[col].str.lower().str.strip(" .?+")
            df.loc[df[col].isin(AEVaguePresentingComplaints), col] = np.nan

        # Cut out all but the top-50 most frequent presenting complaints.
        # These make up about 97% of available values, which is enough
        mask = (
            ~df.AE_PresentingComplaint.isin(
                df.AE_PresentingComplaint.value_counts().head(50).index
            )
        ) & (df.AE_PresentingComplaint.notna())
        df.loc[mask, "AE_PresentingComplaint"] = np.nan

        # Laboratory results cleaning:
        # We eliminate these string values, as they make up very very few entries (mostly single digits each)
        delete_from_bloods = dict(
            Blood_Urea=[">107.0", ">53.6"],
            Blood_Sodium=["<100", "<80", ">200", ">180"],
            Blood_Potassium=[">10.0", "<1.5"],
            Blood_Creatinine=[">2210"],
            Blood_DDimer=["<21", "<100", ">69000"],
            Blood_Albumin=["<10"],
            VBG_O2=["<0.8"],
        )
        df.replace(
            {
                col: {val: np.nan for val in vals}
                for timeseries, vals in delete_from_bloods.items()
                for col in SalfordTimeseries[timeseries]
            },
            inplace=True,
        )

        # These string values in the blood results make up a large proportion of entries
        # We replace them with a midpoint value, such as the midpoint between the value and zero
        # or the midpoint between the value and the next smallest recorded value
        convert_in_bloods = dict(
            Blood_Urea={"<1.8": 0.9},
            Blood_Creatinine={"<18": 9},
            Blood_DDimer={"<150": 125},
            Blood_CRP={"<4.0": 2.25, "<0.5": 0.25},
        )
        df.replace(
            {
                col: replacements
                for timeseries, replacements in convert_in_bloods.items()
                for col in SalfordTimeseries[timeseries]
            },
            inplace=True,
        )

        return cls(df).clean_icd10()


SalfordFeatures = DotDict(
    Admin=["SpellSerial", "PatientNumber",],
    Spell=[
        "AdmissionDate",
        "TotalLOS",
        "LOSBand",
        "Outcode_Area",
        "DischargeDate",
        "CareHome",
        "HasDNAR",
    ],
    Demographics=["Female", "Age", "Ethnicity",],
    AE=[
        "AE_PresentingComplaint",
        "AE_MainDiagnosis",
        "AE_Arrival",
        "AE_Departure",
        "AE_Location",
        "AE_PatientGroup",
        "AE_TriageNote",
    ],
    Text=[
        "AE_PresentingComplaint",
        "AE_MainDiagnosis",
        "AE_TriageNote",
        "MainDiagnosis",
        "MainProcedure",
    ],
    Categorical=[
        "LOSBand",
        "Ethnicity",
        "AE_Location",
        "AE_PatientGroup",
        "AdmitMethod",
        "AdmissionSpecialty",
        "DischargeSpecialty",
        "DischargeDestination",
    ],
    Admission=["ElectiveAdmission", "AdmitMethod", "AdmissionSpecialty"],
    Discharge=[
        "DischargeConsultant",
        "DischargeSpecialty",
        "DischargeDestination",
        "DiedDuringStay",
        "DiedWithin30Days",
    ],
    Wards=["AdmitWard"] + [f"NextWard{_}" for _ in range(2, 10)] + ["DischargeWard"],
    WardLOS=["AdmitWardLOS"]
    + [f"NextWardLOS{_}" for _ in range(2, 10)]
    + ["DischargeWardLOS"],
    Diagnoses=["MainICD10"] + [f"SecDiag{_}" for _ in range(1, 16)],
    Operations=["MainOPCS4"] + [f"SecOper{_}" for _ in range(1, 15)],
    Blood=[
        col
        for parent, cols in SalfordTimeseries.items()
        for col in cols
        if str(parent).startswith("Blood_")
    ],
    AdmissionBlood=[
        col
        for parent, cols in SalfordTimeseries.items()
        for col in cols
        if str(parent).startswith("Blood_") and str(col).endswith("_Admission")
    ],
    VBG=[
        col
        for parent, cols in SalfordTimeseries.items()
        for col in cols
        if str(parent).startswith("VBG_")
    ],
    NEWS=[
        col
        for parent, cols in SalfordTimeseries.items()
        for col in cols
        if str(parent).startswith("NEWS_")
    ],
    AdmissionNEWS=[
        col
        for parent, cols in SalfordTimeseries.items()
        for col in cols
        if str(parent).startswith("NEWS_") and str(col).endswith("_Admission")
    ],
    CompositeScores=["CFS_Score", "Waterlow_Score", "Waterlow_Outcome"],
)

SalfordFeatures["TimeSeries"] = dict(
    NEWS=SalfordFeatures["NEWS"],
    Blood=SalfordFeatures["Blood"],
    VBG=SalfordFeatures["VBG"],
)
