import pandas as pd
import numpy as np
import json
from collections import defaultdict

from .salford_raw import (
    SalfordTimeseries,
    RedundantColumns,
    RawTimeseries,
    AEDiagnosisStems,
    AEVaguePresentingComplaints,
    AVCPU_Ordered_Scale,
    Pain_Ordered_Scale,
    SalfordTimeseriesPrettyPrint,
    SalfordTimeseriesUnits,
)
from .icd10 import ICD10Table
from .ccs import CCSTable
from .utils import DotDict, Series


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
                return SalfordData(self[SalfordFeatures[name]])
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
        wards=["CCU", "HH1M"],
        ignore_admit_ward=False,
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

    def expand_icd10_definitions(self, clean_icd10=True, return_df=False):
        """ Converts the ICD-10 diagnosis entries full description strings
        :param clean_icd10: Whether to standardise the ICD-10 codes first
        :returns: New SalfordData instance with the entries altered
            if return_df: pd.DataFrame instance with the cleaned entries
        """
        diagnoses = self.clean_icd10(return_df=True) if clean_icd10 else self.Diagnoses

        df = diagnoses.applymap(ICD10Table.code_description_lookup().get)

        return self._finalize_derived_feature_wide(
            df, SalfordFeatures.Diagnoses, return_df
        )

    def expand_ccs_definitions(self, derive_ccs=True, grouping="HSMR", return_df=False):
        """ Converts the the CCS codes for the patients' ICD-10 diagnoses into description strings
        :param derive_ccs: Whether to group the ICD-10 codes first. REQUIRED if it has not already been done.
        :param grouping: CCS sub-grouping scheme to use. Must be one of ['SHMI', 'HSMR', None]. 
        :returns: New SalfordData instance with the CCS in place of the ICD-10
            if return_df: pd.DataFrame instance with the cleaned entries
        :raises ValueError: If the $grouping value is invalid.
        """
        diagnoses = (
            self.derive_ccs(grouping=grouping, return_df=True)
            if derive_ccsb
            else self.Diagnoses
        )

        if grouping is None:
            df = diagnoses.applymap(CCSTable.ccs_description_lookup().get)
        elif grouping == "SHMI":
            df = diagnoses.applymap(CCSTable.shmi_description_lookup().get)
        elif grouping == "HSMR":
            df = diagnoses.applymap(CCSTable.hsmr_description_lookup().get)
        else:
            raise ValueError('The `grouping` must be one of "SHMI", "HSMR", or None.')

        return self._finalize_derived_feature_wide(
            df, SalfordFeatures.Diagnoses, return_df
        )

    def augment_derive_all(self, within=1):
        return SalfordData(
            pd.concat(
                [
                    self, # self.drop("AE_MainDiagnosis", axis=1),
                    self.derive_mortality(within=within, return_series=True),
                    self.derive_critical_care(within=within, return_series=True),
                    self.derive_critical_event(within=within, return_series=True),
                    self.derive_readmission(return_series=True),
                    # self.derive_readmission_band(return_series=True),
                    self.derive_sdec(return_series=True),
                    # self.derive_ae_diagnosis_stems(return_series=True),
                    self.derive_charlson_index(return_series=True),
                ],
                axis=1,
            )
        )

    def convert_str_to_categorical(self, columns=None, inplace=True):
        """ Transforms all string columns into categorical types. Obs_AVCPU and Obs_Pain are ordinal (ordered categorical)
        :columns: List of columns to convert. If empty, selects all string/object type columns
        :returns SalfordData instance with the columns altered 
            if inplace: Reference to the existing, altered SalfordData instance
        """
        df = self if inplace else self.copy()

        # AVCPU
        # Convert to ordinal categorical
        avcpu = list(set(SalfordTimeseries["Obs_AVCPU"]).intersection(df.columns))
        category = pd.CategoricalDtype(AVCPU_Ordered_Scale, ordered=True)
        df[avcpu] = df[avcpu].apply(lambda s: s.astype(category))

        # Pain
        # Convert to ordinal categorical
        pain = list(set(SalfordTimeseries["Obs_Pain"]).intersection(df.columns))
        category = pd.CategoricalDtype(Pain_Ordered_Scale, ordered=True)
        df[pain] = df[pain].apply(lambda s: s.astype(category))

        # Rest
        columns = columns or df.select_dtypes("object").columns
        df[columns] = df[columns].astype("category")

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

        # Filter patients under 16
        df = df[df.Age >= 16].copy()

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

        # Strip all string columns
        columns = df.select_dtypes("object").columns
        df[columns] = df[columns].apply(lambda s: s.str.strip())

        return cls(df).clean_icd10().clean_obs()

    def clean_icd10(self, return_df=False):
        """ Standardises the ICD-10 diagnosis entries to match the lookup table 
        :returns: New SalfordData instance with the entries altered
            if return_df: pd.DataFrame instance with the cleaned entries
        """
        df = ICD10Table.fuzzy_match(self.Diagnoses)

        return self._finalize_derived_feature_wide(
            df, SalfordFeatures.Diagnoses, return_df
        )

    def clean_obs(self, return_df=False):
        """ Prepares the manually transcriped clinical obs features
        :returns: New SalfordData instance with the entries altered
            if return_df: pd.DataFrame instance with the cleaned entries
        """
        df = self[SalfordFeatures.Obs].copy() if return_df else self

        # Respiratory Rate
        # Invert negatives, divide triple digit values by 10, clip to 5-80 BPM
        columns = df[SalfordTimeseries["Obs_RespiratoryRate"]]
        columns.apply(Series.invert_negatives).apply(
            Series.shift_triple_digit_values
        ).apply(Series.apply_clip_sentinel(5, 80))

        # Breathing Device
        # Clean up strings and keep the top 16 most frequent values (99% of total)
        columns = SalfordTimeseries["Obs_BreathingDevice"]
        df[columns] = df[columns].apply(
            lambda s: Series.topn_freq_values(
                s.str.upper()
                .str.replace("OTHER;", "")
                .str.strip()
                .str.split(";")
                .str[0],
                n=16,
            )
        )

        # O2 Sats
        # Invert negatives, multiply single digits by 10, clip to 0-100%, clip outliers <= mean-3*std
        columns = df[SalfordTimeseries["Obs_O2Sats"]]
        columns.apply(Series.invert_negatives).apply(
            Series.shift_single_digit_values
        ).apply(Series.apply_clip_sentinel(0, 100))
        columns.apply(
            Series.apply_clip_sentinel(
                columns.stack().mean() - 3 * columns.stack().std(), 100
            )
        )

        # Temperature
        # Invert negatives, clip to 25-45 degrees C
        columns = df[SalfordTimeseries["Obs_Temperature"]]
        columns.apply(Series.invert_negatives).apply(Series.apply_clip_sentinel(25, 45))

        # Blood Pressure
        # Clip diastolic to 20-200 mmHg
        columns_dia = df[SalfordTimeseries["Obs_DiastolicBP"]]
        columns_dia.apply(Series.apply_clip_sentinel(20, 200))

        # Clip systolic to 40-300 mmHg, clip outliers <= diastolic+5
        columns_sys = df[SalfordTimeseries["Obs_SystolicBP"]]
        columns_sys.apply(Series.apply_clip_sentinel(40, 300))
        df[columns_sys.columns] = columns_sys.mask(
            columns_sys.le(columns_dia.values + 5), np.nan,
        )

        # Heart Rate
        # Invert negatives, clip to 25-300 BPM
        columns = df[SalfordTimeseries["Obs_Temperature"]]
        columns.apply(Series.invert_negatives).apply(
            Series.apply_clip_sentinel(25, 300)
        )

        # Nausea & Vomiting
        # Convert to boolean
        columns = SalfordTimeseries["Obs_Nausea"]
        df[columns] = df[columns].__eq__("1 - Nausea present")

        columns = SalfordTimeseries["Obs_Vomiting"]
        df[columns] = df[columns].__eq__("1 - Vomiting since last round")

        return df

    def single_imputation_obs_and_news(self, return_df=False):
        """ Quickly imputes the obs and NEWS scores
        :returns: New SalfordData instance with the entries altered
            if return_df: pd.DataFrame instance with the imputed entries
        """
        df = (
            self[SalfordFeatures.Obs + SalfordFeatures.NEWS].copy()
            if return_df
            else self
        )
        # Start with the zero-fills: All the NEWS scores
        static_fills = {_: 0 for _ in SalfordFeatures.NEWS}

        # Next, the numericals: All the obs, except the categoricals and binaries
        # Calling median with numeric_only=True removes the categoricals for us but not the binaries
        static_fills |= df[SalfordFeatures.Obs].median(numeric_only=True).to_dict()

        # Finally, the categoricals: Pain, AVCPU, Breathing device
        # Also, the binaries: Nausea and Vomiting
        categorical_fills = {
            "Obs_AVCPU": AVCPU_Ordered_Scale[0],  # A - Alert
            "Obs_Pain": Pain_Ordered_Scale[0],  # 0 - No pain
            "Obs_BreathingDevice": "A - AIR",
            "Obs_Nausea": 0,
            "Obs_Vomiting": 0,
        }
        static_fills |= {
            column: fill
            for series, fill in categorical_fills.items()
            for column in set(SalfordTimeseries[series]).intersection(df.columns)
        }

        df.fillna(static_fills, inplace=True)

        return df

    def single_imputation_bloods(self, return_df=False):
        """ Quickly imputes the laboratory results
        :returns: New SalfordData instance with the entries altered
            if return_df: pd.DataFrame instance with the imputed entries
        """
        df = self[SalfordFeatures.Blood].copy() if return_df else self

        static_fills = {_: df[_].median() for _ in SalfordFeatures.Blood}

        df.fillna(static_fills, inplace=True)

        return df

    def tabular_to_text(self, columns=None):
        """ Converts the structured records into unstructured text 
        :param columns: The columns to include in the conversion
        :returns: pd.Series with the text-ified records """
        df = self[columns] if columns is not None else self

        demographics = []

        if "Age" in df:
            demographics.append(df.Age.astype(str) + " years old")
        if "Ethnicity" in df:
            demographics.append(df.Ethnicity)
        if "Female" in df:
            demographics.append(df.Female.replace({True: "Female", False: "Male"}))

        result = None
        if len(demographics):
            result = (
                pd.concat(demographics, axis=1).apply(
                    lambda row: " ".join(row.dropna()), axis=1
                )
                + "; "
            )


        diagnoses = (
            df.loc[:, df.columns.isin(SalfordFeatures.Diagnoses)].apply(
                lambda row: "; ".join(row.dropna()), axis=1
            )
            + "; "
        )
        result = diagnoses if result is None else (result + diagnoses)

        df = df.drop(
            SalfordFeatures.Demographics + SalfordFeatures.Diagnoses,
            axis=1,
            errors="ignore",
        ).rename(columns=SalfordPrettyPrint)

        units = defaultdict(lambda: "") | {
            SalfordPrettyPrint[column]: unit
            for column, unit in SalfordFeatureUnits.items()
        }

        # String columns
        result += (
            df.select_dtypes(object).apply(
                lambda row: "; ".join(
                    (f'{_[0]} is "{_[1]}"' for _ in row.dropna().items())
                ),
                axis=1,
            )
            + "; "
        )

        # Boolean columns
        result += (
            df.select_dtypes(bool).apply(
                lambda row: "; ".join((_[0] for _ in row.dropna().items() if _[1])),
                axis=1,
            )
            + "; "
        )

        # Numerical columns
        result += (
            df.select_dtypes(np.number).apply(
                lambda row: "; ".join(
                    (f"{_[0]} is {_[1]} {units[_[0]]}" for _ in row.dropna().items())
                ),
                axis=1,
            )
            + "; "
        )

        return result.str.lower()

    def bool_to_int(self, inplace=True):
        df = self if inplace else self.copy()
        mask = df.select_dtypes(bool)
        df[mask.columns] = mask.astype(int)

        return df

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
    Obs=[
        col
        for parent, cols in SalfordTimeseries.items()
        for col in cols
        if str(parent).startswith("Obs_")
    ],
    CompositeScores=["CFS_Score", "Waterlow_Score", "Waterlow_Outcome"],
    First_Blood=[
        cols[0]
        for parent, cols in SalfordTimeseries.items()
        if str(parent).startswith("Blood_")
    ],
    First_VBG=[
        cols[0]
        for parent, cols in SalfordTimeseries.items()
        if str(parent).startswith("VBG_")
    ],
    First_NEWS=[
        cols[0]
        for parent, cols in SalfordTimeseries.items()
        if str(parent).startswith("NEWS_")
    ],
    First_Obs=[
        cols[0]
        for parent, cols in SalfordTimeseries.items()
        if str(parent).startswith("Obs_")
    ],
)

SalfordPrettyPrint = (
    {__: __ for _ in SalfordFeatures.values() for __ in _}
    | {
        "PatientNumber": "Patient Number",
        "Female": "Sex",
        "AE_PresentingComplaint": "Presenting Complaint",
        "AE_Arrival": "A&E Arrival Time",
        "AE_Departure": "A&E Departure Time",
        "AE_Location": "A&E Location",
        "AE_PatientGroup": "A&E Patient Group",
        "AE_TriageNote": "Triage Note",
        "AdmissionDate": "Admission Date",
        "DischargeDate": "Discharge Date",
        "ElectiveAdmission": "Elective Admission",
        "AdmitMethod": "Admission Pathway",
        "AdmissionSpecialty": "Admitting Specialty",
        "DischargeConsultant": "Discharge Consultant",
        "DischargeSpecialty": "Discharge Specialty",
        "TotalLOS": "Length of Stay",
        "LOSBand": "LOS Band",
        "AdmitWard": "Admitting Ward",
        "AdmitWardLOS": "Admitting Ward LOS",
        "DischargeWard": "Discharge Ward",
        "DischargeWardLOS": "Discharge Ward LOS",
        "MainICD10": "Main ICD-10",
        "MainDiagnosis": "Main Diagnosis",
        "MainOPCS4": "Main OPCS-4",
        "MainProcedure": "Main Procedure",
        "Outcode_Area": "Outcode Area",
        "CareHome": "Care Home",
        "DiedDuringStay": "Died During Stay",
        "DiedWithin30Days": "Died Within 30 Days",
        "DischargeDestination": "Discharge Destination",
        "CFS_Score": "Clinical Frailty Score",
        "Waterlow_Score": "Waterlow Score",
        "Waterlow_Outcome": "Waterlow Outcome",
        "HasDNAR": "Has DNAR",
        "CriticalCare": "Critical Care",
        "CriticalEvent": "Critical Event",
        "Readmission": "Readmission",
        "SentToSDEC": "SDEC Allocation",
        "AE_MainDiagnosis": "A&E Diagnosis",
        "CharlsonIndex": "Charlson Index",
    }
    | SalfordTimeseriesPrettyPrint
)

SalfordFeatureUnits = (
    {
        "Age": "Years",
        "TotalLOS": "Days",
        "CFS_Score": "",
        "Waterlow_Score": "",
        "CharlsonIndex": "",
    }
    | {_: "Days" for _ in SalfordFeatures.WardLOS}
    | {
        col: SalfordTimeseriesUnits[parent]
        for parent, cols in SalfordTimeseries.items()
        for col in cols
    }
)

def _generate_salford_feature_combinations():
    r = DotDict()
    r['news'] = [
        "Obs_RespiratoryRate_Admission",
        "Obs_O2Sats_Admission",
        "Obs_Temperature_Admission",
        "Obs_SystolicBP_Admission",
        "Obs_HeartRate_Admission",
        "Obs_AVCPU_Admission",
    ]
    r['with_phenotype'] = r['news'] + [
        "Obs_BreathingDevice_Admission",
        "Obs_DiastolicBP_Admission",
        "Obs_Pain_Admission",
        "Obs_Nausea_Admission",
        "Obs_Vomiting_Admission",
    ] + SalfordFeatures.Demographics

    r["with_composites"] = r["with_phenotype"] + SalfordFeatures.CompositeScores + ['CharlsonIndex']
    r["with_labs"] = r["with_composites"] + SalfordFeatures.First_Blood
    r["with_services"] = r["with_labs"] + ["AdmitMethod", "AdmissionSpecialty", "SentToSDEC", "Readmission"]
    return r

SalfordCombinations = _generate_salford_feature_combinations()