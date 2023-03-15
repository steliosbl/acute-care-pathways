import pandas as pd
import numpy as np

class CCSTable:
    """ Represents the CCS lookup table and related methods to utilise it """

    FILENAME = 'data/icd10/ccs.h5'
    CCS_KEY = 'codes'
    SHMI_KEY = 'shmi'
    HSMR_KEY = 'hsmr'

    @classmethod 
    def fuzzy_match(cls, df, ccs=None, sentinel=259):
        """ Given a pd.DataFrame of *standardised* ICD-10 codes in SCI dataset wide format, 
        performs a fuzzy join of the ICD-10 codes with their CCS groupings. 
        Codes that don't match exactly are joined with their 3-codes
        :param df: DataFrame to match the contents of
        :param ccs: The CCS lookup table. If None, will be loaded first
        :param sentinel: What to fill un-matched values with. Default is the code for unclassified codes
        :returns: DataFrame with altered codes """

        if ccs is None:
            ccs = pd.read_hdf(cls.FILENAME, cls.CCS_KEY)
        
        no_dot = (
            df.apply(lambda col: col.str.replace(".", "", regex=False))
            .stack()
            .rename("icd10")
            .to_frame()
        )

        perfect = no_dot.join(ccs, on="icd10").CCSGroup.unstack()
        approx = (
            no_dot.apply(lambda col: col.str[:_])
            .join(ccs, on="icd10")
            .CCSGroup.unstack()
            for _ in (3, 4)
        )

        for _ in approx:
            mask = _.notna() & perfect.isna() & df.notna()
            perfect[mask] = _[mask]

        remaining = perfect.isna() & df.notna()
        perfect[remaining] = sentinel

        perfect.columns = df.columns

        return perfect.reindex_like(df)

    @classmethod
    def _regroup(cls, df_data, df_lookup, col, prefix):
        return (
            df_data
            .stack()
            .rename('ccs')
            .to_frame()
            .join(df_lookup, on='ccs')[col]
            .unstack()
            .reindex_like(df_data)
        )

    @classmethod
    def convert_shmi(cls, df, shmi=None):
        """ Given a DataFrame of CCS codes in SCI dataset wide format,
        converts them to SHMI diagnosis groups 
        :param df: The DataFrame of CCS codes to convert 
        :returns: DataFrame with altered codes """

        if shmi is None:
            shmi = pd.read_hdf(cls.FILENAME, cls.SHMI_KEY)
        
        return cls._regroup(df, shmi, 'SHMIGroup', 'SHMI')

    @classmethod
    def convert_hsmr(cls, df, hsmr=None):
        """ Given a DataFrame of CCS codes in SCI dataset wide format,
        converts them to HSRM aggregate groups 
        :param df: The DataFrame of CCS codes to convert 
        :returns: DataFrame with altered codes """

        if hsmr is None:
            hsmr = pd.read_hdf(cls.FILENAME, cls.HSMR_KEY)
        
        return cls._regroup(df, hsmr, 'AggregateGroup', 'HSMR')
    
    