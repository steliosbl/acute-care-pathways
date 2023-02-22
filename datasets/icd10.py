import string 

import pandas as pd
import numpy as np

from .utils import justify

class ICD10Table:
    """ Represents the ICD-10 standard lookup table and related methods to utilise it """
    FILENAME = 'data/icd10/icd10.h5'
    KEY = 'ICD10_Codes'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def fuzzy_match(cls, df, icd10=None):
        """ Given a pd.DataFrame of ICD-10 codes in SCI dataset wide format, 
        transforms the codes to match the lookup table closely as possible
        :param df: DataFrame to match the contents of
        :param icd10: The ICD-10 lookup table. If None, will be loaded first
        :returns: DataFrame with altered codes """

        if icd10 is None:
            icd10 = pd.read_hdf(cls.FILENAME, cls.KEY)

        r = df.stack()

        # Fix entries matching 'A12.34 D' or 'A12.X'
        mask = r.str.contains(" ") | r.str.endswith(".X")
        r[mask] = r[mask].str[:-2]

        # Fix entries matching 'A12.34D'
        mask = r.str[-1].isin(frozenset(string.ascii_uppercase))
        r[mask] = r.str[:-1]

        # Delete entries not in the external table
        mask = ~r.isin(frozenset(icd10.index))
        r[mask] = np.nan
        r = r.unstack()

        # Justify dataframe after deleting values
        result = justify(r)
        result.index=r.index

        return r
