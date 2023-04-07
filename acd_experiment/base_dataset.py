import pandas as pd
import numpy as np

from typing import Dict, Iterable, Tuple
from itertools import groupby


class BaseDataset(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def omit_redundant(self) -> "BaseDataset":
        pass


    @property
    def numeric_columns(self):
        return [
            col
            for col in self.select_dtypes(include="number")
            if not np.isin(self[col].dropna().unique(), [0, 1]).all()
        ]

    @property
    def binary_columns(self):
        return [
            col for col in self if np.isin(self[col].dropna().unique(), [0, 1]).all()
        ]

    def categorize(self, categories=None) -> "BaseDataset":
        r = self.apply(lambda x: x.replace({True: 1.0, False: 0.0}))

        if categories is None:
            mask = r.select_dtypes(include=object)
            r[mask.columns] = r.select_dtypes(include=object).astype("category")
        else:
            for col, cats in categories.items():
                r[col] = pd.Categorical(r[col], cats)
        
        return r

    @property
    def feature_group_combinations(self) -> Dict[str, Iterable[str]]:
        pass

    def xy(
        self,
        x: Iterable[str] = [],
        dropna: bool = False,
        fillna: bool = False,
        ordinal_encoding: bool = False,
        onehot_encoding: bool = False,
        imputation: bool = False,
        outcome: str = "CriticalEvent",
        outcome_within: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def get_onehot_categorical_columns(
        self, separator="__"
    ) -> Dict[str, Iterable[str]]:
        return {
            key: value
            for key, value in map(
                lambda _: (_[0], list(_[1])),
                groupby(sorted(self.columns), key=lambda _: _.split(separator)[0]),
            )
            if len(value) > 1
        }

    def fill_na(self):
        cat_cols = self.select_dtypes("category").columns
        other_cols = list(set(self.columns) - set(cat_cols))
        for _ in cat_cols:
            self[_] = self[_].cat.add_categories("NAN").fillna("NAN")
        self[other_cols] = self[other_cols].fillna(-1)
        return self

    @property
    def numeric_columns(self):
        return [
            col
            for col in self.select_dtypes(include="number")
            if not np.isin(self[col].dropna().unique(), [0, 1]).all()
        ]
        