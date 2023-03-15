import numpy as np
import pandas as pd
from functools import partial


class DotDict(dict):
    """ Class for forming dict/class hybrids that allow attribute access via dot notation
    e.g. MyDotDict.some_key = MyDotDict['some_key'] """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def justify(df, invalid_val=np.nan, axis=1, side="left"):
    """
    Justifies a 2D array

    Parameters
    ----------
    df : DataFrame
        Input DataFrame to be justified
    axis : int
        Axis along which justification is to be made
    side : str
        Direction of justification. It could be 'left', 'right', 'up', 'down'
        It should be 'left' or 'right' for axis=1 and 'up' or 'down' for axis=0.

    """
    if invalid_val is np.nan:
        mask = df.notna().values
    else:
        mask = df != invalid_val

    justified_mask = np.sort(mask, axis=axis)
    if (side == "up") | (side == "left"):
        justified_mask = np.flip(justified_mask, axis=axis)
    out = np.full(df.shape, invalid_val).astype("O")
    if axis == 1:
        out[justified_mask] = df.values[mask]
    else:
        out.T[justified_mask.T] = df.values.T[mask.T]

    return pd.DataFrame(out, columns=df.columns, index=df.index)

class Series:
    @staticmethod
    def topn_freq_values(s, n=10, sentinel=np.nan):
        s[
            ~s.isin(pd.DataFrame(s).stack().value_counts().head(n).index)
        ] = sentinel
        return s

    @staticmethod
    def invert_negatives(s):
        s[s < 0] *= -1
        return s

    @staticmethod
    def clip_sentinel(s, lower, upper, sentinel=np.nan):
        s[(s <= lower) | (s >= upper)] = sentinel
        return s

    @staticmethod
    def apply_clip_sentinel(lower, upper, sentinel=np.nan):
        return partial(Series.clip_sentinel, lower=lower, upper=upper, sentinel=sentinel)

    @staticmethod
    def shift_triple_digit_values(s):
        s[s >= 100] /= 10
        return s

    @staticmethod
    def shift_single_digit_values(s):
        s[s < 10] *= 10
        return s


def row_to_text(row, column_name=False):
    text = ''

    for i, c in enumerate(row):
        if column_name:
            text += f"{row.index[i]}: "

        text += str(c) + ' [SEP] '

    return text
