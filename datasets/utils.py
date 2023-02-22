import numpy as np
import pandas as pd


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
