import numpy as np
import pandas as pd


def convert_obs_df_to_arr(df: pd.DataFrame) -> np.ndarray:
    # turn validity dates level of row multi-index into column
    df = df.reset_index().set_index('entités')

    # determine shape of array (sites, leadtimes, members, time)
    shape = tuple(map(len, df.index.levels))

    # map dataframe data into array
    arr = np.full(shape, np.nan)
    arr[tuple(df.index.codes)] = df.values.flat

    return arr


def convert_prd_df_to_arr(df: pd.DataFrame) -> np.ndarray:
    # turn validity dates level of row multi-index into column
    df = df.reset_index().set_index(['entités', 'échéances', 'membres'])

    # use validity dates as column index
    df = df.pivot(columns='date validité', values='valeur')

    # append column index as additional level in row index
    df = df.stack(level='date validité', future_stack=True)

    # determine shape of array (sites, leadtimes, members, time)
    shape = tuple(map(len, df.index.levels))

    # map dataframe data into array
    arr = np.full(shape, np.nan)
    arr[tuple(df.index.codes)] = df.values.flat

    return arr
