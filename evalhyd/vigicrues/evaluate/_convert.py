import numpy as np
import pandas as pd


def convert_obs_df_to_arr(df: pd.DataFrame) -> np.ndarray:
    # determine shape of potentially sliced dataframe
    # (sites, leadtimes, members, time)
    #     /!\ not possible directly with `df.index.levshape` as this
    #         returns the shape of the original dataframe (prior to
    #         any potential slicing), see for example
    #         https://github.com/pandas-dev/pandas/issues/3686
    shape = tuple(len(df.index.unique(level)) for level in df.index.names)

    # map dataframe data into array
    #     /!\ `df.index.codes` is of shape of the potentially sliced
    #         dataframe but with indices referring to the locations
    #         in the original dataframe and `df.values` returns the
    #         elements in the potentially sliced dataframe so need to
    #         work on an intermediary array of original shape and then
    #         subset to turn into potentially slided shape
    arr_ = np.full(df.index.levshape, np.nan)
    arr_[tuple(df.index.codes)] = df.values.flat

    arr = np.full(shape, np.nan)
    arr[:] = arr_[tuple(df.index.codes)].reshape(shape)

    return arr


def convert_prd_df_to_arr(df: pd.DataFrame) -> np.ndarray:
    # turn validity dates level of row multi-index into column
    df = df.reset_index().set_index(df.index.names[:-1])

    # use validity dates as column index
    df = df.pivot(columns='dates_validite', values='valeur')

    # append column index as additional level in row index
    df = df.stack(level='dates_validite', future_stack=True)

    # determine shape of potentially sliced dataframe
    # (sites, leadtimes, members, time)
    #     /!\ not possible directly with `df.index.levshape` as this
    #         returns the shape of the original dataframe (prior to
    #         any potential slicing), see for example
    #         https://github.com/pandas-dev/pandas/issues/3686
    shape = tuple(len(df.index.unique(level)) for level in df.index.names)

    # map dataframe data into array
    #     /!\ `df.index.codes` is of shape of the potentially sliced
    #         dataframe but with indices referring to the locations
    #         in the original dataframe and `df.values` returns the
    #         elements in the potentially sliced dataframe so need to
    #         work on an intermediary array of original shape and then
    #         subset to turn into potentially slided shape
    arr_ = np.full(df.index.levshape, np.nan)
    arr_[tuple(df.index.codes)] = df.values.flat

    arr = np.full(shape, np.nan)
    arr[:] = arr_[tuple(df.index.codes)].reshape(shape)

    return arr
