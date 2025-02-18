import numpy as np
import pandas as pd
from typing import List
from libhydro.conv.xml import xml_parser


def _read_frc_from_xml_sandre(xml_file: str) -> pd.DataFrame:
    # read in XML Sandre file with `libhydro`
    x = xml_parser.parse_xml_file(xml_file)

    prd = None

    for sim in x['simulations']:
        # extract ensemble predictions dataframe
        df = sim.prevs_ensemble

        # select result column (drop weight column)
        df = df.loc[:, ['res']]

        # collect forecast issue date
        issue_date = sim.dtderobs

        # convert validity dates into leadtimes
        df.index = df.index.set_levels(
            df.index.levels[0] - issue_date,
            level=0
        )

        # rename column and column index
        df = df.rename(columns={'res': issue_date})
        df.columns.name = "date émission"

        # rename multi-index levels
        df.index = df.index.rename(
            {
                'lb': 'membres',
                'dte': 'échéances'
            }
        )

        # turn column index into level in row multi-index
        s = df.stack('date émission')

        # prepend level to row multi-index for sites
        s = pd.concat({sim.entite.code: s}, names=['entités'])

        # reorder levels in row multi-index to match evalhyd convention
        s.index = s.index.reorder_levels(
            ['entités', 'échéances', 'membres', 'date émission']
        )

        # concatenate with other sites
        prd = pd.concat([prd, s])

    return prd


def _convert_frc_df_to_arr(df: pd.DataFrame) -> np.ndarray:
    # convert emission dates to validity dates
    df = df.to_frame(name='valeur')

    # compute validity dates from issue dates and lead times
    validity_dates = (
        df.index.get_level_values('date émission')
        + df.index.get_level_values('échéances')
    )

    # drop issue dates
    df = df.droplevel('date émission', axis=0)

    # create new column with validity dates
    df.loc[:, 'date validité'] = validity_dates

    # use validity dates as column index
    df = df.pivot(columns='date validité', values='valeur')

    # append column index as additional level in row index
    df = df.stack(level='date validité', dropna=False)

    # determine shape of array (sites, leadtimes, members, time)
    shape = tuple(map(len, df.index.levels))

    # map dataframe data into array
    arr = np.full(shape, np.nan)
    arr[tuple(df.index.codes)] = df.values.flat

    return arr


def read_frc_from_xml_sandre(
        xml_files: List[str], return_format='pandas'
) -> pd.DataFrame | np.ndarray:
    """Read Sandre XML files containing streamflow forecasts and return
    as a Python data structure (either a `pandas.Series` or
    `numpy.ndarray`).

    :Parameters:

        xml_files: `list`
            The list of Sandre XML files from which to extract
            streamflow forecasts.

        return_format: `str`, optional
            The desired returned format, either ``'pandas'`` for a
            `pandas.Series` or ``'numpy'`` for a `numpy.ndarray`. If
            not provided, a `pandas.Series` is returned.

    :Returns:

        `pandas.Series` or `numpy.ndarray`
            The data structure containing the streamflow forecasts.

    **Examples**

    Retreiving streamflow forecasts as a series:

    >>> s = read_frc_from_xml_sandre(['data/GRP_B_20241211_1023_5304.xml'])
    >>> s.xs('K0045510', level='entités', drop_level=False).xs('0001', level='membres', drop_level=False)
    entités   échéances        membres  date émission
    K0045510  0 days 01:00:00  0001     2024-12-11 10:00:00    558.0
              0 days 02:00:00  0001     2024-12-11 10:00:00    553.0
              0 days 03:00:00  0001     2024-12-11 10:00:00    547.0
              0 days 04:00:00  0001     2024-12-11 10:00:00    541.0
              0 days 05:00:00  0001     2024-12-11 10:00:00    535.0
                                                               ...
              4 days 20:00:00  0001     2024-12-11 10:00:00    922.0
              4 days 21:00:00  0001     2024-12-11 10:00:00    904.0
              4 days 22:00:00  0001     2024-12-11 10:00:00    886.0
              4 days 23:00:00  0001     2024-12-11 10:00:00    869.0
              5 days 00:00:00  0001     2024-12-11 10:00:00    852.0
    Length: 120, dtype: float64

    Retreiving streamflow forecasts as an array:

    >>> arr = read_frc_from_xml_sandre(
    ...     ['data/GRP_B_20241211_1023_5304.xml'], return_format='numpy'
    ... )  # doctest: +ELLIPSIS
    >>> arr[0, :, 0, :]
    array([[558.,  nan,  nan, ...,  nan,  nan,  nan],
           [ nan, 553.,  nan, ...,  nan,  nan,  nan],
           [ nan,  nan, 547., ...,  nan,  nan,  nan],
           ...,
           [ nan,  nan,  nan, ..., 886.,  nan,  nan],
           [ nan,  nan,  nan, ...,  nan, 869.,  nan],
           [ nan,  nan,  nan, ...,  nan,  nan, 852.]])
    """

    # check requested return format
    if return_format not in ('numpy', 'pandas'):
        raise ValueError("return_format must be 'numpy' or 'pandas'")

    # loop through XML files
    prd = None
    for xml_file in xml_files:
        prd = pd.concat([prd, _read_frc_from_xml_sandre(xml_file)])

    # return in requested format
    if return_format == 'numpy':
        return _convert_frc_df_to_arr(prd)
    else:  # 'pandas'
        return prd
