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

        # rename result column
        df = df.rename(columns={'res': 'valeur'})

        # rename multi-index levels
        df.index = df.index.rename(
            {
                'lb': 'membres',
                'dte': 'date validité'
            }
        )

        # create new column with validity dates
        df.loc[:, 'échéances'] = (
            df.index.get_level_values('date validité') - issue_date
        )

        # append column as additional level in row multi-index
        df = df.set_index('échéances', append=True)

        # prepend level to row multi-index for sites
        df = pd.concat({sim.entite.code: df}, names=['entités'])

        # reorder levels in row multi-index to match evalhyd convention
        df.index = df.index.reorder_levels(
            ['entités', 'échéances', 'membres', 'date validité']
        )

        # concatenate with other sites
        prd = pd.concat([prd, df])

    return prd


def _convert_frc_df_to_arr(df: pd.DataFrame) -> np.ndarray:
    # turn validity dates level of row multi-index into column
    df = df.reset_index().set_index(['entités', 'échéances', 'membres'])

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
        xml_files: List[str], return_type='dataframe'
) -> pd.DataFrame | np.ndarray:
    """Read Sandre XML files containing streamflow forecasts and return
    as a Python data structure (either a `pandas.DataFrame` or
    `numpy.ndarray`).

    :Parameters:

        xml_files: `list`
            The list of Sandre XML files from which to extract
            streamflow forecasts.

        return_type: `str`, optional
            The desired returned format, either ``'dataframe'`` for a
            `pandas.DataFrame` or ``'array'`` for a `numpy.ndarray`. If
            an array in requested, its shape corresponds to `evalhyd`
            convention, i.e. (sites, lead times, members, time). If not
            provided, a `pandas.DataFrame` is returned.

    :Returns:

        `pandas.DataFrame` or `numpy.ndarray`
            The data structure containing the streamflow forecasts.

    **Examples**

    Retreiving streamflow forecasts as a dataframe:

    >>> df = read_frc_from_xml_sandre(['data/GRP_B_20241211_1023_5304.xml'])
    >>> df.xs('K0045510', level='entités', drop_level=False).xs('0001', level='membres', drop_level=False)
                                                            valeur
    entités  échéances       membres date validité
    K0045510 0 days 01:00:00 0001    2024-12-11 11:00:00 558.00000
             0 days 02:00:00 0001    2024-12-11 12:00:00 553.00000
             0 days 03:00:00 0001    2024-12-11 13:00:00 547.00000
             0 days 04:00:00 0001    2024-12-11 14:00:00 541.00000
             0 days 05:00:00 0001    2024-12-11 15:00:00 535.00000
    ...                                                        ...
             4 days 20:00:00 0001    2024-12-16 06:00:00 922.00000
             4 days 21:00:00 0001    2024-12-16 07:00:00 904.00000
             4 days 22:00:00 0001    2024-12-16 08:00:00 886.00000
             4 days 23:00:00 0001    2024-12-16 09:00:00 869.00000
             5 days 00:00:00 0001    2024-12-16 10:00:00 852.00000
    [120 rows x 1 columns]

    Retreiving streamflow forecasts as an array:

    >>> arr = read_frc_from_xml_sandre(
    ...     ['data/GRP_B_20241211_1023_5304.xml'], return_type='array'
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
    if return_type not in ('dataframe', 'array'):
        raise ValueError("return_type must be 'dataframe' or 'array'")

    # loop through XML files
    prd = None
    for xml_file in xml_files:
        prd = pd.concat([prd, _read_frc_from_xml_sandre(xml_file)])

    # return in requested format
    if return_type == 'array':
        return _convert_frc_df_to_arr(prd)
    else:  # 'dataframe'
        return prd
