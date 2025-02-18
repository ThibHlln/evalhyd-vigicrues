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
