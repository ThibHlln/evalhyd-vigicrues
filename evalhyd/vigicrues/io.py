import numpy as np
import pandas as pd
from typing import List
from libhydro.conv.xml import xml_parser


def _read_xml_sandre(xml_file: str) -> pd.DataFrame:
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


def _convert_df_to_arr(df: pd.DataFrame) -> np.ndarray:
    # determine shape of array (sites, leadtimes, members, time)
    shape = tuple(map(len, df.index.levels))

    # map dataframe data into array
    arr = np.full(shape, np.nan)
    arr[tuple(df.index.codes)] = df.values.flat

    return arr


def read_xml_sandre(xml_files: List[str]) -> pd.DataFrame:
    prd = None

    for xml_file in xml_files:
        prd = pd.concat([prd, _read_xml_sandre(xml_file)])

    return prd
