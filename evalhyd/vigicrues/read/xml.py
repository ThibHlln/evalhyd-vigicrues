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


def convert_frc_df_to_arr(df: pd.DataFrame) -> np.ndarray:
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
        xml_files: List[str], return_format='dataframe'
) -> pd.DataFrame | np.ndarray:
    """Lire les fichiers au format XML-SANDRE contenant les prédictions
    de débits et retourner sous forme de structure de données Python
    (soit une `pandas.DataFrame` ou une `numpy.ndarray`).

    :Paramètres:

        xml_files: `list`
            La liste de fichiers au format XML-SANDRE contenant les
            prédictions de débits.

        return_format: `str`, optionnel
            The desired returned format, either ``'dataframe'`` for a
            `pandas.DataFrame` or ``'array'`` for a `numpy.ndarray`. If
            an array in requested, its shape corresponds to `evalhyd`
            convention, i.e. (sites, lead times, members, time). If not
            provided, a `pandas.DataFrame` is returned.

            Le format désiré pour les prédictions de débit, soit
            ``'dataframe'`` pour obtenir une `pandas.DataFrame` ou
            ``'array'`` pour obtenir une `numpy.ndarray`. Si le format
            n'est pas fourni, une dataframe est retournée.

    :Retourne:

        `pandas.DataFrame` ou `numpy.ndarray`
            La structure de données contenant les prédictions de débits.

    **Exemples**

    Récupérer les prédictions de débits sous forme de dataframe :

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

    Récupérer les prédictions de débits sous forme de matrice :

    >>> arr = read_frc_from_xml_sandre(
    ...     ['data/GRP_B_20241211_1023_5304.xml'], return_format='array'
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
    if return_format not in ('dataframe', 'array'):
        raise ValueError("return_format doit être 'dataframe' ou 'array'")

    # loop through XML files
    prd = None
    for xml_file in xml_files:
        prd = pd.concat([prd, _read_frc_from_xml_sandre(xml_file)])

    # return in requested format
    if return_format == 'array':
        return convert_frc_df_to_arr(prd)
    else:  # 'dataframe'
        return prd
