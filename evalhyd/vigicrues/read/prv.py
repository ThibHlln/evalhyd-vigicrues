import pandas as pd
from typing import List
from pyspc.io.prv import read_prv


def read_prd_from_prv(prv_files: List[str], datatype: str) -> pd.DataFrame:
    """Lire les fichiers au format PRV contenant les prédictions
    de débits et retourner sous forme de `pandas.DataFrame`.

    :Paramètres:

        xml_files: `list`
            La liste de fichiers au format PRV contenant les prédictions
            de débits.

    :Retourne:

        `pandas.DataFrame`
            La structure de données contenant les prédictions de débits.

    **Exemples**

    Récupérer les prédictions de débits sous forme de dataframe :

    >>> df = read_prd_from_prv(['data/GRP_B_20241211_1023_5304.prv'])
    >>> df.xs('K0045510', level='entités', drop_level=False).xs('0001', level='membres', drop_level=False)
                                                            valeur
    entités  échéances       membres date validité
    K0045510 0 days 01:00:00 0001    2024-12-11 11:00:00   0.558
             0 days 02:00:00 0001    2024-12-11 12:00:00   0.553
             0 days 03:00:00 0001    2024-12-11 13:00:00   0.547
             0 days 04:00:00 0001    2024-12-11 14:00:00   0.541
             0 days 05:00:00 0001    2024-12-11 15:00:00   0.535
    ...                                                      ...
             4 days 20:00:00 0001    2024-12-16 06:00:00   0.922
             4 days 21:00:00 0001    2024-12-16 07:00:00   0.904
             4 days 22:00:00 0001    2024-12-16 08:00:00   0.886
             4 days 23:00:00 0001    2024-12-16 09:00:00   0.869
             5 days 00:00:00 0001    2024-12-16 10:00:00   0.852
    [120 rows x 1 columns]
    """
    df1 = None

    for prv_file in prv_files:
        # read in PRV file with `pyspc`
        s = read_prv(prv_file, datatype=datatype)

        # retrieve dataframe from `pyspc.Series`
        df0 = s.concat()

        # flatten nested column multi-index into simple multi-index
        df0.columns = pd.MultiIndex.from_tuples(
            [tuple([c[0], c[1], *c[2]]) for c in df0.columns],
            names=[
                'entités', 'variables', 'date émission',
                'modèles', 'membres', 'probabilités'
            ]
        )

        # drop irrelevant levels for `evalhyd`
        df0 = df0.droplevel(('variables', 'modèles', 'probabilités'), axis=1)

        # give name to current row index
        df0.index.names = ['date validité']

        # move column multi-index levels to row multi-index levels
        df0 = df0.stack(df0.columns.names, future_stack=True).to_frame('valeur')

        # compute leadtimes from validity dates and issue dates
        df0.loc[:, 'échéances'] = (
                df0.index.get_level_values('date validité')
                - df0.index.get_level_values('date émission')
        )

        # introduce new level in row multi-index for leadtimes
        df0 = df0.set_index('échéances', append=True)

        # drop issue dates level from row multi-index
        df0 = df0.droplevel('date émission', axis=0)

        # reorder levels in row multi-index to match evalhyd convention
        df0.index = df0.index.reorder_levels(
            ['entités', 'échéances', 'membres', 'date validité']
        )

        # sort index to guarantee later conversion to array is safe
        df0 = df0.sort_index()

        df1 = pd.concat([df1, df0])

    return df1
