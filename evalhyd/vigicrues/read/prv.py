import pandas as pd
import io
from typing import List


def read_prd_from_prv(prv_files: List[str], datatype: str) -> pd.DataFrame:
    """Lire les fichiers au format PRV contenant les prédictions
    de débits et retourner sous forme de `pandas.DataFrame`.

    :Paramètres:

        prv_files: `list`
            La liste de fichiers au format PRV contenant les prédictions
            de débits.

        datatype: `str`
            Le type de données fournies. Il peut être défini soit
            comme ``'ensemble'`` (quand les fichiers PRV contiennent
            des scénarios) ou ``'tendance'`` (quand les fichiers PRV
            contiennent des tendances).

    :Retourne:

        `pandas.DataFrame`
            La structure de données contenant les prédictions de débits.

    **Exemples**

    Récupérer les prédictions de débits sous forme de dataframe :

    >>> df = read_prd_from_prv(['data/GRP_B_20241211_1023_5304.prv'], datatype='ensemble')
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
    # check declared datatype
    if datatype not in ('ensemble', 'tendance'):
        raise ValueError(
            "'datatype' doit être 'ensemble' ou 'tendance'"
        )

    df1 = None

    for prv_file in prv_files:
        # read in PRV file as text
        with open(prv_file, 'r') as f:
            txt = f.read()

            # uncomment relevant lines for creation of dataframe multi-index
            if datatype == 'ensemble':
                txt = txt.replace('# Scenarios;', 'Scenarios;')
                txt = txt.replace('# DtDerObs;', 'DtDerObs;')
            elif datatype == 'tendance':
                txt = txt.replace('# Tendances;', 'Tendances;')
                txt = txt.replace('# DtDerObs;', 'DtDerObs;')

        # get dataframe from text
        df0 = pd.read_csv(
            io.StringIO(txt),
            sep=';',
            comment='#',
            header=[0, 1, 2, 3, 4],
            index_col=0,
            parse_dates=True,
            date_format='%d-%m-%Y %H:%M',
            na_values=(-99.900, '-99.900', -999.999, '-999.999'),
            keep_default_na=True,
        )

        # drop irrelevant levels for `evalhyd`
        df0 = df0.droplevel(('Grandeurs', 'IdSeries'), axis=1)

        # rename indexes corresponding to `evalhyd` dimensions
        df0.columns = df0.columns.rename(
            {
                'Stations': 'entités',
                'Tendances': 'tendances',
                'Scenarios': 'membres',
                'DtDerObs': 'date émission',
            }
        )
        df0.index.name = 'date validité'

        # move column multi-index levels to row multi-index levels
        df0 = df0.stack(
            df0.columns.names, future_stack=True
        ).to_frame('valeur')

        # parse issue dates to timestamp
        df0.index = df0.index.set_levels(
            pd.to_datetime(
                df0.index.levels[df0.index.names.index('date émission')],
                format='%d-%m-%Y %H:%M'
            ),
            level='date émission'
        )

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
            [
                'entités',
                'échéances',
                'membres' if datatype == 'ensemble' else 'tendances',
                'date validité'
            ]
        )

        # sort index to guarantee later conversion to array is safe
        df0 = df0.sort_index()

        df1 = pd.concat([df1, df0])

    return df1
