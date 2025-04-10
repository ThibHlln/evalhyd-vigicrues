import pandas as pd
import io
from typing import List


def read_prd_from_prv(prv_files: List[str]) -> pd.DataFrame:
    """Lire les fichiers au format PRV contenant les prédictions
    de débits et retourner sous forme de `pandas.DataFrame`.

    :Paramètres:

        prv_files: `list`
            La liste de fichiers au format PRV contenant les prédictions
            de débits.

    :Retourne:

        `pandas.DataFrame`
            La structure de données contenant les prédictions de débits.

    **Exemples**

    Récupérer les prédictions de débits sous forme de dataframe :

    >>> df = read_prd_from_prv(['data/GRP_B_20241211_1023_5304.prv'])
    >>> df.xs('K0045510', level='entites', drop_level=False).xs('0001', level='membres', drop_level=False)
                                                          valeur
    entite   echeance        membre  date_validite
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
        # read in PRV file as text
        with open(prv_file, 'r') as f:
            txt = f.read()

            # uncomment relevant lines for creation of dataframe multi-index
            if '# Scenarios;' in txt:
                txt = txt.replace('# Scenarios;', 'Scenarios;')
            elif '# Tendances;' in txt:
                txt = txt.replace('# Tendances;', 'Tendances;')
            else:
                raise RuntimeError(
                    f"Le fichier {prv_file} ne contient pas de "
                    f"prévisions ensemblistes ou de tendances"
                )

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

        # skip timeseries that are not on streamflow
        try:
            df0 = df0.xs('Q', axis=1, level='Grandeurs', drop_level=False)
        except KeyError:
            raise RuntimeError(
                f"Le fichier {prv_file} ne contient pas de séries de débit"
            )

        # drop irrelevant levels for `evalhyd`
        df0 = df0.droplevel(('Grandeurs', 'IdSeries'), axis=1)

        # rename indexes corresponding to `evalhyd` dimensions
        df0.columns = df0.columns.rename(
            {
                'Stations': 'entite',
                'Tendances': 'tendance',
                'Scenarios': 'membre',
                'DtDerObs': 'date_emission',
            }
        )
        df0.index.name = 'date_validite'

        # move column multi-index levels to row multi-index levels
        df0 = df0.stack(
            df0.columns.names, future_stack=True
        ).to_frame('valeur')

        # parse issue dates to timestamp
        df0.index = df0.index.set_levels(
            pd.to_datetime(
                df0.index.unique('date_emission'),
                format='%d-%m-%Y %H:%M'
            ),
            level='date_emission'
        )

        # compute leadtimes from validity dates and issue dates
        df0.loc[:, 'echeance'] = (
            df0.index.get_level_values('date_validite')
            - df0.index.get_level_values('date_emission')
        )

        # introduce new level in row multi-index for leadtimes
        df0 = df0.set_index('echeance', append=True)

        # drop issue dates level from row multi-index
        df0 = df0.droplevel('date_emission', axis=0)

        # reorder levels in row multi-index to match evalhyd convention
        df0.index = df0.index.reorder_levels(
            [
                'entite',
                'echeance',
                'membre' if 'membre' in df0.index.names else 'tendance',
                'date_validite'
            ]
        )

        # sort index to guarantee later conversion to array is safe
        df0 = df0.sort_index()

        df1 = pd.concat([df1, df0])

    return df1
