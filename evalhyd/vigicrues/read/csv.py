import pathlib
import pandas as pd
from typing import List


def read_obs_from_csv_hydroportail(csv_files: List[str]) -> pd.DataFrame:
    """Lire les fichiers au format CSV exportés de l'HydroPortail
    contenant les observations de débits et retourner sous forme de
    `pandas.DataFrame`.

    :Paramètres:

        csv_files: `list`
            La liste de fichiers au format CSV exportés de l'HydroPortail
            contenant les observations de débits.

            .. warning::

               Il est attendu que les noms de fichiers contiennent
               en préfixe le code entité (code site ou code station)
               suivi d'un caractère "underscore", comme c'est le cas
               lors de l'export de données depuis l'HydroPortail. Ces
               préfixes sont utilisés pour remplir le niveau "entites"
               du multi-index de la dataframe retournée.

    :Retourne:

        `pandas.DataFrame`
            La structure de données contenant les observations de débits.

    **Exemples**

    Récupérer les observations de débits sous forme de dataframe :

    >>> df = read_obs_from_csv_hydroportail(
    ...     ['K010002010_QmnJ(n=1_non-glissant).csv',
    ...      'K025302002_QmnJ(n=1_non-glissant).csv',
    ...      'K025801001_QmnJ(n=1_non-glissant).csv']
    ... )
    >>> df
                               valeur
    entite     date_validite
    K010002010 2019-01-01        3430
               2019-01-02        3320
               2019-01-03        3030
               2019-01-04        2890
               2019-01-05        2800
    ...                           ...
    K025801001 2019-12-27         623
               2019-12-28         563
               2019-12-29         505
               2019-12-30         450
               2019-12-31         416
    [1095 rows x 1 columns]
    """
    # loop through CSV files
    df1 = None

    for csv_file in csv_files:
        # determine entity code from filename
        entite = pathlib.Path(csv_file).stem.split('_')[0]

        # read in CSV with `pandas`
        df0 = pd.read_csv(
            csv_file, parse_dates=['Date (TU)'],
        )

        # remove timezone from datetime
        df0['Date (TU)'] = df0['Date (TU)'].dt.tz_localize(None)

        # check that timeseries are on streamflow
        if not (
                set(df0.columns)
                .intersection(['Valeur (en l/s)', 'Valeur (en m³/s)'])
        ):
            raise RuntimeError(
                f"Le fichier {csv_file} ne contient pas de séries de débit"
            )

        # rename columns
        df0 = df0.rename(
            columns={
                'Valeur (en l/s)': 'valeur',
                'Valeur (en m³/s)': 'valeur',
                'Date (TU)': 'date_validite'
            }
        )

        # only keep relevant columns
        df0 = df0.loc[:, ['date_validite', 'valeur']]

        # set dates as index
        df0 = df0.set_index('date_validite')

        # prepend level to row multi-index for sites
        df0 = pd.concat({entite: df0}, names=['entite'])

        # concatenate with other sites
        df1 = pd.concat([df1, df0])

    return df1
