import pandas as pd
import numpy as np
from typing import List
from libhydro.conv.xml import Message


def read_prd_from_xml_sandre(
        xml_files: List[str], seek_issue_date: bool = True
) -> pd.DataFrame:
    """Lire les fichiers au format XML-SANDRE contenant les prédictions
    de débits et retourner sous forme de `pandas.DataFrame`.

    :Paramètres:

        xml_files: `list`
            La liste de fichiers au format XML-SANDRE contenant les
            prédictions de débits.

        seek_issue_date: `bool`, optional
            Choix de rechercher ou non les dates d'émission dans les
            fichiers pour en déduire les échéances de prévision. Si
            le choix est fait de ne pas les rechercher, les échéances de
            prévision seront des rangs au lieu d'être des durées. Ceci
            implique que les échéances entre les entités sont supposées
            être les mêmes puisque, après l'assignation des rangs en
            lieu et place des durées, elles seront identifiées par les
            mêmes intitulés. Par défaut, les dates d'émission sont
            recherchées et une erreur est générée si elles sont absentes.

    :Retourne:

        `pandas.DataFrame`
            La structure de données contenant les prédictions de débits.

    **Exemples**

    Récupérer les prédictions de débits sous forme de dataframe :

    >>> df = read_prd_from_xml_sandre(['data/GRP_B_20241211_1023_5304.xml'])
    >>> df.xs('K0045510', level='entites', drop_level=False).xs('0001', level='membres', drop_level=False)
                                                            valeur
    entites  echeances       membres dates_validite
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
    """
    # loop through XML files
    df2 = None

    for xml_file in xml_files:
        # read in XML Sandre file with `libhydro`
        d = Message.from_file(xml_file)

        if not d.simulations:
            raise RuntimeError(
                f"Le fichier {xml_file} ne contient "
                f"pas de series de simulations"
            )

        df1 = None

        for sim in d.simulations:
            # extract predictions dataframe
            if sim.prevs_ensemble is not None:
                # extract ensemble predictions
                df0 = sim.prevs_ensemble

                # rename multi-index levels
                df0.index = df0.index.rename(
                    {
                        'lb': 'membres',
                        'dte': 'dates_validite'
                    }
                )
            elif sim.previsions_tend is not None:
                # extract trend predictions
                df0 = sim.previsions_tend

                # rename multi-index levels
                df0.index = df0.index.rename(
                    {
                        'tend': 'tendances',
                        'dte': 'dates_validite'
                    }
                )
            else:
                raise RuntimeError(
                    f"Le fichier {xml_file} ne contient pas de "
                    f"prévisions ensemblistes ou de tendances"
                )

            # select result column (drop other columns)
            df0 = df0.loc[:, ['res']]

            # rename result column
            df0 = df0.rename(columns={'res': 'valeur'})

            if seek_issue_date:
                # collect forecast issue date
                if sim.dtbase is not None:
                    issue_date = sim.dtbase
                elif sim.dtderobs is not None:
                    issue_date = sim.dtderobs
                else:
                    raise RuntimeError(
                        f"Le fichier {xml_file} ne contient pas "
                        f"de date d'émission de la prévision"
                    )

                # create new column with validity dates
                df0.loc[:, 'echeances'] = (
                    df0.index.get_level_values('dates_validite') - issue_date
                )
            else:
                # create new column with ranks as lead times
                shape = tuple(map(len, df0.index.levels))
                d = 0 if df0.index.names[0] == 'dates_validite' else 1
                m = 1 if d == 0 else 1

                df0.loc[:, 'echeances'] = (
                    np.arange(shape[d])[:, np.newaxis]
                    .repeat(shape[m], axis=1).flatten()
                )

            # append column as additional level in row multi-index
            df0 = df0.set_index('echeances', append=True)

            # prepend level to row multi-index for sites
            df0 = pd.concat({sim.entite.code: df0}, names=['entites'])

            # reorder levels in row multi-index to match evalhyd convention
            df0.index = df0.index.reorder_levels(
                [
                    'entites',
                    'echeances',
                    'membres' if sim.prevs_ensemble is not None
                    else 'tendances',
                    'dates_validite'
                ]
            )

            # concatenate with other sites
            df1 = pd.concat([df1, df0])

        df2 = pd.concat([df2, df1])

    return df2


def read_obs_from_xml_sandre(xml_files: List[str]) -> pd.DataFrame:
    """Lire les fichiers au format XML-SANDRE contenant les observations
    de débits et retourner sous forme de `pandas.DataFrame`.

    :Paramètres:

        xml_files: `list`
            La liste de fichiers au format XML-SANDRE contenant les
            observations de débits.

    :Retourne:

        `pandas.DataFrame`
            La structure de données contenant les observations de débits.

    **Exemples**

    Récupérer les observations de débits sous forme de dataframe :

    >>> df = read_obs_from_xml_sandre(['data/export_hydro_series.xml'])
    >>> df
                                 valeur
    entites    dates_validite
    H5201010   2010-01-01      165549.0
               2010-01-02      183860.0
               2010-01-03      186781.0
               2010-01-04      165038.0
               2010-01-05      130174.0
    ...                             ...
    H507101002 2019-12-28      111972.0
               2019-12-29      112633.0
               2019-12-30       97809.0
               2019-12-31       92378.0
               2020-01-01       72622.0

    [18039 rows x 1 columns]
    """
    # loop through XML files
    df2 = None

    for xml_file in xml_files:
        # read in XML Sandre file with `libhydro`
        d = Message.from_file(xml_file)

        if not d.seriesobselab:
            raise RuntimeError(
                f"Le fichier {xml_file} ne contient "
                f"pas de series d'observations"
            )

        df1 = None

        for obs in d.seriesobselab:
            # extract observations dataframe
            if obs.observations is not None:
                df0 = obs.observations
            else:
                raise RuntimeError(
                    f"Le fichier {xml_file} ne contient pas d'observations"
                )

            # rename multi-index levels
            df0.index.name = 'dates_validite'

            # select result column (drop other columns)
            df0 = df0.loc[:, ['res']]

            # rename result column
            df0 = df0.rename(columns={'res': 'valeur'})

            # prepend level to row multi-index for sites
            df0 = pd.concat({obs.entite.code: df0}, names=['entites'])

            # concatenate with other sites
            df1 = pd.concat([df1, df0])

        df2 = pd.concat([df2, df1])

    return df2
