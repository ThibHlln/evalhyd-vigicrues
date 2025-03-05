import pandas as pd
from typing import List
from libhydro.conv.xml import xml_parser


def read_prd_from_xml_sandre(xml_files: List[str]) -> pd.DataFrame:
    """Lire les fichiers au format XML-SANDRE contenant les prédictions
    de débits et retourner sous forme de `pandas.DataFrame`.

    :Paramètres:

        xml_files: `list`
            La liste de fichiers au format XML-SANDRE contenant les
            prédictions de débits.

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
        d = xml_parser.parse_xml_file(xml_file)

        df1 = None

        for sim in d['simulations']:
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

            # collect forecast issue date
            issue_date = sim.dtderobs

            # rename result column
            df0 = df0.rename(columns={'res': 'valeur'})

            # create new column with validity dates
            df0.loc[:, 'echeances'] = (
                    df0.index.get_level_values('dates_validite') - issue_date
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
