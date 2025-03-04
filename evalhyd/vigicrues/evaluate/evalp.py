import os
import toml
import numpy as np
import pandas as pd
from typing import List, Dict
import evalhyd

from ._convert import convert_obs_df_to_arr, convert_prd_df_to_arr


_levels = toml.load(
    f'{os.path.abspath(os.path.dirname(__file__))}{os.sep}evalp.toml'
)


def evalp(
        df_obs: pd.DataFrame, df_prd: pd.DataFrame, metrics: List[str],
        q_thr: np.ndarray = None, events: str = None, c_lvl: np.ndarray = None,
        t_msk: np.ndarray = None, m_cdt: np.ndarray = None,
        bootstrap: Dict[str, int] = None, seed: int = None,
        diagnostics: List[str] = None,
        return_format: str = 'dataframes'
) -> Dict[str, np.ndarray | pd.DataFrame]:
    """Fonction pour évaluer des predictions probabilistes de débits.

    :Paramètres:

        df_obs: `pandas.DataFrame`
            La dataframe contenant les observations de débits. Elle doit
            posséder un multi-index en lignes avec quatre niveaux nommés
            'entités' et 'date validité' (respectivement de types `str`
            et `pd.Timestamp`) et une colonne nommée 'valeur' (de type
            `float`) contenant des débits dans une unité identique à
            celle de *df_prd* et *q_thr*.
            dimensions : (entités, temps)

            *Exemple de paramètre :*

            .. code-block:: python

               import numpy as np
               import pandas as pd

               df_obs = pd.DataFrame(
                   data=np.random.randint(100, 400, 6),
                   index=pd.MultiIndex.from_product(
                       [['entité 1', 'entité 2', 'entité 3'],
                        [pd.to_datetime('2001-08-07'), pd.to_datetime('2001-08-08')]],
                       names=['entités', 'date validité']
                   ),
                   columns=pd.Index(['valeur'])
               )

        df_prd: `pandas.DataFrame`
            La dataframe contenant les prédictions de débits. Elle doit
            posséder un multi-index pour les lignes avec quatre niveaux
            nommés 'entités', 'échéances', 'membres' et 'date validité'
            (respectivement de types `str`, `pd.Timedelta`, `str` et
            `pd.Timestamp`) et une colonne nommée 'valeur' (de type
            `float`) contenant des débits dans une unité identique à
            celle de *df_obs* et *q_thr*.
            dimensions : (entités, échéances, membres, temps)

            *Exemple de paramètre :*

            .. code-block:: python

               import numpy as np
               import pandas as pd

               df_prd = pd.DataFrame(
                   data=np.random.randint(100, 400, 24),
                   index=pd.MultiIndex.from_product(
                       [['entité 1', 'entité 2', 'entité 3'],
                        [pd.to_timedelta('1 day')],
                        ['a', 'b', 'c', 'd'],
                        [pd.to_datetime('2001-08-07'), pd.to_datetime('2001-08-08')]],
                       names=['entités', 'échéances', 'membres', 'date validité']
                   ),
                   columns=pd.Index(['valeur'])
               )

        metrics: `List[str]`
            La liste d'indicateurs d'évaluation à calculer.
            dimensions : (indicateurs,)

        q_thr: `numpy.ndarray` ``[dtype('float64')]``, optionnel
            La matrice 2D contenant le(s) seuil(s) de débits à
            considérer pour les indicateurs évaluant les prédictions
            de dépassement de seuils. Si le nombre de seuils diffère
            entre les entités, `numpy.nan` peut être utilisé comme seuil
            pour les entités ayant moins de seuils que les autres.
            dimensions : (entités, seuils)

        events: `str`, optionnel
            Le type de dépassement de seuil à considérer pour les
            indicateurs basés sur des seuils de dépassement. Il peut
            être défini soit comme `"high"` pour l'évaluation
            d'événements de crues (c'est-à-dire quand le débit passe
            au-dessus du seuil) soit comme `"low"` pour l'évaluation
            d'événements d'étiages (c'est-à-dire quand le débit passe
            en-dessous du seuil). Il doit être fourni si *q_thr* est
            fourni.

        c_lvl: `numpy.ndarray` ``[dtype('float64')]``, optionnel
            Le vecteur d'intervalle(s) de confiance en pourcents à
            considérer pour les indicateurs basés sur des intervalles.
            dimensions : (intervalles,)

        t_msk: `numpy.ndarray` ``[dtype('bool')]``, optionnel
            La matrice 4D contenant les masques permettant des générer
            des sous-ensembles des chroniques de débits (où `True`/
            `False` est utilisé pour inclure/exclure les pas de temps
            dans un sous-ensemble donné). Si la matrice n'est pas
            fournie et *m_cdt* n'est pas fournie non plus, aucun
            sous-ensemble n'est produit et un seul jeu d'indicateurs
            correspondant à la période entière est généré. Si la
            matrice est fournie, autant de jeux d'indicateurs que de
            masques fournis sont générés.
            dimensions : (entités, échéances, sous-ensembles, temps)

        m_cdt: `numpy.ndarray` ``[dtype('|S32')]``, optionnel
            La matrice 2D contenant les conditions permettant de
            générer des sous-ensembles temporels. Chaque condition
            consiste en une chaîne de caractères et elle peut être
            spécifiée sur des valeurs débits observés ou prédits
            (moyenne, médiane, quantile) ou sur des indices temporels.
            Si la matrice est fournie et *t_msk* est également fourni,
            ce dernier est prioritaire et la matrice sera ignorée.
            Si la matrice n'est pas fournie et *t_msk* n'est pas fourni
            non plus, aucun sous-ensemble n'est produit et un seul jeu
            d'indicateurs correspondant à la période entière est généré.
            Si la matrice est fournie seule, autant de jeux
            d'indicateurs que de conditions fournies sont générés.
            dimensions : (entités, sous-ensembles)

        bootstrap: `dict`, optionnel
            Les valeurs des paramètres pour la méthode de bootstrap
            utilisée pour estimer l'incertitude d'échantillonnage dans
            l'évaluation des prédictions. Les trois paramètres sont :
            `"n_samples"` le nombre d'échantillons aléatoires,
            `"len_samples"` la longueur d'un échantillon en nombre
             d'années et `"summary"` les statistiques à calculer pour
             caractériser la distribution d'échantillonnage. Si les
             valeurs ne sont pas fournies, aucun bootstrap n'est
             effectué.

             *Exemple de paramètre :*

            .. code-block:: python

               bootstrap={"n_samples": 100, "len_sample": 10, "summary": 0}

        seed: `int`, optionnel
            Un nombre entier pour la graine utilisée par le générateur
            pseudo-aléatoire. Ce paramètre garanti la reproductibilité
            des valeurs d'indicateurs entre appels à la fonction.

        diagnostics: `List[str]`, optionnel
            La liste de variables de diagnostic de l'évaluation à
            calculer.
            dimensions : (variables,)

        return_format: `str`, optionnel
            Le format désiré pour les indicateurs d'évaluation, soit
            ``'dataframes'`` pour obtenir des `pandas.DataFrame` ou
            ``'arrays'`` pour obtenir des `numpy.ndarray`. Si le format
            n'est pas fourni, des dataframes sont retournées.

    :Retourne:

        `dict` de `pandas.DataFrame` ou de `numpy.ndarray`
            Les valeurs des indicateurs d'évaluation (et des variables
            de diagnostic d'évaluation le cas échéant).

            .. note::

               Pour les indicateurs avec dimensions tels que le CRPS ou
               le QS, l'unité de l'indicateur est identique à l'unité
               des données fournies à *df_obs*, *df_prd* et *q_thr*.

    """
    # check requested return format
    if return_format not in ('dataframes', 'arrays'):
        raise ValueError(
            "'return_format' must be 'dataframes' or 'arrays'"
        )

    # check coherence between temporal levels
    if not (df_prd.index.levels[3] == df_obs.index.levels[1]).all():
        raise ValueError(
            "dates de validité différentes entre les observations "
            "et les prévisions de débits"
        )
    else:
        dts = (
            df_prd.index.levels[3].strftime('%Y-%m-%s %H:%M:%S').to_numpy()
        )

    # convert observation data
    arr_obs = convert_obs_df_to_arr(df_obs)

    # convert prediction data
    arr_prd = convert_prd_df_to_arr(df_prd)

    # call evalhyd function
    res_as_arr = evalhyd.evalp(
        arr_obs, arr_prd, metrics,
        q_thr, events, c_lvl, t_msk, m_cdt,
        bootstrap, dts, seed,
        diagnostics
    )

    if return_format == 'arrays':
        # return arrays wrapped in a dictionary rather than a list
        return {
            indicator: res_as_arr[i]
            for i, indicator in enumerate(metrics + diagnostics)
        }
    else:  # 'dataframes'

        res_as_df = {}

        for i, indicator in enumerate(metrics + diagnostics):

            n_mbr = arr_prd.shape[2]

            df = None

            for s, site in enumerate(df_prd.index.levels[0]):
                # determine values to use for row multi-index levels
                level_values = {
                    'entités':
                        [site],
                    'toutes entités':
                        ['toutes'],
                    'échéances':
                        df_prd.index.levels[1],
                    'sous-ensembles': (
                        np.arange(t_msk.shape[2]) + 1 if t_msk is not None
                        else m_cdt[s] if m_cdt is not None
                        else 1
                    ),
                    'échantillons':
                        np.arange(bootstrap['n_samples']) + 1
                        if bootstrap is not None
                        else ['aucun'],
                    'seuils': [
                        f"{'≥' if events == 'high' else '≤'}{q}"
                        for q in q_thr[s]
                    ],
                    'composantes':
                        dict(
                            BS_CRD=['fiabilité', 'finesse', 'incertitude'],
                            BS_LBD=['biais', 'discrimination', 'finesse'],
                        ).get(indicator, None),
                    'axes':
                        dict(
                            REL_DIAG=['x', 'y', 'ordinates'],
                        ).get(indicator, None),
                    'niveaux':
                        np.arange(n_mbr + 1),
                    'classes':
                        np.arange(n_mbr + 1) / n_mbr,
                    'quantiles':
                        [
                            f'{q:.3f}'
                            for q in (np.arange(n_mbr) + 1) / (n_mbr + 1.)
                        ],
                    'cellules':
                        ['a', 'b', 'c', 'd'],
                    'rangs':
                        np.arange(n_mbr + 1) + 1,
                    'intervalles':
                        [f'{c}%' for c in c_lvl]
                        if c_lvl is not None else None,
                }

                # wrap results array in multi-index dataframe
                df = pd.concat(
                    [
                        df,
                        pd.DataFrame(
                            data=res_as_arr[i][s].flatten(),
                            index=pd.MultiIndex.from_product(
                                iterables=[level_values[lvl] for lvl in
                                           _levels[indicator]],
                                names=_levels[indicator]
                            )
                        )
                    ]
                )

                # special case for multi-sites metrics
                if 'toutes entités' in _levels[indicator]:
                    # rename index level name
                    df.index = df.index.set_names(
                        'entités', level='toutes entités'
                    )
                    # leave sites loop as there is only one item
                    # for multi-sites metrics
                    break

            res_as_df[indicator] = df

        return res_as_df
