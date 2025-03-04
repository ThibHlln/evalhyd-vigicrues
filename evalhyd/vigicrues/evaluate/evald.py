import os
import toml
from typing import List, Dict
import numpy as np
import pandas as pd
from numpy import dtype
from numpy.typing import NDArray
import evalhyd

from ._convert import convert_prd_df_to_arr


_levels = toml.load(
    f'{os.path.abspath(os.path.dirname(__file__))}{os.sep}evald.toml'
)


def evald(
        df_obs: pd.DataFrame, df_prd: pd.DataFrame,
        metrics: List[str], transform: str = None, exponent: float = None,
        q_thr: np.ndarray = None, events: str = None,
        epsilon: float = None, t_msk: NDArray[dtype('bool')] = None,
        m_cdt: NDArray[dtype('|S32')] = None, bootstrap: Dict[str, int] = None,
        seed: int = None, diagnostics: List[str] = None,
        member_agg_method: str = 'mean', return_format: str = 'dataframes'
) -> Dict[str, np.ndarray | pd.DataFrame]:
    """Fonction pour évaluer des predictions déterministes de débits.

    :Paramètres:

        df_obs: `pandas.DataFrame`
            La dataframe contenant les observations de débits. Elle doit
            posséder un index nommé 'date validité' (de type
            `pd.Timestamp`) et une colonne nommée 'valeur' (de type
            `float`) contenant des débits dans une unité identique à
            celle de *df_prd* et *q_thr*.
            dimensions : (temps,)

            *Exemple de paramètre :*

            .. code-block:: python

               import numpy as np
               import pandas as pd

               df_obs = pd.DataFrame(
                   data=np.random.randint(100, 400, 2),
                   index=pd.Index(
                       [pd.to_datetime('2001-08-07'), pd.to_datetime('2001-08-08')],
                       name='date validité'
                   ),
                   columns=pd.Index(['valeur'])
               )

        df_prd: `pandas.DataFrame`
            La dataframe contenant les prédictions de débits. Elle doit
            posséder un multi-index pour les lignes avec deux niveaux
            nommés 'échéances' et 'date validité' (respectivement de
            types `str` et `pd.Timestamp`) et une colonne nommée
            'valeur' (de type `float`) contenant des débits dans une
            unité identique à celle de *df_obs* et *q_thr*.
            dimensions : (échéances, temps)

            *Exemple de paramètre :*

            .. code-block:: python

               import numpy as np
               import pandas as pd

               df_prd = pd.DataFrame(
                   data=np.random.randint(100, 400, 2),
                   index=pd.MultiIndex.from_product(
                       [[pd.to_timedelta('1 day')],
                        [pd.to_datetime('2001-08-07'), pd.to_datetime('2001-08-08')]],
                       names=['échéances', 'date validité']
                   ),
                   columns=pd.Index(['valeur'])
               )

        metrics: `List[str]`
            La liste d'indicateurs d'évaluation à calculer.
            dimensions : (indicateurs,)

        q_thr: `numpy.ndarray` ``[dtype('float64')]``, optionnel
            Le vecteur contenant le(s) seuil(s) de débits à considérer
            pour les indicateurs évaluant les prédictions de dépassement
            de seuils.
            dimensions : (seuils,)

        events: `str`, optionnel
            Le type de dépassement de seuil à considérer pour les
            indicateurs basés sur des seuils de dépassement. Il peut
            être défini soit comme `"high"` pour l'évaluation
            d'événements de crues (c'est-à-dire quand le débit passe
            au-dessus du seuil) soit comme `"low"` pour l'évaluation
            d'événements d'étiages (c'est-à-dire quand le débit passe
            en-dessous du seuil). Il doit être fourni si *q_thr* est
            fourni.

        t_msk: `numpy.ndarray` ``[dtype('bool')]``, optionnel
            La matrice 2D contenant les masques permettant des générer
            des sous-ensembles des chroniques de débits (où `True`/
            `False` est utilisé pour inclure/exclure les pas de temps
            dans un sous-ensemble donné). Si la matrice n'est pas
            fournie et *m_cdt* n'est pas fournie non plus, aucun
            sous-ensemble n'est produit et un seul jeu d'indicateurs
            correspondant à la période entière est généré. Si la
            matrice est fournie, autant de jeux d'indicateurs que de
            masques fournis sont générés.
            dimensions : (sous-ensembles, temps)

        m_cdt: `numpy.ndarray` ``[dtype('|S32')]``, optionnel
            Le vecteur contenant les conditions permettant de
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
            dimensions : (sous-ensembles,)

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

        member_agg_method: `str`, optionnel
            La méthode d'agrégation à utiliser pour réduire les membres
            de l'ensemble à une valeur unique afin de pouvoir appliquer
            les indicateurs déterministes. Les possibilités sont la
            moyenne ``'mean'`` ou médiane ``'median'``. Si la méthode
            n'est pas fournie, la moyenne des membres sera utilisée.

        return_format: `str`, optionnel
            Le format désiré pour les indicateurs d'évaluation, soit
            ``'dataframes'`` pour obtenir des `pandas.DataFrame` ou
            ``'arrays'`` pour obtenir des `numpy.ndarray`. Si le format
            n'est pas fourni, des dataframes sont retournées.

    :Retourne:

        `dict` de `pandas.DataFrame` ou de `numpy.ndarray`
            Les valeurs des indicateurs d'évaluation (et des variables
            de diagnostic d'évaluation le cas échéant).

    """
    # check requested return format
    if return_format not in ('dataframes', 'arrays'):
        raise ValueError("return_format must be 'dataframes' or 'arrays'")

    # check aggregation method for ensemble members
    if member_agg_method not in ('mean', 'median'):
        raise ValueError("member_agg_method must be 'mean' or 'median'")

    # check coherence between temporal levels
    if not (df_prd.index.levels[1] == df_obs.index).all():
        raise ValueError(
            "dates de validité différentes entre les observations "
            "et les prévisions de débits"
        )
    else:
        dts = (
            df_obs.index.strftime('%Y-%m-%s %H:%M:%S').to_numpy()
        )

    # convert observation data
    arr_obs = df_obs.to_numpy().T

    # convert prediction data
    arr_prd = convert_prd_df_to_arr(df_prd)

    # call evalhyd function (one site at a time)
    res_as_arr = evalhyd.evald(
        arr_obs, arr_prd, metrics,
        q_thr[np.newaxis, :].repeat(arr_prd.shape[0], 0),
        events, transform, exponent, epsilon,
        t_msk[np.newaxis, ...].repeat(arr_prd.shape[0], 0),
        m_cdt[np.newaxis, :].repeat(arr_prd.shape[0], 0),
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
            # determine values to use for row multi-index levels
            level_values = {
                'échéances':
                    df_prd.index.levels[0],
                'sous-ensembles': (
                    np.arange(t_msk.shape[0]) + 1 if t_msk is not None
                    else m_cdt if m_cdt is not None
                    else 1
                ),
                'échantillons':
                    np.arange(bootstrap['n_samples']) + 1
                    if bootstrap is not None
                    else ['aucun'],
                'seuils': [
                    f"{'≥' if events == 'high' else '≤'}{q}"
                    for q in q_thr
                ],
                'composantes':
                    dict(
                        KGE_D=['r_pearson', 'alpha', 'beta'],
                        KGEPRIME_D=['r_pearson', 'gamma', 'beta'],
                        KGENP_D=['r_spearman', 'alpha_np', 'beta'],
                    ).get(indicator, None),
                'cellules':
                    ['a', 'b', 'c', 'd'],
            }

            # wrap results array in multi-index dataframe
            df = pd.DataFrame(
                data=res_as_arr[i].flatten(),
                index=pd.MultiIndex.from_product(
                    iterables=[level_values[lvl] for lvl in
                               _levels[indicator]],
                    names=_levels[indicator]
                )
            )

            res_as_df[indicator] = df

        return res_as_df
