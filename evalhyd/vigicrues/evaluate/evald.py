import os
import toml
from typing import List, Dict
import numpy as np
import pandas as pd
from numpy import dtype
from numpy.typing import NDArray
import evalhyd

from ..read import read_frc_from_xml_sandre
from ._convert import convert_frc_df_to_arr


_levels = toml.load(
    f'{os.path.abspath(os.path.dirname(__file__))}{os.sep}evald.toml'
)


def evald(
        q_obs: NDArray[dtype('float64')], xml_files_prd: List[str],
        metrics: List[str], transform: str = None, exponent: float = None,
        epsilon: float = None, t_msk: NDArray[dtype('bool')] = None,
        m_cdt: NDArray[dtype('|S32')] = None, bootstrap: Dict[str, int] = None,
        dts: NDArray[dtype('|S32')] = None, seed: int = None,
        diagnostics: List[str] = None, member_agg_method: str = 'mean',
        return_format: str = 'dataframes'
) -> Dict[str, np.ndarray | pd.DataFrame]:
    """Fonction pour évaluer des predictions déterministes de débits.

    :Paramètres:

        q_obs: `numpy.ndarray` ``[dtype('float64')]``
            La matrice 2D contenant les observations de débits. Les pas
            de temps sans observations doivent être assignés des
            valeurs `numpy.nan`. Ces pas de temps seront ignorés à la
            fois dans les observations et les prédictions avant que les
            indicateurs soient calculés.
            dimensions : (entités, temps)

        xml_files_prd: `list`
            La liste de fichiers au format XML-SANDRE contenant les
            prédictions de débits. Les pas de temps sans observations
            doivent être assignés des valeurs `numpy.nan`. Ces pas de
            temps seront ignorés à la fois dans les observations et les
            prédictions avant que les indicateurs soient calculés.
            dimensions : (entités, échéances, membres, temps)

        metrics: `List[str]`
            La liste d'indicateurs d'évaluation à calculer.
            dimensions : (indicateurs,)

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
             effectué. Si les valeurs sont fournies, *dts* doit
             également être fourni.

             *Exemple de paramètre :*

            .. code-block:: python

               bootstrap={"n_samples": 100, "len_sample": 10, "summary": 0}

        dts: `numpy.ndarray` ``[dtype('|S32')]``, optionnel
            Le vecteur de dates et heures correspondant à la dimension
            temporelle des observations et prédictions de débits. La
            date et l'heure doit être spécifiée suivant la norme
            ISO 8601-1:2019, c'est-à-dire "AAAA-MM-JJ hh:mm:ss" (par
            exemple, le 21 mai 2007 à 4 heures de l'après-midi s'écrit
            "2007-05-21 16:00:00"). Si le vecteur est fourni, il est
            seulement considéré si *bootstrap* est aussi fourni.
            dimensions : (temps,)

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

    # load prediction data
    df_prd = read_frc_from_xml_sandre(xml_files_prd)
    arr_prd = convert_frc_df_to_arr(df_prd)

    # apply aggregation to ensemble members
    if member_agg_method == 'mean':
        arr_prd = np.mean(arr_prd, axis=2)
    elif member_agg_method == 'median':
        arr_prd = np.median(arr_prd, axis=2)

    # call evalhyd function (one leadtime at a time)
    res_as_arr = None
    for l, leadtime in df_prd.index.levels[1]:
        res = evalhyd.evald(
            q_obs[:, l, ...], arr_prd[:, l, ...], metrics,
            transform, exponent, epsilon, t_msk[:, l, ...], m_cdt,
            # TODO: drop requirement for dts and use input dataframes instead
            bootstrap, dts, seed,
            diagnostics
        )

        # stack arrays in leadtime order on new intermediate axis
        if res_as_arr is not None:
            res_as_arr = [
                np.stack([r1, r2], axis=1) for r1, r2 in zip(res_as_arr, res)
            ]
        else:
            res_as_arr = res

    if return_format == 'arrays':
        # return arrays wrapped in a dictionary rather than a list
        return {
            indicator: res_as_arr[i]
            for i, indicator in enumerate(metrics + diagnostics)
        }
    else:  # 'dataframes'

        res_as_df = {}

        for i, indicator in enumerate(metrics + diagnostics):

            df = None

            for s, site in df_prd.index.levels[0]:
                # determine values to use for row multi-index levels
                level_values = {
                    'entités':
                        [site],
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
                        else ['tout'],
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
                df = pd.concat(
                    [
                        df,
                        pd.DataFrame(
                            data=res_as_arr[i][s].flatten(),
                            index=pd.MultiIndex.from_product(
                                iterables=[level_values[dim] for dim in
                                           _levels[indicator]],
                                names=_levels[indicator]
                            )
                        )
                    ]
                )

            res_as_df[indicator] = df

        return res_as_df
