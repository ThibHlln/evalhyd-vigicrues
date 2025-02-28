import os
import toml
from typing import List, Dict
import numpy as np
import pandas as pd
from numpy import dtype
from numpy.typing import NDArray
import evalhyd

from ..read import read_frc_from_xml_sandre, read_frc_from_prv
from ._convert import convert_frc_df_to_arr


_levels = toml.load(
    f'{os.path.abspath(os.path.dirname(__file__))}{os.sep}evald.toml'
)


def evald(
        q_obs: NDArray[dtype('float64')], prd_files: List[str],
        metrics: List[str], transform: str = None, exponent: float = None,
        q_thr: np.ndarray = None, events: str = None,
        epsilon: float = None, t_msk: NDArray[dtype('bool')] = None,
        m_cdt: NDArray[dtype('|S32')] = None, bootstrap: Dict[str, int] = None,
        dts: NDArray[dtype('|S32')] = None, seed: int = None,
        diagnostics: List[str] = None, member_agg_method: str = 'mean',
        prv_datatype: str = None, return_format: str = 'dataframes'
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

        prd_files: `List[str]`
            La liste de fichiers au format XML-SANDRE (extension *.xml)
            ou au format PRV (extension *.prv) contenant les prédictions
            de débits. Le format de fichier est déterminé à partir de
            l'extension du premier fichier dans la liste. Si les
            fichiers sont au format PRV, le paramètre *prv_datatype*
            doit être défini.
            dimensions : (entités, échéances, membres, temps)

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

        prv_datatype: `str`, optionnel
            Le type du fichier de données PRV parmi ``'otamin16_fcst'``,
            ``'otamin18_fcst'``, ``'scores_fcst'``. Ce paramètre est
            obligatoire si la liste de fichiers fournie pour
            *prd_files*, sinon ce paramètre ignoré.

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
    if prd_files[0][-4:] == ".xml":
        df_prd = read_frc_from_xml_sandre(prd_files)
    elif prd_files[0][-4:] == ".prv":
        if prv_datatype is None:
            raise RuntimeError(
                "'prv_datatype' doit être fourni quand les "
                "fichiers de prédictions sont au format PRV"
            )
        df_prd = read_frc_from_prv(prd_files, datatype=prv_datatype)
    else:
        raise ValueError(
            "les fichiers de prédictions doivent contenir "
            "l'extension *.xml ou *.prv"
        )
    arr_prd = convert_frc_df_to_arr(df_prd)

    # apply aggregation to ensemble members
    if member_agg_method == 'mean':
        arr_prd = np.mean(arr_prd, axis=2)
    elif member_agg_method == 'median':
        arr_prd = np.median(arr_prd, axis=2)

    # call evalhyd function (one site at a time)
    res = list()
    for s, site in enumerate(df_prd.index.levels[0]):
        res.append(
            evalhyd.evald(
                q_obs[[s], ...], arr_prd[s, ...], metrics,
                q_thr[[s], :].repeat(arr_prd.shape[1], 0), events,
                transform, exponent, epsilon, t_msk[s, ...],
                m_cdt[[s], ...].repeat(arr_prd.shape[1], 0),
                # TODO: drop requirement for dts and use input dataframes instead
                bootstrap, dts, seed,
                diagnostics
            )
        )

    # stack arrays in site order on new leading axis
    res_as_arr = [
        np.stack(arrays, axis=0)
        for arrays in zip(*res)
    ]

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

            for s, site in enumerate(df_prd.index.levels[0]):
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
                        else ['aucun'],
                    'seuils': [
                        f"{'≥' if events == 'high' else '≤'}{q}"
                        for q in q_thr[s]
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

            res_as_df[indicator] = df

        return res_as_df
