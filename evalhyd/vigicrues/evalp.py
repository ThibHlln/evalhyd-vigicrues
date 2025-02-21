import os
import toml
import numpy as np
import pandas as pd
from typing import List, Dict
from evalhyd import evalp as evalhyd_evalp

from .read import read_frc_from_xml_sandre, convert_frc_df_to_arr


_levels = toml.load(
    f'{os.path.abspath(os.path.dirname(__file__))}{os.sep}evalp.toml'
)


def evalp(
        q_obs: np.ndarray, xml_files_prd: List[str], metrics: List[str],
        q_thr: np.ndarray = None, events: str = None, c_lvl: np.ndarray = None,
        t_msk: np.ndarray = None, m_cdt: np.ndarray = None,
        bootstrap: Dict[str, int] = None, dts: np.ndarray = None,
        seed: int = None, diagnostics: List[str] = None,
        return_format: str = 'dataframes'
) -> Dict[str, np.ndarray | pd.DataFrame]:
    """Fonction pour évaluer des predictions probabilistes de débits.

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
             effectué. Si les valeurs sont fournies, *dts* doit
             également être fourni.

             *Parameter example:*

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

    # load prediction data
    df_prd = read_frc_from_xml_sandre(
        xml_files_prd, return_format='dataframe'
    )
    arr_prd = convert_frc_df_to_arr(df_prd)

    # call evalhyd
    res_as_arr = evalhyd_evalp(
        q_obs, arr_prd, metrics,
        q_thr, events, c_lvl, t_msk, m_cdt,
        # TODO: drop requirement for dts and use input dataframes instead
        bootstrap, dts, seed,  
        diagnostics
    )

    if return_format == 'arrays':
        return {
            metric: res_as_arr[m] for m, metric in enumerate(metrics)
        }
    else:  # 'dataframes'

        res_as_df = {}

        for m, metric in enumerate(metrics + diagnostics):

            n_mbr = arr_prd.shape[2]

            df = None

            for s, site in df_prd.index.levels[0]:
                # determine values to use for row multi-index levels
                level_values = {
                    'entités':
                        df_prd.index.levels[0][[s]],
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
                        else ['tout'],
                    'seuils': [
                        f"{'≥' if events == 'high' else '≤'}{q}"
                        for q in q_thr[s]
                    ],
                    'components':
                        dict(
                            BS_CRD=['fiabilité', 'finesse', 'incertitude'],
                            BS_LBD=['biais', 'discrimination', 'finesse'],
                        ).get(metric, None),
                    'axes':
                        dict(
                            REL_DIAG=['x', 'y', 'ordinates'],
                        ).get(metric, None),
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
                            data=res_as_arr[m][s].flatten(),
                            index=pd.MultiIndex.from_product(
                                iterables=[level_values[dim] for dim in
                                           _levels[metric]],
                                names=_levels[metric]
                            )
                        )
                    ]
                )

                # special case for multi-sites metrics
                if 'toutes entités' in _levels[metric]:
                    df.index = df.index.set_names(
                        'entités', level='toutes entités'
                    )
                    break

            res_as_df[metric] = df

        return res_as_df
