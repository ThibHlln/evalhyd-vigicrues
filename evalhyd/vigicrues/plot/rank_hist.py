import os
import re
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from typing import Tuple, List

from ._utils import format_timedelta


def plot_rank_hist(
        rank_hist: pd.DataFrame, row: str = None, col: str = None,
        figsize: Tuple[float | int, float | int] = None,
        savefig_kwargs: dict = None, output_dir: str = '.'
) -> List[str]:
    """Générer des diagrammes de rangs à partir des données de sortie
    de la fonction `evalhyd.vigicrues.evalp`.

    .. note::

       Pour des données multi-entités et/ou multi-échéances et/ou
       multi-sous-ensembles, une figure par entité, par échéance et par
       sous-ensemble est générée par défaut. Les paramètres *row* et
       *col* peuvent être utilisés pour afficher toutes les entités
       et/ou toutes les échéances et/ou tous les sous-ensembles sur une
       même figure.

    .. note::

       Les noms de fichiers sont standardisés et détaillent leur
       contenu. Ainsi, les noms contiennent les entités suivies des
       échéances suivies des sous-ensembles. Si les paramètres *row*
       et/ou *col* sont utilisés, les noms de fichiers sont modifiés
       en conséquence et contiennent toutes-entités et/ou
       toutes-échéances et/ou tous-sous-ensembles selon les cas.

    .. warning::

       Les données de sortie comportant un échantillonnage par bootstrap
       ne sont pas supportées par cette fonction.

    :Paramètres:

        rank_hist: `pandas.DataFrame`
            La dataframe produite par `evalhyd.vigicrues.evalp`
            correspondant à l'indicateur RANK_HIST.

        row: `str`, optionnel
            À utiliser pour produire des figures contenant plusieurs
            diagrammes de rangs côte-à-côte. La dimension fournie
            via ce paramètre sera utilisée pour former une ligne
            de diagrammes de rangs. Utilisé en combinaison avec
            le paramètre *col* produira une matrice 2D de diagrammes
            de rangs.

        col: `str`, optionnel
            À utiliser pour produire des figures contenant plusieurs
            diagrammes de rangs côte-à-côte. La dimension fournie
            via ce paramètre sera utilisée pour former une colonne
            de diagrammes de rangs. Utilisé en combinaison avec
            le paramètre *row* produira une matrice 2D de diagrammes
            de rangs.

        figsize: `tuple`, optionnel
            La largeur et la hauteur en pouces des figures produites.

        savefig_kwargs: `str`, optionnel
            Les arguments à passer à `matplotlib.figure.Figure.savefig`
            pour personnaliser les figures produites. Parmi les
            paramètres possibles, ceux définis par cette fonction pour
            défaut sont ``format='png'`` et ``dpi=300``. Ils seront
            écrasés par ceux passés via ce paramètre le cas échéant.

        output_dir: `str`, optionnel
            Le chemin absolu ou relatif vers le répertoire où les
            images seront sauvegardées. Si ce paramètre n'est pas
            fourni, le répertoire de travail courant est utilisé.

    :Retourne:

        `List[str]`
            L'ensemble des chemins des fichiers produits par la fonction.

    """
    # check levels of multi-index
    if rank_hist.index.names != [
            'entite', 'echeance', 'sous_ensemble', 'echantillon', 'rang'
    ]:
        raise RuntimeError(
            "'rank_hist' ne semble pas être une "
            "dataframe de diagrammes de rangs"
        )

    # check validity of the X,Y axes
    lvl_axes = (row if row else None, col if col else None)
    for axis in (row, col):
        if axis not in ('entite', 'echeance', 'sous_ensemble', None):
            raise ValueError(
                "les axes x et y ne peuvent être que 'entite' "
                "ou 'echeance' ou 'sous_ensemble'"
            )

    lvl_left = (
        {'entite', 'echeance', 'sous_ensemble'}.difference(lvl_axes)
    )

    # retrieve multi-index level values
    level_values = {
        level: rank_hist.index.unique(level)
        for level in rank_hist.index.names
    }
    level_values[None] = [slice(None)]

    # check that RANK_HIST was not computed with bootstrapping
    if len(level_values['echantillon']) > 1:
        raise ValueError(
            "visualisation non autorisée pour des résultats issus "
            "d'un échantillonnage par bootstrap"
        )

    # determine values of potential levels to loop through
    sites = (
        level_values['entite'] if 'entite' in lvl_left
        else [slice(None)]
    )
    leadtimes = (
        level_values['echeance'] if 'echeance' in lvl_left
        else [slice(None)]
    )
    subsets = (
        level_values['sous_ensemble'] if 'sous_ensemble' in lvl_left
        else [slice(None)]
    )

    # loop through levels
    filepaths = list()

    for site in sites:
        for leadtime in leadtimes:
            for s, subset in enumerate(subsets, start=1):
                rows = level_values[row]
                cols = level_values[col]

                # create figure and grid spec
                w_in = 8.
                h_in = 6.
                width = w_in * len(cols)
                height = h_in * len(rows)
                if width > height:
                    scale = w_in / width
                else:
                    scale = h_in / height

                fig = plt.figure(
                    figsize=(
                        figsize if figsize
                        else (width * scale + 1, height * scale + 1)
                    ),
                    layout='compressed'
                )
                gs = mpl.gridspec.GridSpec(
                    len(rows), len(cols), figure=fig
                )

                # slice dataframe to focus on content in a single figure
                df = rank_hist.loc[(site, leadtime, subset, 'aucun'), :]

                # plot histograms on separate axes
                for r, row_ in enumerate(rows):
                    for c, col_ in enumerate(cols):
                        # further slice dataframe to focus on content
                        # in a single axis
                        df_ = df
                        if row:
                            df_ = df_.xs(row_, level=row)
                        if col:
                            df_ = df_.xs(col_, level=col)

                        # plot histogram
                        ranks = np.arange(len(df_)) + 1

                        ax = fig.add_subplot(gs[r, c])
                        ax.bar(
                            ranks, df_.to_numpy().squeeze(),
                            color="tab:blue", width=1.0
                        )

                        ax.set_xticks(ranks[[0, -1]], labels=[])
                        ax.set_yticks([])

                        if row and c == 0:
                            ax.set_ylabel(
                                f"+{format_timedelta(row_)}"
                                if row == 'echeance' else row_
                            )
                        if r == len(rows) - 1:
                            ax.set_xticklabels(
                                ranks[[0, -1]].astype(int).astype(str)
                            )
                            if col:
                                ax.set_xlabel(
                                    f"+{format_timedelta(col_)}"
                                    if col == 'echeance' else col_
                                )

                if row:
                    fig.supylabel(row)
                if col:
                    fig.supxlabel(col)

                formatted_leadtime = (
                    'toutes-echeances'
                    if leadtime == slice(None)
                    else format_timedelta(leadtime)
                    if isinstance(leadtime, pd.Timedelta)
                    else leadtime
                )

                fig.suptitle(
                    ', '.join(
                        filter(
                            None,
                            [
                                f"entite : {site}" if site != slice(None)
                                else None,
                                f"echeance : {formatted_leadtime}"
                                if formatted_leadtime != 'toutes-echeances'
                                else None,
                                f"sous-ensemble : {s}" if subset != slice(None)
                                else None
                            ]
                        )
                    )
                )

                # save figure with custom file name
                kwargs = dict(
                    format='png', dpi=300
                )
                if savefig_kwargs:
                    kwargs.update(savefig_kwargs)

                # save figure with custom file name
                filename = (
                    "rankhist"
                    f"+{site if site != slice(None) else 'toutes-entites'}"
                    f"+{formatted_leadtime}"
                    f"+{s if subset != slice(None) else 'tous-sous-ensembles'}"
                )

                # clean up filename from problematic characters
                # (https://stackoverflow.com/a/71199182)
                filename = re.sub(
                    r"[/\\?%*:|\"<>\x7F\x00-\x1F]", "_", filename
                )

                filepath = f"{output_dir}{os.sep}{filename}.{kwargs['format']}"

                fig.savefig(filepath, **kwargs)

                filepaths.append(os.path.abspath(filepath))

                plt.close(fig)

    return filepaths
