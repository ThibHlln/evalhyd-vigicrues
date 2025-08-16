from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import re
import os

from ._utils import format_timedelta


def plot_rel_diag(
        rel_diag: pd.DataFrame,
        savefig_kwargs: dict = None, output_dir: str = '.'
) -> List[str]:
    """Générer des diagrammes de fiabilité à partir des données de sortie
    de la fonction `evalhyd.vigicrues.evalp`.

    .. note::

       Pour des données multi-entités et/ou multi-échéances et/ou
       multi-sous-ensembles, une figure par entité, par échéance, par
       sous-ensemble et par seuil est générée.

    .. note::

       Les noms de fichiers sont standardisés et détaillent leur
       contenu. Ainsi, les noms contiennent les entités suivies des
       échéances suivies des sous-ensembles suivis des seuils.

    .. warning::

       Les données de sortie comportant un échantillonnage par bootstrap
       ne sont pas supportées par cette fonction.

    :Paramètres:

        rel_diag: `pandas.DataFrame`
            La dataframe produite par `evalhyd.vigicrues.evalp`
            correspondant à l'indicateur REL_DIAG.

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
    if rel_diag.index.names != [
        'entite', 'echeance', 'sous_ensemble', 'echantillon',
        'seuil', 'classe', 'axe'
    ]:
        raise RuntimeError(
            "'rel_diag' ne semble pas être une "
            "dataframe de diagrammes de fiabilité"
        )

    # retrieve multi-index level values
    level_values = {
        level: rel_diag.index.unique(level)
        for level in rel_diag.index.names
    }
    level_values[None] = [slice(None)]

    # check that REL_DIAG was not computed with bootstrapping
    if len(level_values['echantillon']) > 1:
        raise ValueError(
            "visualisation non autorisée pour des résultats issus "
            "d'un échantillonnage par bootstrap"
        )

    # loop through sites, leadtimes, subsets, and thresholds
    filepaths = list()

    for site in level_values['entite']:
        for leadtime in level_values['echeance']:
            for s, subset in enumerate(level_values['sous_ensemble']):
                for threshold in (
                        rel_diag.loc[(site, leadtime, subset)]
                                .index.unique('seuil')
                ):
                    # create figure and grid spec
                    fig = plt.figure(
                        figsize=(8, 8)
                    )
                    gs = mpl.gridspec.GridSpec(
                        100, 100, figure=fig
                    )

                    # slice dataframe to focus on content in a single figure
                    df = rel_diag.sort_index().loc[
                         (site, leadtime, subset, 'aucun', threshold), :
                    ].unstack('axe').droplevel(0, axis=1)

                    # plot reliability diagram over entire grid spec
                    ax0 = fig.add_subplot(gs[:, :])
                    ax0.plot(
                        df.loc[:, 'x'], df.loc[:, 'y'], 'o-',
                        clip_on=False, zorder=2
                    )

                    # add axis labels
                    ax0.set_xlabel("Probabilités prédictives [-]")
                    ax0.set_ylabel("Fréquences observées [-]")

                    # set axis ticks and limits
                    ax0.set_xticks([0, 1])
                    ax0.set_xlim([0, 1])
                    ax0.set_yticks([0, 1])
                    ax0.set_ylim([0, 1])

                    # add a 1:1 line
                    ax0.axline(
                        (0, 0), slope=1, color="black", zorder=1,
                        linewidth=mpl.rcParams['axes.linewidth']
                    )

                    # plot corresponding histogram at the bottom left
                    ax1 = fig.add_subplot(gs[85:98, 85:98])
                    ax1.bar(
                        np.arange(df.shape[0]), df.loc[:, 'ordinates'],
                        color="tab:blue", width=1.0
                    )

                    # turn off axis ticks for histogram
                    ax1.set_xticks([])
                    ax1.set_yticks([])

                    # update savefig parameters if provided
                    kwargs = dict(
                        format='png', dpi=300
                    )
                    if savefig_kwargs:
                        kwargs.update(savefig_kwargs)

                    # replace comparison operators in thresholds by text
                    threshold = threshold.replace('≥', 'SUP')
                    threshold = threshold.replace('≤', 'INF')

                    # standardise file name
                    formatted_leadtime = (
                        format_timedelta(leadtime)
                    )

                    filename = (
                        f"{site}+{formatted_leadtime}+{s}+{threshold}"
                    )

                    # get rid of problematic characters in filename
                    # (https://stackoverflow.com/a/71199182)
                    filename = re.sub(
                        r"[/\\?%*:|\"<>\x7F\x00-\x1F]", "_", filename
                    )

                    filepath = (
                        f"{output_dir}{os.sep}{filename}.{kwargs['format']}"
                    )

                    # save figure
                    fig.savefig(filepath, **kwargs)

                    # store path to image file
                    filepaths.append(os.path.abspath(filepath))

                    plt.close(fig)

    return filepaths
