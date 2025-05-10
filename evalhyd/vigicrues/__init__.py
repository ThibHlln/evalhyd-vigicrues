"""A Python add-on to evalhyd providing pre- and post-processing functionalities specific to VigiCrues forecasts."""

from .version import __version__

from .read import (
    read_prd_from_xml_sandre, read_obs_from_xml_sandre,
    read_prd_from_prv,
    read_obs_from_csv_hydroportail
)
from .evaluate import evald
from .evaluate import evalp
from .plot import plot_rank_hist, plot_rel_diag
