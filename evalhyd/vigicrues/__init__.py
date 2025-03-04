"""A Python add-on to evalhyd providing pre- and post-processing functionalities specific to VigiCrues forecasts."""

from .version import __version__

from .read import read_prd_from_xml_sandre, read_prd_from_prv
from .evaluate import evald
from .evaluate import evalp
from .plot import plot_rank_hist
