"""elessar, the Elfstone for pandas: see everything about a DataFrame, and how to heal it.

>>> import elessar
>>> report = elessar.see(df, target="price")   # prints the report, writes plots
"""

from .config import StoneConfig
from .core import see, see_column
from .correlations import see_corr
from .distributions import see_dist
from .plots import see_plot
from .quality import see_quality
from .report import Vision
from .stats import see_stats
from .types import see_types

__version__ = "0.1.0"

__all__ = [
    "StoneConfig",
    "Vision",
    "see",
    "see_column",
    "see_corr",
    "see_dist",
    "see_plot",
    "see_quality",
    "see_stats",
    "see_types",
]
