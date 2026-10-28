"""elessar: automated exploratory data analysis and data-quality profiling for pandas DataFrames.

>>> import elessar
>>> report = elessar.see(df, target="price")        # prints the report, writes plots + report.html
>>> report = elessar.see("sales.parquet", target="ask")  # any file; pick the target in a window
"""

__version__ = "0.2.0"

from .config import StoneConfig  # noqa: E402 - __version__ first: submodules read it
from .core import see, see_column  # noqa: E402
from .correlations import see_corr  # noqa: E402
from .distributions import see_dist  # noqa: E402
from .picker import pick_target, suggest_targets  # noqa: E402
from .plots import see_plot  # noqa: E402
from .quality import see_quality  # noqa: E402
from .report import Vision  # noqa: E402
from .sources import LoadError, Source, load  # noqa: E402
from .stats import see_stats  # noqa: E402
from .types import see_types  # noqa: E402

__all__ = [
    "LoadError",
    "Source",
    "StoneConfig",
    "Vision",
    "load",
    "pick_target",
    "see",
    "see_column",
    "see_corr",
    "see_dist",
    "see_plot",
    "see_quality",
    "see_stats",
    "see_types",
    "suggest_targets",
]
