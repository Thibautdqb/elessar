"""Configuration: every threshold used by elessar lives here."""

from __future__ import annotations

from dataclasses import dataclass

CONTINUOUS_CANDIDATES: tuple[str, ...] = (
    "norm",
    "lognorm",
    "expon",
    "gamma",
    "weibull_min",
    "beta",
    "uniform",
    "logistic",
    "t",
    "laplace",
    "pareto",
    "chi2",
    "cauchy",
)
DISCRETE_CANDIDATES: tuple[str, ...] = ("poisson", "nbinom", "geom", "binom")

PLACEHOLDER_STRINGS: tuple[str, ...] = (
    "", "na", "n/a", "n.a.", "nan", "null", "none", "nil", "-", "--", "?", ".",
    "missing", "unknown", "undefined", "#n/a", "#null!", "not available",
)
PLACEHOLDER_NUMBERS: tuple[float, ...] = (-999, -9999, -99999, 999, 9999, 99999, 999999)


@dataclass
class StoneConfig:
    """Thresholds and knobs. Percentages are expressed in 0-100."""

    # sampling
    sample_size: int = 100_000
    """Max rows used by expensive steps (correlations, Kendall, KDE, quality pairs)."""
    fit_sample_size: int = 20_000
    """Max values used for distribution fitting and normality tests."""
    random_state: int = 42
    max_rows: int = 1_000_000
    """Larger tables are profiled on a uniform random sample of this many rows; row counts, missing
    values, min/max/mean/std, distinct counts and duplicate rows stay exact over every row."""
    chunk_rows: int = 250_000
    """Rows per chunk when a file is streamed (files too big to profile in one piece)."""
    max_hashed_rows: int | None = None
    """Duplicate rows are counted exactly up to this many rows (16 bytes of memory per row); None = as
    many as a quarter of the available memory allows."""
    memory_fraction: float = 0.5
    """Share of the available RAM a table may take once loaded; bigger files are streamed."""

    # parallelism
    n_jobs: int | None = None
    """Processes for the heavy steps: None = automatic (all cores for big tables), 1 = no
    parallelism, -1 = all cores, k = at most k."""
    parallel_min_cells: int = 20_000
    """Below this many cells (rows x columns) the automatic mode stays in one process."""

    # types
    id_unique_ratio: float = 0.95
    pattern_match_ratio: float = 0.9
    ordinal_max_levels: int = 12
    ordinal_max_int_levels: int = 7
    discrete_max_levels: int = 50
    text_min_avg_words: float = 3.0
    text_min_avg_length: float = 40.0

    # stats
    top_n: int = 10
    rare_category_pct: float = 1.0
    max_categories: int = 50
    trim_fraction: float = 0.1

    # distributions
    alpha: float = 0.05
    candidates_continuous: tuple[str, ...] = CONTINUOUS_CANDIDATES
    candidates_discrete: tuple[str, ...] = DISCRETE_CANDIDATES
    fit_timeout: float = 5.0
    """Seconds allowed per candidate fit (enforced on POSIX main thread)."""
    n_best_fits: int = 3

    # correlations
    corr_threshold: float = 0.5
    """Pairs with |association| above this are listed as strong."""
    high_corr: float = 0.95
    """Pairs above this are flagged as redundant."""
    leakage_threshold: float = 0.98
    vif_threshold: float = 10.0
    """Variance inflation factor above which a column is flagged as multicollinear."""

    # quality
    high_missing_pct: float = 50.0
    medium_missing_pct: float = 20.0
    quasi_constant_pct: float = 99.0
    row_missing_pct: float = 50.0
    co_missing_jaccard: float = 0.8
    outlier_alert_pct: float = 5.0
    placeholder_strings: tuple[str, ...] = PLACEHOLDER_STRINGS
    placeholder_numbers: tuple[float, ...] = PLACEHOLDER_NUMBERS

    # plots
    plot_top_n: int = 15
    n_scatter: int = 4
    n_target_plots: int = 8
    dpi: int = 144
    single_file_report: bool = False
    """Also write ``report_single.html`` with every chart inlined (one file to send or attach)."""
