"""falsifier -- an adversarial referee for quantitative research claims.

The machine does not look for reasons a claim might work. It runs a fixed
battery of attacks along four axes -- process, point-in-time, mechanism, cost --
and reports which one killed it. The best available outcome is SURVIVES, which
means "not yet falsified" and never means "true".
"""
from .econ import (cost_gate, e3_execution_delay, e4_cost_convention, e5_capacity,
                   quantile_portfolio, trade_metrics)
from .charts import plot_increment
from .gauntlet import Study, run
from .increment import (Increment, excess, i1_incremental_contribution,
                        i2_increment_null, render_yearly, run_increment)
from .mech import (m0_harness, m1_matched_null, m2_identity_null,
                   m3_orthogonalize, m4_beats_naive, orient)
from .nulls import (cs_shuffle, empirical_p, identity_permutation,
                    matched_permutation, null_distribution, orthogonalize,
                    percentile_of, random_selection)
from .pit import a0_truncation, a1_label_shuffle_refit, a2_feature_shift, a3_label_delay, audit
from .pipeline import RollingFit
from .prereg import Prereg, p3_frozen_config, p4_fill_convention, p5_external_facts
from .robust import (default_control, s5_seed_stability,
                     m5_print_quality, m6_input_freshness, s6_positive_control,
                     m7_event_integrity, m8_threshold_or_slope, s7_input_staleness,
                     m9_cross_sectional_independence, s8_knob_monotonicity,
                     s9_label_persistence)
from .priors import Index, Prior, checklist, load as load_priors, render as render_priors, search as search_priors
from .taxonomy import MODES, BY_ID, Mode, by_family, coverage, uncovered
from .seal import SealedSplit, SealError
from .strategy import (StrategyStudy, backtest, free_selection_null,
                       matched_selection_null, perf, run_strategy)
from .stats import (deflated_threshold, forward_returns, ic_summary,
                    newey_west_t, rank_ic)
from .verdict import Check, Report

__version__ = "0.1.0"
__all__ = [
    "s5_seed_stability", "s6_positive_control", "s7_input_staleness",
    "s8_knob_monotonicity", "s9_label_persistence", "m5_print_quality",
    "m6_input_freshness", "m7_event_integrity", "m8_threshold_or_slope",
    "m9_cross_sectional_independence",
    "e3_execution_delay", "e4_cost_convention", "e5_capacity",
    "p3_frozen_config", "p4_fill_convention", "p5_external_facts", "default_control",
    "RollingFit", "Prior", "Index", "load_priors", "search_priors", "checklist", "render_priors",
    "MODES", "BY_ID", "Mode", "by_family", "coverage", "uncovered",
    "Increment", "excess", "render_yearly", "i1_incremental_contribution",
    "i2_increment_null", "plot_increment", "run_increment",
    "Study", "run", "StrategyStudy", "run_strategy", "backtest", "perf",
    "matched_selection_null", "free_selection_null", "Prereg", "SealedSplit", "SealError", "Check", "Report",
    "audit", "a0_truncation", "a1_label_shuffle_refit", "a2_feature_shift", "a3_label_delay",
    "cs_shuffle", "matched_permutation", "identity_permutation", "orthogonalize",
    "random_selection", "null_distribution", "percentile_of", "empirical_p",
    "m0_harness", "m1_matched_null", "m2_identity_null", "m3_orthogonalize", "m4_beats_naive", "orient",
    "rank_ic", "ic_summary", "newey_west_t", "forward_returns", "deflated_threshold",
    "quantile_portfolio", "trade_metrics", "cost_gate",
]
