"""A taxonomy of ways quantitative research results turn out to be wrong.

Every entry below was extracted from a post-mortem of a real study that had
already been built, believed for a while, and then killed. They are grouped by
what actually did the killing, and each one names the check in this package
that catches it -- or records that no check does yet, which is the more useful
half of the list.

The mapping is the point. A failure mode with a `caught_by` is one the gauntlet
already defends against. A failure mode with `caught_by=()` is a hole, and the
holes are where the next checks should go.

Every mode here currently names a check, which is worth stating carefully. It
means every way of being wrong *that has been written down* is defended against.
It does not mean the list is complete: this is a record of what has gone wrong
so far, and the next entry will be added the way all the others were, by a study
that was built, believed, and then killed. A full-looking table is a reason to
add modes, not a reason to relax.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class Mode:
    id: str
    family: str
    name: str
    looks_like: str
    """What you observe before you know it is a problem -- the seductive form."""
    why: str
    """Why the observation is not what it appears to be."""
    caught_by: Tuple[str, ...] = ()
    """Check ids in this package that detect it. Empty means nothing does yet."""
    probe: str = ""
    """The cheapest experiment that settles it."""


MODES: Tuple[Mode, ...] = (
    # ---- nulls -------------------------------------------------------------
    Mode("null-turnover-unmatched", "null", "null does not match turnover",
         "The strategy sits at the 99th percentile of random portfolios.",
         "The random comparison rebalances far more often, pays far more cost, and "
         "would lose to almost anything. The percentile measures trading frequency, "
         "not selection.",
         ("SM1",), "Force the null to retain the same number of positions each period."),
    Mode("null-exposure-unmatched", "null", "null does not match exposure",
         "A switching rule beats a static allocation.",
         "The rule spends more time in the asset that happened to rise. Permuting the "
         "weight runs keeps every bucket's average weight and switch count, and the "
         "edge usually goes with it.",
         ("SM1", "M1"), "Permute the strategy's own weight runs in time."),
    Mode("null-dimension-unmatched", "null", "null misses a dimension that decided entry",
         "Treated names beat size-matched controls at the 100th percentile.",
         "Something other than size decided who entered -- liquidity, coverage, "
         "tradability. Match on every dimension that governed selection, not the one "
         "that is easiest to match.",
         ("M1",), "List what determined entry, then match the null on all of it."),
    Mode("null-random-structure-wins", "null", "a random structure does as well",
         "A learned graph, basis, regime split or weighting scheme improves the metric.",
         "Substituting a random structure of the same shape often matches or beats it. "
         "The gain came from having any structure, not from the learned one.",
         ("M4",), "Rerun with a random adjacency / basis / split / weight vector."),
    Mode("null-selection-unaware", "null", "placebo does not rerun the selection",
         "The chosen factor beats a shuffled version of itself.",
         "You chose it out of hundreds. The placebo has to repeat the whole selection "
         "on permuted data, otherwise it controls none of the search freedom.",
         ("S4",), "Circularly shift the factor panel and rerun the entire pick."),
    Mode("null-absent", "null", "no null baseline at all",
         "Significance, out-of-sample fit, and a plausible story.",
         "All three are compatible with a result a coin could produce at the same "
         "turnover and breadth.",
         ("M0", "M1", "SM1"), "Run one. It is usually the cheapest check available."),

    # ---- point-in-time -----------------------------------------------------
    Mode("leak-alignment-offset", "pit", "signal and label off by one step",
         "Implausibly high R-squared or IC, often above 0.5.",
         "The label is the same period as the feature, usually from an unsorted "
         "shift on a descending table or an index built the wrong way round.",
         ("A2",), "Slide the feature across shifts and find where the score first jumps."),
    Mode("leak-label-in-feature", "pit", "feature window covers the label period",
         "A window statistic predicts the very window it spans.",
         "The window's right edge sits after the timestamp it is published under.",
         ("A2", "A0"), "Rebuild the feature from history truncated at its timestamp."),
    Mode("leak-future-filtered-events", "pit", "triggers filtered using future labels",
         "Precision of 1.00, or a trigger set that never misses.",
         "The event list was pruned using the outcome. Under a symmetric rule the "
         "detector usually collapses to a volatility baseline.",
         ("M7",), "Rebuild the trigger set using only information available at trigger time."),
    Mode("leak-full-sample-statistic", "pit", "a threshold or scaling fitted on everything",
         "Clean-looking quantile thresholds, z-scores or shares outstanding.",
         "The quantile, mean or divisor was computed over the full sample including "
         "the evaluation period.",
         ("A0", "A1"), "Recompute every constant from a truncated history and compare."),
    Mode("leak-cumulative-state-gate", "pit", "a gate built on cumulative state",
         "A drawdown, running-high or net-asset-value gate improves everything.",
         "Cumulative series computed through the current close embed today's return.",
         ("A0",), "Rebuild the state series with the current bar excluded."),
    Mode("leak-same-day-survivorship", "pit", "universe requires surviving the day",
         "A tradable-universe filter that quietly requires a later price to exist.",
         "Requiring a close at 14:57 to exist selects names that did not halt.",
         ("A0",), "Define the universe strictly from pre-decision information."),
    Mode("leak-weights-at-close", "pit", "constituent or weight file is a close snapshot",
         "Index weights line up suspiciously well with same-day returns.",
         "The published weight for day D reflects day D's close and cannot be used "
         "to trade day D.",
         ("A0",), "Lag the weight file one period and see what survives."),
    Mode("leak-backtester-gives-today", "pit", "the backtest hands you the current bar",
         "A vendor backtest matches on the current close and its history call "
         "already includes today.",
         "The engine's convention, not your code, is doing the leaking.",
         ("P4",), "Check the engine's matching convention before trusting any number."),

    # ---- cost --------------------------------------------------------------
    Mode("cost-eats-it", "cost", "gross edge smaller than its turnover",
         "A monotone, significant, well-behaved signal.",
         "Break-even cost is below what the book would actually pay.",
         ("E1",), "Report break-even cost in basis points next to every IC."),
    Mode("cost-two-legs-do-not-net", "cost", "long-short costs assumed to cancel",
         "A long-short spread that looks cheap to run.",
         "Both legs pay. Netting them halves the true cost and can flip the sign.",
         ("E1",), "Charge each leg separately."),
    Mode("cost-spread-not-fee", "cost", "the bid-ask, not the commission, is the cost",
         "A cheap instrument by commission, expensive by quote.",
         "A low-priced ETF can have a spread ten times another's on the same index.",
         ("E1", "E3"), "Price the entry and exit at the touch, not at the trade print."),
    Mode("cost-double-counted", "cost", "spread charged twice",
         "A strategy that dies under 'realistic' costs.",
         "Executing at bid/ask already includes the spread; adding a spread "
         "assumption on top double-charges it.",
         ("E4",), "State exactly which frictions each price convention already contains."),
    Mode("cost-capacity-ceiling", "cost", "the edge lives in a window too small to use",
         "Large per-trade numbers in a specific execution window.",
         "The closing auction can be one percent of daily volume; the edge does not "
         "scale past a token size.",
         ("E5",), "Express the edge as a share of the volume available in that window."),

    # ---- statistics --------------------------------------------------------
    Mode("multiple-testing-unadjusted", "stat", "candidates searched but not charged for",
         "Two or three of thirty variants clear the threshold.",
         "That is what noise produces at a five percent threshold over thirty tries.",
         ("S4",), "Count every variant you looked at, including abandoned ones."),
    Mode("overlap-inflated-t", "stat", "overlapping windows treated as independent",
         "A t-statistic in the double digits on a few dozen observations.",
         "Consecutive forward windows share most of their span; the effective sample "
         "is a fraction of the row count.",
         ("S4", "A3"), "Use non-overlapping sampling or HAC with lag >= horizon-1."),
    Mode("icir-inflated-by-autocorrelation", "stat", "ICIR on a slow-moving label",
         "ICIR above 0.5 on more than half of a factor library.",
         "A label that barely changes makes the IC series strongly autocorrelated, "
         "so the ratio has no discriminating power.",
         ("S9",), "Rank on IC magnitude plus out-of-sample agreement, not on ICIR."),
    Mode("seed-instability", "stat", "one seed, one happy number",
         "A configuration that works, run once.",
         "Rerunning across fifty seeds can show the positive fraction is under a "
         "quarter and the mean is negative.",
         ("S5",), "Report the distribution across seeds and the share that are positive."),
    Mode("single-period-driven", "stat", "a handful of periods carry everything",
         "Strong annualised numbers.",
         "Removing the best few periods, or one crisis day, removes the result.",
         ("E2", "SE2"), "Report the top-5 share of P&L and the result without it."),
    Mode("sample-too-thin", "stat", "a real effect on too few events",
         "A clean threshold effect with a good t-statistic.",
         "Thirty-eight events in ten years cannot support a trading rule, and the "
         "threshold moves the sign.",
         ("S4",), "Report the event count and move the threshold to see the sign."),
    Mode("power-not-verified", "stat", "a zero result with unknown power",
         "A clean null finding.",
         "Zero is only informative if the same panel detects a known effect. Without "
         "that, the zero may be the pipeline.",
         ("S6",), "Run a known effect through the identical panel as a positive control."),
    Mode("train-up-valid-down", "stat", "the textbook overfitting signature",
         "Monotone improvement in training as a knob is turned.",
         "Validation moves monotonically the other way. The knob is fitting noise.",
         ("S8",), "Plot both segments against the knob before believing either."),
    Mode("test-set-consumed", "stat", "the sealed period was read and then tuned on",
         "A test result that improves after 'one small fix'.",
         "Once read, that segment is development data. Anything measured on it "
         "afterwards is in-sample.",
         ("P2",), "Log the unseal and refuse to re-tune; carry forward instead."),
    Mode("relative-metric-without-absolute", "stat", "a ratio improves while the level is bad",
         "Certainty-equivalent gain, utility improvement, or an information ratio.",
         "The absolute return can still sit below the risk-free rate.",
         ("E1",), "Report the absolute number beside every relative one."),

    # ---- mechanism ---------------------------------------------------------
    Mode("signal-is-self-reversal", "mech", "the signal is mostly its own past return",
         "A network, peer or gap factor with a healthy IC.",
         "Subtracting a peer average leaves the name's own reversal, which carries "
         "the IC regardless of who the peers are.",
         ("M1", "M3", "M4"), "Compare against random peers, not just against zero."),
    Mode("signal-is-exposure", "mech", "the result is a factor exposure",
         "A conditional rule that improves returns.",
         "Executing it unconditionally does as well or better -- the condition "
         "carries no information, the tilt does.",
         ("M1", "M3"), "Run the same operation unconditionally and at random times."),
    Mode("signal-is-market-not-cross-section", "mech", "a market event dressed as a factor",
         "A cross-sectional score that predicts returns.",
         "The whole cross-section moves together on those days; there is one "
         "observation, not hundreds. A market-level effect produces no "
         "cross-sectional IC at all, so M0 rejects it; M9 says why.",
         ("M0", "M9"), "Check whether within-day dispersion carries any of the effect."),
    Mode("signal-duplicates-existing", "mech", "a new name for an existing factor",
         "A new signal with an independent-sounding story.",
         "Correlation with something already in the book is above 0.9.",
         ("M3", "M4"), "Correlate against everything already in production first."),
    Mode("complexity-no-increment", "mech", "the machinery is not paying for itself",
         "An elaborate model beats a weak baseline.",
         "A naive construction on the same inputs does as well; the gain is in the "
         "inputs, not the architecture.",
         ("M4",), "Build the dumbest version that uses the same information."),
    Mode("stale-price-artifact", "mech", "the index is stale, the edge is not real",
         "Strong opening or intraday effects on an index.",
         "Index prints carry yesterday's close for names that have not traded; the "
         "effect grows monotonically as you go down in capitalisation.",
         ("E3",), "Rerun on a tradable instrument -- ETF or futures ticks, at the touch."),
    Mode("synthetic-data-badprints", "mech", "bad prints in a constructed series",
         "A defensive overlay that avoids sharp drops beautifully.",
         "The drops are spike-and-revert artefacts of the construction, not market "
         "moves.",
         ("M5",), "Re-run against the authoritative series before believing any overlay."),
    Mode("benchmark-wrong-basis", "mech", "the benchmark is computed on another basis",
         "A large, stable excess return.",
         "Unadjusted prices, an intraday-only chain, or an equal-weight proxy "
         "compared against a real index.",
         ("SE1",), "Reconcile the benchmark against its official series first."),
    Mode("data-frozen-silently", "mech", "an upstream input stopped updating",
         "Results that look normal and a gate that never fires.",
         "A cache or vendor series froze; the code reads the last good value forever.",
         ("S7",), "Assert on last observation date, not on whether values look plausible."),
    Mode("data-encoding-changed", "mech", "a filter silently matches nothing",
         "A feature that is exactly zero for a long stretch, not NaN.",
         "The source changed an enum and the filter now returns no rows; zero is a "
         "legal value downstream and propagates through rolling windows.",
         ("S7",), "Check against an external fact: how many names *should* have hit the limit?"),
    Mode("mechanism-true-but-untradable", "mech", "a real effect that cannot be traded",
         "A statistically solid conditional effect.",
         "Too rare, too small against costs, or concentrated where the position "
         "would already be profitable for other reasons.",
         ("E1", "E2"), "Price it as a standalone leg before adding it to anything."),
    Mode("threshold-effect-diluted", "mech", "a threshold effect modelled as continuous",
         "A continuous modulation that shows nothing.",
         "The effect lives in a few dozen extreme days; spreading it across every "
         "day averages it away.",
         ("M8",), "Test the extreme bucket separately before fitting a slope."),

    # ---- increments --------------------------------------------------------
    Mode("increment-is-the-act-not-the-thing", "null", "adding anything would have done it",
         "A candidate improves the book's curve.",
         "Adding a component with the same construction and turnover but no information "
         "moves the curve about as much. What was measured is the act of adding, not the "
         "thing added.",
         ("I2",), "Add a shuffled or random candidate of the same turnover and compare."),
    Mode("increment-buys-good-years-with-bad", "cost", "the addition takes the bad years with it",
         "A candidate raises the annualised number.",
         "It turns years that were flat into losing ones, or its whole gain is a single "
         "year. The book that gets approved is not the book that was measured.",
         ("I1",), "Report the delta year by year, signed, and count the losing years."),

    # ---- process -----------------------------------------------------------
    Mode("frozen-config-not-enforced", "proc", "the frozen file is written but never read",
         "A config committed as frozen.",
         "Downstream scripts recompute their own cut points, so upstream drift "
         "silently changes results without changing the file.",
         ("P3",), "Assert the recomputed values equal the frozen ones, and halt if not."),
    Mode("recompute-vs-rerun", "proc", "rerunning the script is not recomputing the data",
         "A refreshed report.",
         "Intermediate caches were not rebuilt, so the report shows old numbers "
         "computed from a newer-looking pipeline.",
         ("M6",), "Assert every intermediate's last date equals the source's."),
    Mode("internal-consistency-not-enough", "proc", "value-by-value agreement proves little",
         "A reimplementation that matches the original to machine precision.",
         "It proves the two agree, not that either is right. A shared conceptual "
         "error survives every value-by-value check.",
         ("P5",), "Check against an external fact the pipeline never sees."),
)


BY_ID: Dict[str, Mode] = {m.id: m for m in MODES}
FAMILIES = ("null", "pit", "cost", "stat", "mech", "proc")


def by_family(family: str) -> List[Mode]:
    return [m for m in MODES if m.family == family]


def uncovered() -> List[Mode]:
    """Modes no check in this package detects -- the roadmap, in other words."""
    return [m for m in MODES if not m.caught_by]


def coverage() -> Dict[str, Tuple[int, int]]:
    out = {}
    for f in FAMILIES:
        fam = by_family(f)
        out[f] = (sum(1 for m in fam if m.caught_by), len(fam))
    return out


def render_checklist(modes: Sequence[Mode], width: int = 96) -> str:
    lines = []
    for m in modes:
        caught = ", ".join(m.caught_by) if m.caught_by else "no check yet"
        lines.append(f"  [{m.family}] {m.name}  ({caught})")
        lines.append(f"      looks like : {m.looks_like}")
        lines.append(f"      actually   : {m.why}")
        if m.probe:
            lines.append(f"      probe      : {m.probe}")
    return "\n".join(lines)
