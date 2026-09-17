"""A taxonomy of ways quantitative research results turn out to be wrong.

Every entry below was extracted from a post-mortem of a real study that had
already been built, believed for a while, and then killed. They are grouped by
what actually did the killing, and each one names the check in this package
that catches it -- or records that no check does yet, which is the more useful
half of the list.

The mapping is the point. A failure mode with a `caught_by` is one the gauntlet
already defends against. A failure mode with `caught_by=()` is a hole, and the
holes are where the next checks should go.

The table has been full twice, and is not full now. Both times, backfilling more
post-mortems into the prior records reopened it -- which is the traffic this is
built for, and the reason `uncovered()` stays published instead of being quietly
closed. A full table only ever meant that every way of being wrong *that had
been written down* was defended against, and the list of what has been written
down is the part that grows.
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
    Mode("shift-audit-blind-on-smoothed-signals", "pit",
         "the leak detector cannot see a step this small",
         "An INCONCLUSIVE from the feature-shift audit, read as a suspicion of look-ahead.",
         "It is a statement about the instrument. The shift audit locates a leak by sliding the "
         "feature across the label and finding where the score first steps up; when the signal "
         "is an average over a span of L, one step of slide changes about 1/L of it, so the step "
         "shrinks as 1/L while the sampling error of the IC series does not shrink at all. Past "
         "some amount of smoothing there is no step left to find and the audit returns "
         "INCONCLUSIVE whatever is true -- and a clean result then gets treated as a leak, which "
         "is the expensive direction of the mistake.",
         ("A4",),
         "Size the step against the panel's own noise floor before reading the verdict, and "
         "settle it with A0/A1 instead -- rebuilding and refitting do not slide, so smoothing "
         "does not touch them."),
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
    Mode("panel-includes-the-sealed-period", "proc",
         "the split was declared and the data ignored it",
         "A study with a clean train/valid/test split, a sealed test period, and an untouched "
         "seal ledger.",
         "The peek did not come from unsealing. A yearly diagnostic was printed without "
         "excluding the test rows; a panel was assembled over the whole history and handed on. "
         "Every check then runs, every number is computed correctly, and all of them are about "
         "a panel that was not allowed to exist -- which is worse than an obvious error, "
         "because nothing looks wrong. Masking the rows is not enough either: a full-sample "
         "scaling, a quantile cut point or a rolling statistic consumes them anyway.",
         ("P9",),
         "Check the rows the panel actually carries against the boundary, before anything "
         "reads it, and take the sealed rows out rather than merely leaving them unused."),
    Mode("test-set-consumed", "stat", "the sealed period was read and then tuned on",
         "A test result that improves after 'one small fix'.",
         "Once read, that segment is development data. Anything measured on it "
         "afterwards is in-sample.",
         ("P2",), "Log the unseal and refuse to re-tune; carry forward instead."),
    Mode("metric-depends-on-the-base-rate", "stat",
         "a threshold carried over from a market where the event was rarer",
         "A decision rule with a threshold, an F-score or a precision target taken from the "
         "paper that introduced it.",
         "Those numbers are not properties of the classifier, they are properties of the "
         "classifier *at that prevalence*. A probe on one A-share name put the ten-second fill "
         "rate at 0.43 where the paper had 0.04 -- an order of magnitude -- so every threshold "
         "calibrated against the rarer event is meaningless here, and the imported F-score "
         "target is unreachable and unmeaningful at once. The tell is that the rule was tuned "
         "somewhere with a different event frequency and moved without re-tuning.",
         ("P5",),
         "Measure the base rate on your own data before importing any threshold, and re-derive "
         "the operating point rather than the number."),
    Mode("decision-space-degenerate", "mech",
         "the thing the strategy chooses has nothing to choose between",
         "A well-specified rule about where to post, which venue to use, or which tier to take.",
         "In this market the control variable is pinned. Measured on Shenzhen snapshots, the "
         "spread is exactly one tick 99.3% of the time on a 11.8-yuan name and still 88.8% at "
         "45 yuan; it takes a price above 100 to open up. A rule that optimises the posting "
         "distance is then optimising over a single available value, and whatever it reports is "
         "the value of that one choice rather than of the optimisation.",
         (),
         "Before building the optimiser, plot the distribution of the variable it is supposed to "
         "choose. If it is a point mass, the strategy has no decision to make here."),
    Mode("negative-without-a-reproduced-baseline", "proc",
         "a rejection nobody can attribute",
         "A careful replication that fails, reported as the effect not holding in this market.",
         "The original result was never reproduced on its own data, so the failure has at least "
         "two explanations -- the market differs, or this implementation does -- and nothing in "
         "the study separates them. That is not a weaker version of a rejection; it is a "
         "different claim, and the cheap fix is to run the original leg first on a market where "
         "the answer is known, which is why free US tick data is worth the download before "
         "touching anything proprietary.",
         ("S6",),
         "Reproduce the paper's own number on the paper's own market first. Until that leg "
         "exists, a negative result is unattributable."),
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
    Mode("baseline-sees-fewer-inputs", "mech",
         "the baseline was given less information than the model",
         "A learned representation beats the textbook baseline on the same panel.",
         "The baseline is the one from the literature, built on one series; the model "
         "eats a wider input. The measured 'increment' then mixes two things -- what the "
         "representation extracted, and what the baseline was simply never shown. "
         "complexity-no-increment does not cover this: its premise is *the same inputs*.",
         ("M4", "M3"),
         "Two steps, and the second is the one that gets skipped: (1) list the model's "
         "input columns and build the baseline on all of them; (2) ask again what "
         "derived quantities those same columns still support -- above all at another "
         "sampling frequency. Alignment is recursive, not a one-off check: a baseline "
         "that already used every column was still short a whole frequency's worth. "
         "And check the baseline's calibration before reading any side-split: a "
         "systematically high baseline makes anything that corrects downward look "
         "one-sidedly useful."),
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
    Mode("absurd-number-is-a-data-question", "mech",
         "the number is too large to be about the market",
         "A result so good it is exciting -- a cumulative multiple in the hundreds of "
         "thousands, a rank IC above 0.3, a precision of 1.00.",
         "At that size the answer is almost never that the signal is very good. Daily "
         "equal-weight rebalancing on names that barely trade produced 498,424x; an "
         "unadjusted close turned a +1.40% day into -28.4% and cost eleven points a year; a "
         "nan_to_num turned an inf into 1.8e308 and poisoned the cumulative sum. The "
         "expensive mistake is not believing the number -- it is spending the next week "
         "improving a model that is fitting a defect. **An absurd result is a data question, "
         "not a model question.**",
         ("M13",),
         "Before anything else: the return definition, non-finite values, bad prints, whether "
         "the universe trades, alignment, and whether the weights are decimals or percent."),
    Mode("universe-predates-the-index", "mech",
         "constituent history from before the index existed",
         "A clean membership file covering ten years of an index.",
         "The index was published two years ago. Everything before that is the vendor "
         "applying today's methodology backwards, by someone who knew what that methodology "
         "was built to select -- nine years of it for one index on this machine, with nothing "
         "in the store to say so. Nobody could have held that universe, so nothing on the "
         "reconstructed stretch is tradable; and nobody else's result on it is comparable, so "
         "a finding that seems to contradict the literature may be contradicting the universe.",
         ("M14",),
         "Compare the panel's first date against the index's publication date before building "
         "the universe."),
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
         "Something upstream changed so the selection now matches nothing -- the source "
         "changed an enum, or the loader hard-codes a filename that covers only part of the "
         "universe. Zero is a legal value downstream and propagates through rolling windows, "
         "so the arm that depends on it never fires and every report looks ordinary.",
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

    # ---- mechanism, continued ----------------------------------------------
    Mode("mechanism-stated-but-not-the-one-working", "mech",
         "the result replicates, its stated mechanism does not",
         "A model's pricing or predictive content reproduces closely, sometimes to the decimal.",
         "The economic channel it claims to work through does not: the proposed "
         "interaction has the wrong sign, or the proxy for it carries nothing. The "
         "number survives and the story does not, and only the story generalises.",
         ("P6",), "Write the implications down before testing, then test them."),
    Mode("non-stationary-needs-retraining", "mech", "a frozen representation goes stale",
         "A learned representation validates well and is frozen for reuse.",
         "The relationship it encodes moves. Held fixed it decays and can reverse sign, "
         "while the same architecture retrained on a rolling window keeps working -- so "
         "the retraining is the mechanism, not an optimisation.",
         ("M10",), "Compare a frozen fit against a rolling one on the later half of the sample."),
    Mode("mechanism-not-established", "proc",
         "a mechanical argument used as a premise instead of tested",
         "A story so mechanical it does not feel like a hypothesis -- cost is fixed and the "
         "gross edge scales with volatility, so the high-volatility days must be the good ones.",
         "It is still a hypothesis, and the direction of a whole search gets chosen by it "
         "before anyone measures it. Measured, it can be flat or backwards: the rank "
         "correlation between the driver and the edge comes out at zero and the top bucket "
         "is the worst one. Everything searched along that direction was then searched for "
         "no reason, and whatever survived is the multiple-testing residue.",
         ("P6",),
         "Regress the outcome on the quantity the argument says drives it, and bucket it, "
         "before searching along it."),
    Mode("control-contains-the-target", "mech",
         "orthogonalised against a piece of the answer",
         "A residual IC that comes out *higher* than the raw IC after controlling for a covariate.",
         "Residualising against something legitimate can only take information away. A "
         "residual that rises has inherited the part of the label that was sitting inside the "
         "regressor -- a control measured over a window that overlaps the target, or one that "
         "is a component of it. Every matched null and every orthogonalisation built on that "
         "control is then inflated rather than cleaned.",
         ("M12", "M3"),
         "Run the boundary locator on each control, not only on the signal, and separate the "
         "controls by decision timestamp."),
    Mode("entry-blocked-at-the-limit", "cost", "the book bought what was locked",
         "An event signal with a clean IC, cheap turnover and a cost gate it passes easily.",
         "It fires on the names that just jumped, which are the names sitting on a locked "
         "board, and a locked board cannot be bought. The constraint is not a price, it is an "
         "absence, so the cost axis structurally cannot see it: the backtest fills the order "
         "at a price nobody could have paid and every number downstream is computed on a book "
         "that was never available. In one published decomposition this single line cost 10.8 "
         "percentage points a year against a final 13.3 -- larger than cost, the liquidity "
         "floor and the position cap together.",
         ("E6",),
         "Remove the entries the book could not have taken -- do not charge more for them, "
         "remove them -- and fill the slots further down the ranking as a desk would."),
    Mode("filtered-away-the-exposure", "mech",
         "the losses that were filtered out were the price of the exposure",
         "A working strategy whose losing trades look like an obvious defect, and filters that "
         "remove them and raise the backtest Sharpe every time one is added.",
         "The losses were not a defect. A trend book earns a few large gains by paying for many "
         "small false starts, and the false starts are what buys the exposure -- remove enough "
         "of them and what is left is not a cleaner trend book, it is a timing model, which is "
         "a different thing and usually one nobody has an edge in. The tell is that the "
         "backtest improves monotonically as conditions are added while live results go the "
         "other way, and the diagnosis is not overfitting: the author understood the source of "
         "the profit incorrectly and then optimised efficiently against the wrong target. "
         "Research capability makes this worse rather than better.",
         ("M4", "I1"),
         "Take the version *before* the filters as the naive baseline the filtered version has "
         "to beat, rather than as the thing being improved."),
    Mode("improvement-outside-the-traded-set", "cost",
         "the metric improved where the book does not hold",
         "A model that sharpens the ranking on the event sample by a clear margin.",
         "The population it improved on is not the population the book holds. Judging the "
         "bottom of a distribution more accurately changes nothing about a top-fifty "
         "portfolio, so a per-event metric can rise while the book's excess falls -- one "
         "meta-label reported 0.85% a month on events while taking the portfolio from 4.9% to "
         "3.8%. The improvement is real and unreachable, which is not the same as absent.",
         ("I1", "I2"),
         "Measure the candidate on the book that will actually be held, not on the sample it "
         "was scored on."),
    Mode("ic-not-tradable", "cost",
         "the IC is stable and the money is not",
         "A signal whose IC holds up out of sample, sometimes more strongly than in sample.",
         "The rule that consumes it is binary -- hold or do not hold, in or out -- and a "
         "binary split throws away most of the ranking the IC measured. Where the IC's "
         "contribution sits in the tail, the split puts the tail and the middle on the same "
         "side. A stable IC and a negative economic increment are not a contradiction.",
         ("M8", "E1"),
         "Price the rule, never the IC, and check whether the effect is graded before "
         "consuming it with a threshold."),
    Mode("breadth-too-narrow-for-the-ic", "cost",
         "the same IC does not survive a narrow universe",
         "A signal with a healthy, stable IC is carried into a small universe -- an index of "
         "fifty names, a sector, a themed basket -- on the reasoning that the IC is positive there too.",
         "Information ratio scales as IC times the square root of the number of independent bets. "
         "Holding five names out of fifty, idiosyncratic variance swamps the edge a 0.05 IC buys, "
         "and the same construction that earns in a thousand-name universe loses double digits in a "
         "fifty-name one. The IC being positive in the narrow universe is not the question; whether "
         "it is large enough for that few positions is.",
         ("E1",),
         "Before porting a signal to a narrower universe, solve IR = IC x sqrt(N_holdings) for the IC "
         "the new holding count would require, and compare it against the IC you actually have."),
    Mode("rule-degenerates-to-the-benchmark", "stat",
         "the rule that passed had stopped being a rule",
         "One cell of a threshold sweep clears every criterion -- positive in both segments, beating "
         "its matched null -- while its neighbours do not.",
         "At that threshold the rule is in the market almost all the time and trades a handful of "
         "times over the whole sample. It has degenerated into buy-and-hold, so the excess it reports "
         "is the benchmark's own return arriving under another name. A sweep that does not exclude "
         "degenerate cells will eventually hand one back as its best result.",
         (),
         "Record exposure share and trade count for every cell, and drop any cell above ~95% exposure "
         "or with a single-digit trade count before ranking; then compare the pass count against the "
         "number the criteria would pass by chance."),
    Mode("validated-on-substitute-data", "mech", "it worked on the stand-in source",
         "A construction validated on sample, vendor or reconstructed data.",
         "On the source that will actually be traded from, the values are degenerate -- "
         "mostly zero, mostly missing, or on a different scale. The construction assumed "
         "a distribution the real feed does not have.",
         ("M0",), "Rebuild the factor on the production source before reading any IC."),
    Mode("conclusion-does-not-transfer-across-frequency", "mech",
         "the same construction, a different frequency, the opposite answer",
         "A pattern established at one sampling frequency.",
         "At another it is weaker, absent, or reversed -- a shape signal that works daily "
         "can backfire at one minute, and a rule that pays weekly can be eaten by costs "
         "daily. Frequency is not a hyper-parameter; it is a different question.",
         ("M11",), "Re-measure at the target frequency rather than rescaling the result."),

    # ---- increments --------------------------------------------------------
    Mode("increment-is-the-act-not-the-thing", "null", "adding anything would have done it",
         "A candidate improves the book's curve.",
         "Adding a component with the same construction and turnover but no information "
         "moves the curve about as much. What was measured is the act of adding, not the "
         "thing added.",
         ("I2",), "Add a shuffled or random candidate of the same turnover and compare."),
    Mode("composite-dilutes-the-good-component", "null",
         "an equal-weight composite is worse than its best part",
         "Several signals combined into one score.",
         "One of them carries no information, and at equal weight it takes half the "
         "score with it. Combining is not free: a good component and a null component "
         "average to something worse than the good component alone.",
         ("I2", "M4"), "Report each component's residual IC before combining anything."),
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
    Mode("estimator-not-identified", "stat", "the objective has more unknowns than constraints",
         "A calibration reports a parameter surface, with quantiles and a story about "
         "what moves it.",
         "The objective is solved for two or more parameters against one observation "
         "per contract, so its solution set is a curve rather than a point. What gets "
         "reported is where the optimiser stopped, which is a fact about the starting "
         "value and the search, not about the data. The tell in a published table is "
         "quantiles that collapse onto the same constant, and worse, the same constant "
         "in two unrelated subsamples.",
         ("S10",), "Rerun the identical fit from two different starting points. A parameter "
             "that follows the start is not estimated. Check which parameters move and "
             "which do not: often one is identified and the other is along for the ride."),
    Mode("hypothesis-contaminated-by-hindsight", "proc",
         "the idea came from something that already knew the answer",
         "A hypothesis that arrives well-aimed, and a pipeline that audits clean end to end.",
         "Truncating the price database at a date does not establish that the system has "
         "never seen what happened next. Training corpora, search results, retrieved "
         "documents and tool output all carry later events back across the boundary, so the "
         "hypothesis can be formed with the outcome in hand while every line of code that "
         "tests it is scrupulously point-in-time. A0 and A1 then audit a pipeline that was "
         "pointed in the right direction for the wrong reason, and both come back clean -- "
         "the leak is in what was proposed, not in how it was computed, and no downstream "
         "audit reaches it.",
         ("P8",),
         "Ask the source a question whose answer became knowable only after the boundary it "
         "claims to reason from, plus one from before it so that silence can be read."),
    Mode("check-structurally-cannot-fire", "proc", "a check that could never have failed",
         "A validation step that has passed on every run since it was written.",
         "It cannot fail. A no-NaN assertion written with `~isfinite` runs on a nullable "
         "dtype where NA passes straight through and is skipped by the sum; a release gate "
         "that compares `old.notna() & new.notna()` excludes exactly the rows where a value "
         "appeared or vanished, which is what it was there to find. A guard nobody has "
         "watched reject anything is not a guard, and its green is the most expensive kind "
         "of reassurance because it is spent on the thing you thought you had covered.",
         ("P7",),
         "Corrupt the data in the way the check exists to catch, and require it to fail. "
         "A validator that cannot be made to fail is not a validator."),
    Mode("cache-indexed-by-position", "proc", "a cache keyed by position into a growing set",
         "A cache that reproduces exactly on the day it was built and drifts afterwards.",
         "The cache stores integer offsets into a sorted universe, and the universe grows. "
         "Every new listing inserts and pushes everything after it along by one, while the "
         "stored offsets are never remapped, so the same cache entry resolves to a different "
         "name every day. Nothing errors, the shapes match, the row counts match, and the "
         "members are wrong by a sliding amount that looks like noise.",
         (),
         "Resolve a cached entry on two dates and check it names the same thing; better, "
         "check the pipeline's own declared invariant on the resolved output."),
    Mode("fields-from-inconsistent-sources", "mech", "one row, two selection rules",
         "A panel row whose every column is individually correct.",
         "The columns were selected under different rules -- one takes the nearest-expiry "
         "contract, another everything expiring at least two months out -- so the row "
         "describes no single instrument. Each column audits clean on its own, and the "
         "defect only appears when the selection rule is compared across columns.",
         (),
         "For every column in the row, write down the rule that chose it, and require them "
         "to be the same rule."),
    Mode("assumption-measurable-and-wrong", "proc",
         "an assumption that could have been measured instead",
         "A method built around a stated limitation of the data -- an interval rather than a "
         "point, a simulator rule, a conservative bound -- taken over from the paper that "
         "introduced it.",
         "The limitation is a fact about *that* market's data, and it travelled here with the "
         "method rather than being re-checked. One paper's entire premise was that aggregated "
         "depth loses which resting order was cancelled, so passive fill rates can only be "
         "bounded; on this market the cancelling order names itself in both exchanges' feeds "
         "and the bound collapses to a measurement. The cost runs both ways: an assumption "
         "that is falsely restrictive throws away precision you have, and one that is falsely "
         "permissive is a leak.",
         ("P5",),
         "Before adopting a method's stated limitation, check whether it binds on your data. "
         "Queue discipline, tick size, settlement timing, borrow availability and publication "
         "lag are all routinely assumed and all directly observable."),
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
