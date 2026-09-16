"""The economic axis: what survives contact with the cost of trading it.

A statistic that clears every null and every audit is still worthless if the
turnover it needs eats the edge. Report the per-trade block on every result --
trigger count, hit rate, average trade, payoff ratio, profit factor -- because
an annualised number alone hides a strategy that made all its money on four
days.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .verdict import FAIL, INCONCLUSIVE, NA, PASS, Check


def quantile_portfolio(signal: np.ndarray, ret1d: np.ndarray, mask: np.ndarray,
                       q: float = 0.1, hold: int = 1, long_short: bool = True,
                       min_n: int = 30) -> Dict[str, np.ndarray]:
    """Equal-weight quantile portfolio with a minimum holding period.

    Rebalances whenever the cross-section actually carries a signal and at
    least ``hold`` steps have passed, rather than on a fixed calendar modulus.
    A signal published every k-th step -- which is what any study evaluated at
    a k-step horizon looks like -- lands on dates that no modulus knows about,
    and a scheduler that insists on its own calendar simply never trades.

    Returns the gross per-step return, one-way turnover, and the dates on which
    a rebalance actually happened, so the cost gate uses the strategy's own
    trading pattern rather than an assumed one.
    """
    sig, r = np.asarray(signal, float), np.asarray(ret1d, float)
    T, N = sig.shape
    w = np.zeros(N)
    gross, turn, nsel = np.full(T, np.nan), np.zeros(T), np.full(T, np.nan)
    rebalances = []
    last = -(10 ** 9)
    for t in range(T - 1):
        m = np.isfinite(sig[t]) & mask[t]
        idx = np.flatnonzero(m)
        if idx.size >= min_n and (t - last) >= hold:
            k = max(1, int(idx.size * q))
            order = idx[np.argsort(sig[t, idx])]
            new = np.zeros(N)
            new[order[-k:]] = 1.0 / k
            if long_short:
                new[order[:k]] = -1.0 / k
            turn[t] = float(np.abs(new - w).sum() / 2.0)
            w = new
            nsel[t] = k * (2 if long_short else 1)
            last = t
            rebalances.append(t)
        gross[t] = float(w @ np.nan_to_num(r[t + 1], nan=0.0))
    return {"gross": gross, "turnover": turn, "n_selected": nsel,
            "rebalances": np.array(rebalances, dtype=int)}


def trade_metrics(trade_bp: np.ndarray) -> Dict[str, float]:
    """The per-trade block that must accompany every reported result."""
    v = np.asarray(trade_bp, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n_trades": 0, "win_rate": np.nan, "avg_bp": np.nan,
                "payoff_ratio": np.nan, "profit_factor": np.nan, "worst_bp": np.nan}
    win, loss = v[v > 0], v[v < 0]
    return {
        "n_trades": int(v.size),
        "win_rate": float((v > 0).mean()),
        "avg_bp": float(v.mean()),
        "payoff_ratio": float(win.mean() / abs(loss.mean())) if loss.size and loss.mean() != 0 else np.inf,
        "profit_factor": float(win.sum() / abs(loss.sum())) if loss.size and loss.sum() != 0 else np.inf,
        "worst_bp": float(v.min()),
    }


def cost_gate(gross: np.ndarray, turnover: np.ndarray, cost_bp: float,
              periods_per_year: int = 252, min_trades: int = 30) -> Check:
    """E1 -- does the edge survive its own turnover at the stated cost?

    ``cost_bp`` is one-way, in basis points, and should be the rate you would
    actually pay including impact -- not the exchange fee. A long-short book
    pays it on both legs; the two legs never net out.
    """
    g = np.asarray(gross, float)
    to = np.asarray(turnover, float)
    m = np.isfinite(g) & np.isfinite(to)
    if m.sum() < min_trades:
        return Check("E1", "economic", "net of cost", INCONCLUSIVE,
                     detail=f"only {int(m.sum())} usable periods")
    g, to = g[m], to[m]
    net = g - to * 2.0 * cost_bp / 1e4  # both legs
    ny = g.size / periods_per_year
    ann_gross = float(np.prod(1 + g) ** (1 / ny) - 1) if ny > 0 else np.nan
    ann_net = float(np.prod(1 + net) ** (1 / ny) - 1) if ny > 0 else np.nan
    ann_to = float(to.mean() * periods_per_year)
    breakeven = float(g.mean() / to.mean() * 1e4 / 2.0) if to.mean() > 0 else np.inf
    ok = ann_net > 0
    return Check("E1", "economic", "net of cost", PASS if ok else FAIL,
                 statistic=ann_net * 100, threshold=0.0,
                 detail=(f"gross {ann_gross:+.2%} - turnover {ann_to:.1f}x @ {cost_bp}bp "
                         f"=> net {ann_net:+.2%}; break-even cost {breakeven:.2f}bp"),
                 evidence={"ann_gross": ann_gross, "ann_net": ann_net,
                           "ann_turnover": ann_to, "breakeven_cost_bp": breakeven,
                           "ir_net": float(net.mean() / net.std(ddof=1) * np.sqrt(periods_per_year)) if net.std(ddof=1) > 0 else np.nan})


def trade_block_check(gross: np.ndarray, turnover: np.ndarray, cost_bp: float,
                      hold: int = 1, rebalances: Optional[np.ndarray] = None) -> Check:
    """E2 -- advisory: is the result carried by a handful of periods?

    Blocks run from one rebalance to the next where those are known, so a trade
    is a holding period rather than an arbitrary slice of the calendar.
    """
    g = np.asarray(gross, float)
    to = np.asarray(turnover, float)
    net_full = np.where(np.isfinite(g), g, 0.0) - np.nan_to_num(to) * 2.0 * cost_bp / 1e4
    if rebalances is not None and len(rebalances) > 1:
        edges = list(rebalances) + [len(net_full)]
        blocks = [net_full[edges[i]:edges[i + 1]].sum() * 1e4 for i in range(len(edges) - 1)]
    else:
        m = np.isfinite(g)
        net = net_full[m]
        blocks = [net[i:i + hold].sum() * 1e4 for i in range(0, net.size, hold)]
    tm = trade_metrics(np.array(blocks))
    total = float(np.sum(blocks))
    # A concentration ratio is only meaningful against a total worth
    # concentrating; near zero it explodes and says nothing.
    top5 = (float(np.sort(np.array(blocks))[-5:].sum() / total)
            if len(blocks) > 5 and total > 1.0 else np.nan)
    conc_ok = not (np.isfinite(top5) and top5 > 0.8)
    return Check("E2", "economic", "per-trade block", PASS if conc_ok else FAIL,
                 blocking=False, statistic=tm["avg_bp"],
                 detail=(f"n={tm['n_trades']} win={tm['win_rate']:.1%} avg={tm['avg_bp']:+.1f}bp "
                         f"payoff={tm['payoff_ratio']:.2f} PF={tm['profit_factor']:.2f} "
                         f"worst={tm['worst_bp']:.0f}bp"
                         + ("" if conc_ok else f"; top-5 periods are {top5:.0%} of total P&L")),
                 evidence=tm | {"top5_share": top5})


def e3_execution_delay(signal: np.ndarray, ret: np.ndarray, mask: np.ndarray,
                       horizon: int = 1, q: float = 0.1, cost_bp: float = 0.0,
                       max_delay: int = 3, tradable_ret: Optional[np.ndarray] = None,
                       price_source: str = "tradable", min_n: int = 30,
                       periods_per_year: int = 252) -> Check:
    """E3 -- push the entry later, and price it on what you would actually trade.

    An index print carries the previous close for every constituent that has not
    traded yet, so the index partly reports information the market has already
    seen. A signal formed on that information then appears to predict the next
    print, and the effect grows monotonically as you go down in capitalisation
    because thinner names go stale more often. The same shape comes from
    bid-ask bounce on an illiquid contract: the edge is the spread coming back,
    not a forecast.

    Two ways to settle it. The decisive one is to re-run on the instrument a
    book would hold -- an ETF, a future, the actual contract -- and see how much
    survives; pass it as ``tradable_ret``. Failing that, delay the entry: a real
    h-step edge keeps roughly (h-1)/h of itself after one step of delay, because
    the windows still overlap, while an artefact is gone at the first step.

    At a one-step horizon with no tradable series the two are not separable from
    this panel alone, and the check says so rather than guessing.
    """
    # Without a tradable series the two explanations for a fast decay -- a stale
    # print and a genuinely short-lived edge -- cannot be separated. That used to
    # make this NA, which threw away the decay profile itself: how much of the
    # edge survives a one-step delay is worth knowing either way, and on a weekly
    # book it is most of the question. So the profile runs and the check says
    # what it cannot settle, rather than not running.
    _undecidable = (price_source == "tradable" and tradable_ret is None)

    def ann_of(sig, rr):
        pf = quantile_portfolio(sig, rr, mask, q=q, hold=max(1, horizon),
                                long_short=True, min_n=min_n)
        g = pf["gross"]
        m = np.isfinite(g)
        if m.sum() < 20:
            return np.nan
        net = g[m] - np.nan_to_num(pf["turnover"][m]) * 2.0 * cost_bp / 1e4
        ny = net.size / periods_per_year
        return float(np.prod(1 + net) ** (1 / ny) - 1) if ny > 0 else np.nan

    def lag(sig, d):
        if d == 0:
            return sig
        out = np.full_like(sig, np.nan)
        out[d:] = sig[:-d]
        return out

    sig = np.asarray(signal, float)
    base = ann_of(sig, ret)
    prof = {d: ann_of(lag(sig, d), ret) for d in range(0, max_delay + 1)}
    ev = {"ann_by_delay": {str(d): float(v) for d, v in prof.items()}, "base": float(base)}

    if not np.isfinite(base) or abs(base) < 1e-9:
        return Check("E3", "economic", "execution delay / price artefact", INCONCLUSIVE,
                     blocking=False,
                     detail="no edge at zero delay, nothing to decay", evidence=ev)

    if tradable_ret is not None:
        tr = ann_of(sig, tradable_ret)
        keep = float(tr / base)
        ev |= {"ann_tradable": float(tr), "retained_on_tradable": keep}
        ok = keep >= 0.5
        return Check("E3", "economic", "execution delay / price artefact",
                     PASS if ok else FAIL, statistic=keep * 100, threshold=50.0,
                     detail=(f"{keep:.0%} of the edge survives on the tradable instrument "
                             f"({base:+.2%} -> {tr:+.2%})"
                             if ok else
                             f"only {keep:.0%} of the edge survives on the instrument a book "
                             f"would actually hold ({base:+.2%} -> {tr:+.2%}): the rest was in "
                             "prints nobody could have traded"),
                     evidence=ev)

    if horizon < 2:
        return Check("E3", "economic", "execution delay / price artefact", INCONCLUSIVE,
                     blocking=False, statistic=float(prof[1] / base),
                     detail=(f"a one-step edge on a non-tradable series cannot be separated from "
                             f"a stale-print artefact by delay alone (delay 1 keeps "
                             f"{prof[1] / base:.0%}). Supply `tradable_ret` -- an ETF, a future, "
                             "the contract itself -- and this becomes decisive"),
                     evidence=ev)

    keep1 = float(prof[1] / base)
    floor = 0.4 * (horizon - 1) / horizon
    ok = keep1 >= floor
    ev |= {"retained_at_delay_1": keep1, "overlap_floor": floor}
    trace = ", ".join(f"d{d}:{v:+.2%}" for d, v in prof.items() if np.isfinite(v))
    # Advisory when the price source was not declared. The profile itself is
    # worth having -- it used to be thrown away entirely, and on a weekly book
    # how much survives one step of delay is most of the question -- but a veto
    # is not justified: a fast decay on an undeclared series is either a stale
    # print or a genuinely short-lived edge, and if the series really is
    # tradable the second one is not a defect.
    return Check("E3", "economic", "execution delay / price artefact", PASS if ok else FAIL,
                 blocking=not _undecidable,
                 statistic=keep1, threshold=floor,
                 detail=(f"the edge decays gracefully with entry delay [{trace}]"
                         if ok else
                         f"one step of delay removes {1 - keep1:.0%} of the edge, far past the "
                         f"{1 - floor:.0%} the window overlap alone allows [{trace}] -- the edge "
                         "lives in the first print after the signal, which is what a stale "
                         "quote or a spread bouncing back looks like")
                 + ("  Without `tradable_ret` this cannot separate a stale print "
                    "from a genuinely short-lived edge; it says only that the edge "
                    "is in the first print, which either way is where a book "
                    "cannot reach it." if _undecidable else ""),
                 evidence=ev | {"undecidable_without_tradable_ret": _undecidable})


COST_COMPONENTS = ("commission", "spread", "impact", "tax", "borrow")
PRICE_CONVENTIONS = ("mid", "touch", "trade", "vwap")


def e4_cost_convention(price_convention: Optional[str] = None,
                       cost_components: Optional[Sequence[str]] = None) -> Check:
    """E4 -- check the cost assumption against the price it is added to.

    Executing at the touch already pays the spread. Adding a spread assumption
    on top charges it twice, which reads as conservatism and is simply wrong --
    and wrong in the direction that kills real results quietly, since nobody
    audits a strategy for being too pessimistic.

    The mirror failure is the same mistake with the sign flipped: costing a mid
    price with commission only, and never paying the spread at all.
    """
    if price_convention is None and cost_components is None:
        return Check("E4", "economic", "cost convention", NA, blocking=False,
                     detail="no price convention declared. State which price the returns are "
                            "measured at and what `cost_bp` is meant to contain -- charging the "
                            "spread twice and never charging it look identical from here")
    if price_convention is not None and price_convention not in PRICE_CONVENTIONS:
        return Check("E4", "economic", "cost convention", INCONCLUSIVE,
                     detail=f"unrecognised price convention {price_convention!r}")
    comps = {c.lower() for c in (cost_components or ())}
    unknown = comps - set(COST_COMPONENTS)
    if unknown:
        return Check("E4", "economic", "cost convention", INCONCLUSIVE,
                     detail=f"unrecognised cost component(s): {', '.join(sorted(unknown))}")
    if price_convention == "touch" and "spread" in comps:
        return Check("E4", "economic", "cost convention", FAIL,
                     detail="returns are measured at the touch, which already pays the spread, "
                            "and `cost_bp` includes a spread component as well -- the spread is "
                            "being charged twice",
                     evidence={"price_convention": price_convention, "components": sorted(comps)})
    if price_convention in ("mid", "trade", "vwap") and comps and "spread" not in comps:
        return Check("E4", "economic", "cost convention", FAIL,
                     detail=f"returns are measured at the {price_convention}, which does not pay "
                            "the spread, and `cost_bp` does not include one either -- the "
                            "strategy is crossing for free",
                     evidence={"price_convention": price_convention, "components": sorted(comps)})
    return Check("E4", "economic", "cost convention", PASS,
                 detail=f"{price_convention} prices with {', '.join(sorted(comps)) or 'no'} costs: "
                        "the spread is accounted for exactly once",
                 evidence={"price_convention": price_convention, "components": sorted(comps)})


def e5_capacity(signal: np.ndarray, mask: np.ndarray, dollar_volume: np.ndarray,
                q: float = 0.1, hold: int = 1, capital: Optional[float] = None,
                max_participation: float = 0.1, min_n: int = 30) -> Check:
    """E5 -- how much money fits before the edge is the trade.

    An edge expressed in basis points says nothing about whether it can be
    taken. A window that holds one percent of the day's volume will produce
    excellent per-trade numbers for a size nobody would bother running, and the
    honest form of the result is a capital figure rather than a return.

    Reports the capital at which the book would take ``max_participation`` of
    the traded value in its own names. With ``capital`` supplied it becomes a
    verdict on that size.
    """
    sig = np.asarray(signal, float)
    dv = np.asarray(dollar_volume, float)
    T, N = sig.shape
    caps = []
    for t in range(0, T - 1, max(1, hold)):
        m = mask[t] & np.isfinite(sig[t]) & np.isfinite(dv[t]) & (dv[t] > 0)
        idx = np.flatnonzero(m)
        if idx.size < min_n:
            continue
        k = max(1, int(idx.size * q))
        order = idx[np.argsort(sig[t, idx])]
        legs = np.concatenate([order[-k:], order[:k]])
        w = 1.0 / k                                  # equal weight within each leg
        # capital at which the smallest name in the book hits the participation cap
        caps.append(float(np.min(dv[t, legs]) * max_participation / w))
    if len(caps) < 5:
        return Check("E5", "economic", "capacity", INCONCLUSIVE, blocking=False,
                     detail="too few rebalances with usable volume")
    cap = float(np.median(caps))
    ev = {"capacity_median": cap, "capacity_p10": float(np.percentile(caps, 10)),
          "max_participation": max_participation, "n_rebalances": len(caps)}
    if capital is None:
        return Check("E5", "economic", "capacity", PASS, blocking=False,
                     statistic=cap,
                     detail=(f"at {max_participation:.0%} participation the book holds about "
                             f"{cap:,.0f} of currency (10th pct {ev['capacity_p10']:,.0f}). "
                             "State the size you intend to run and this becomes a verdict"),
                     evidence=ev)
    ok = capital <= cap
    ev["capital"] = float(capital)
    return Check("E5", "economic", "capacity", PASS if ok else FAIL,
                 statistic=cap, threshold=float(capital),
                 detail=(f"{capital:,.0f} fits: the book holds about {cap:,.0f} at "
                         f"{max_participation:.0%} participation"
                         if ok else
                         f"{capital:,.0f} does not fit. At {max_participation:.0%} participation "
                         f"this book holds about {cap:,.0f}, so the stated size would be a large "
                         "share of the volume in its own names and the edge becomes the trade"),
                 evidence=ev)

def _leg_aware_book(signal: np.ndarray, ret: np.ndarray, mask: np.ndarray,
                    buyable: Optional[np.ndarray], shortable: Optional[np.ndarray],
                    q: float, hold: int, long_short: bool, min_n: int) -> Dict[str, np.ndarray]:
    """A quantile book where the two legs face different constraints.

    Which positions are available is not one fact. A board locked up cannot be
    bought and can be sold; a board locked down cannot be shorted and can be
    covered. Applying one mask to both legs quietly removes the names the short
    leg most wanted, which flatters the constrained book -- the first version of
    this check did exactly that and reported a residual edge that was an
    artifact of its own simplification.
    """
    sig, r = np.asarray(signal, float), np.asarray(ret, float)
    T, N = sig.shape
    w = np.zeros(N)
    gross, turn = np.full(T, np.nan), np.zeros(T)
    rebalances, blocked, wanted = [], 0, 0
    last = -(10 ** 9)
    for t in range(T - 1):
        m = np.isfinite(sig[t]) & mask[t]
        idx = np.flatnonzero(m)
        if idx.size >= min_n and (t - last) >= hold:
            k = max(1, int(idx.size * q))
            order = idx[np.argsort(sig[t, idx])]
            # Fill k slots from the top of the ranking, skipping what cannot be
            # taken and going further down for the rest. That is what a desk
            # does, and it matters: re-weighting the survivors instead would
            # lever the book up exactly when half its picks were unavailable,
            # and the leverage would show up as a residual edge that is an
            # artifact of the accounting. The first version of this did that.
            def _fill(cands, avail):
                out, miss = [], 0
                for j in cands:
                    if avail is not None and not avail[t, j]:
                        miss += 1
                        continue
                    out.append(j)
                    if len(out) == k:
                        break
                return np.array(out, dtype=int), miss

            longs, miss_l = _fill(order[::-1], buyable)
            shorts, miss_s = (_fill(order, shortable) if long_short
                              else (np.array([], dtype=int), 0))
            wanted += 2 * k if long_short else k
            blocked += miss_l + miss_s
            new = np.zeros(N)
            if longs.size:
                new[longs] = 1.0 / k
            if shorts.size:
                new[shorts] = -1.0 / k
            turn[t] = float(np.abs(new - w).sum() / 2.0)
            w = new
            last = t
            rebalances.append(t)
        gross[t] = float(w @ np.nan_to_num(r[t + 1], nan=0.0))
    return {"gross": gross, "turnover": turn, "blocked": blocked, "wanted": wanted,
            "rebalances": np.array(rebalances, dtype=int)}


def e6_entry_constraints(signal: np.ndarray, ret: np.ndarray, mask: np.ndarray,
                         q: float = 0.1, hold: int = 1, long_short: bool = True,
                         cost_bp: float = 0.0, dollar_volume: Optional[np.ndarray] = None,
                         min_dollar_volume: Optional[float] = None,
                         price_limit: Optional[float] = 0.0995,
                         max_lost: float = 0.5, min_n: int = 30) -> "Check":
    """E6 -- price the constraints on *getting in*, one at a time.

    E1 asks whether the edge survives its turnover cost. This asks the question
    that kills more of these than cost does: whether the book could have taken
    the positions at all. A published decomposition of one event-driven book put
    the annual cost of "cannot buy what is locked limit-up" at 10.8 percentage
    points against a final 13.3 -- the largest single line, bigger than cost,
    the liquidity floor and the position cap together.

    That constraint is invisible in the usual setup because it is not a price,
    it is an absence. A signal fires on the names that just jumped, those are
    exactly the names sitting at the limit, the backtest fills them at a price
    nobody could have paid, and every downstream number is then computed on a
    book that was never available. Nothing in the statistics objects, because
    nothing in the statistics is wrong.

    The two legs face different constraints and are treated separately: a board
    locked up cannot be bought, a board locked down cannot be shorted.

    Three outcomes, and the middle one is the common case:

      the edge is gone once the constraints bind          REJECTED
      the edge survives but most of it was unavailable    INCONCLUSIVE -- what
        was measured is not what could have been held, so the verdict on the
        measured number does not transfer. Re-state the claim at the size that
        remains and put it through the battery again; a residual that passed
        nothing at its own size has not been judged.
      the constraints barely bind                         PASS

    Only the mechanical constraints are priced. A per-name weight cap does not
    bind on an equal-weight quantile book, and execution delay is E3's question;
    both are said rather than silently folded in.
    """
    from .verdict import FAIL, INCONCLUSIVE, NA, PASS, Check

    sig, r = np.asarray(signal, float), np.asarray(ret, float)
    base_mask = np.asarray(mask, bool)
    ok = np.ones_like(base_mask, dtype=bool)

    def _net(m, buyable, shortable):
        pf = _leg_aware_book(sig, r, m, buyable, shortable, q, hold, long_short, min_n)
        g = np.asarray(pf["gross"], float)
        net = g - np.asarray(pf["turnover"], float) * (cost_bp / 1e4)
        net = net[np.isfinite(net)]
        return (float(net.mean() * 1e4) if net.size else np.nan), pf

    base, _ = _net(base_mask, None, None)
    if not np.isfinite(base):
        return Check("E6", "economic", "entry constraints", INCONCLUSIVE, blocking=False,
                     detail="the unconstrained book is not measurable on this panel")

    rows: Dict[str, float] = {}
    buyable = shortable = None
    m = base_mask
    blocked = wanted = 0

    if price_limit:
        rr = np.nan_to_num(r, nan=0.0)
        up, down = rr >= float(price_limit), rr <= -float(price_limit)
        if up.any() or down.any():
            buyable, shortable = ~up, ~down
            after, pf = _net(m, buyable, shortable)
            rows["cannot enter a locked board"] = base - after
            blocked, wanted = pf["blocked"], pf["wanted"]
        else:
            rows["cannot enter a locked board"] = np.nan

    if dollar_volume is not None and min_dollar_volume:
        before, _ = _net(m, buyable, shortable)
        m = m & (np.nan_to_num(np.asarray(dollar_volume, float), nan=0.0)
                 >= float(min_dollar_volume))
        after, _ = _net(m, buyable, shortable)
        rows["liquidity floor"] = before - after
    else:
        rows["liquidity floor"] = np.nan

    priced = {k: v for k, v in rows.items() if np.isfinite(v)}
    if not priced:
        return Check("E6", "economic", "entry constraints", NA, blocking=False,
                     detail="no entry constraint could be priced -- no locked prints in this "
                            "panel and no liquidity floor declared. If the market this claim is "
                            "about has daily price limits, the book has not been shown to be "
                            "one that could have been entered")

    final, _ = _net(m, buyable, shortable)
    lost = (base - final) / abs(base) if base else np.nan
    ev = {"unconstrained_bp": base, "constrained_bp": final, "share_lost": lost,
          "marginal_bp": priced, "blocked_entries": blocked, "wanted_entries": wanted}
    table = "; ".join(f"{k}: {v:+.2f}bp/step" for k, v in priced.items())
    share = (f" {blocked}/{wanted} ({blocked / wanted:.1%}) of the positions the book wanted "
             f"were unavailable." if wanted else "")

    if base > 0 and final <= 0:
        return Check("E6", "economic", "entry constraints", FAIL,
                     statistic=final, threshold=0.0, evidence=ev,
                     detail=(f"{base:+.2f}bp/step as measured, {final:+.2f}bp/step once the book "
                             f"can only take the positions it could have taken -- {table}.{share} "
                             f"The constraints are not a haircut on this result, they are part of "
                             f"what it means, and it does not survive them"))
    if base > 0 and np.isfinite(lost) and lost > max_lost:
        return Check("E6", "economic", "entry constraints", INCONCLUSIVE,
                     statistic=final, threshold=0.0, evidence=ev,
                     detail=(f"{lost:.0%} of the edge was in positions the book could not have "
                             f"taken: {base:+.2f}bp/step as measured, {final:+.2f}bp/step "
                             f"available -- {table}.{share} The remainder may well be real, but "
                             f"it is a different claim and nothing here has judged it at that "
                             f"size. Put the constraints in the mask and run the battery again"))
    return Check("E6", "economic", "entry constraints", PASS, blocking=False,
                 statistic=final, threshold=0.0, evidence=ev,
                 detail=(f"{base:+.2f}bp/step -> {final:+.2f}bp/step under the entry constraints "
                         f"({table}).{share} A per-name weight cap does not bind on an "
                         f"equal-weight book and was not simulated; execution delay is E3"))
