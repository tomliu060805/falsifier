"""Charts, written to PNG on disk and nowhere else.

Two rules, both learned the expensive way.

Fonts are registered explicitly and a missing CJK face raises rather than
falling back. Matplotlib's cache does not pick up a font just because it is
installed, and the failure mode is silent: every Chinese label renders as an
empty box, the figure looks finished, and nobody notices until it is in a
report. Raising is the cheaper outcome.

And the output is a file path. No HTML, no JSON sidecar, no hosted artifact --
a chart that lives somewhere other than the project directory is a chart that
will be out of date the first time the data moves.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

_CJK = re.compile(r"[一-鿿]")
# Searched, not hardcoded: a user-installed font lives under $HOME and a
# hardcoded system path finds nothing there, which is the same silent failure
# this function exists to prevent.
_FONT_DIRS = ("~/.fonts", "~/.local/share/fonts", "/usr/share/fonts")
_CJK_PATTERN = re.compile(r"(cjk|notosanssc|sourcehan|wqy|msyh|simhei)", re.I)


def _find_cjk_fonts():
    found = []
    for d in _FONT_DIRS:
        root = Path(d).expanduser()
        if not root.exists():
            continue
        for f in root.rglob("*"):
            if f.suffix.lower() in (".otf", ".ttf", ".ttc") and _CJK_PATTERN.search(f.name):
                found.append(f)
    return sorted(found, key=lambda f: ("serif" in f.name.lower(), f.name))


def _prepare(texts: Sequence[str]):
    try:
        import matplotlib
    except ImportError as exc:                       # pragma: no cover - install path
        raise ImportError(
            "drawing needs matplotlib, which is an optional extra here because the "
            "checks themselves draw nothing: pip install 'falsifier[charts]'"
        ) from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm

    needs_cjk = any(_CJK.search(t or "") for t in texts)
    if needs_cjk:
        picked = None
        for f in _find_cjk_fonts():
            try:
                fm.fontManager.addfont(str(f))        # the cache will not do this for you
                picked = fm.FontProperties(fname=str(f)).get_name()
                break
            except Exception:
                continue
        if picked is None:
            raise RuntimeError(
                "the labels contain Chinese and no CJK font is registered. Install "
                "Noto Sans CJK or pass English labels -- rendering would produce "
                "boxes, and a figure full of boxes looks finished."
            )
        plt.rcParams["font.sans-serif"] = [picked, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def plot_increment(inc, path: str, title: Optional[str] = None, dpi: int = 140) -> str:
    """Two panels: what the curve did, and what each year did.

    The second panel is the one that decides things. A single improved curve is
    compatible with an addition that bought its gain in one year and made the
    bad years worse, and the only way to see that is per year, signed.
    """
    labels = [title or "", inc.label_baseline, inc.label_combined]
    plt = _prepare(labels)

    c = inc.curves()
    rows = inc.yearly()
    s = inc.summary()
    x = np.arange(len(c["nav_baseline"]))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 8), height_ratios=[2, 1.1],
        gridspec_kw={"hspace": 0.34})

    from .increment import _years
    yr_of = _years(inc.dates)
    ticks = [0] + [i for i in range(1, len(yr_of)) if yr_of[i] != yr_of[i - 1]]

    ax1.plot(x, c["nav_baseline"], lw=1.4, color="#8c8c8c", label=inc.label_baseline)
    ax1.plot(x, c["nav_combined"], lw=1.6, color="#1f77b4", label=inc.label_combined)
    ax1.axhline(1.0, lw=0.8, color="#cccccc", zorder=0)
    ax1.set_ylabel("cumulative excess (compounded)")
    ax1.set_xticks(ticks)
    ax1.set_xticklabels([yr_of[i] for i in ticks], rotation=45)
    ax1.set_xlim(0, len(x) - 1)
    ax1.legend(loc="upper left", frameon=False)
    ax1.set_title(title or "Incremental contribution", loc="left", fontsize=12)

    axd = ax1.twinx()
    axd.plot(x, c["nav_delta"], lw=1.0, ls="--", color="#d62728", alpha=0.75)
    axd.set_ylabel("combined / baseline", color="#d62728")
    axd.tick_params(axis="y", colors="#d62728")

    yrs = [r["year"] for r in rows]
    deltas = np.array([r["delta"] for r in rows])
    ax2.bar(yrs, deltas * 100,
            color=["#2ca02c" if d > 0 else "#d62728" for d in deltas], width=0.62)
    ax2.axhline(0, lw=0.8, color="#444444")
    ax2.set_ylabel("delta, % of excess")
    ax2.tick_params(axis="x", rotation=45)
    # A year the combined book *lost* in is marked on the tick label rather than
    # on the bar: the bar shows the delta, and a year can improve and still be a
    # losing year. Putting the mark on the bar hides it behind the bar whenever
    # the delta is negative, which is exactly when it matters.
    for lbl, r in zip(ax2.get_xticklabels(), rows):
        if r["combined"] < 0:
            lbl.set_color("#d62728")
            lbl.set_fontweight("bold")


    foot = (f"red year = the combined book lost that year\n"
            f"annualised {s['ann_baseline']:+.2%} -> {s['ann_combined']:+.2%} "
            f"({s['ann_delta']:+.2%})   years improved {s['years_improved']}/{s['years']}   "
            f"losing years {s['negative_years_baseline']} -> {s['negative_years_combined']}")
    tm = inc.per_trade().get(inc.label_combined, {})
    if tm.get("n_trades"):
        foot += (f"\nper trade: n={tm['n_trades']} win={tm['win_rate']:.1%} "
                 f"avg={tm['avg_bp']:+.1f}bp payoff={tm['payoff_ratio']:.2f} "
                 f"PF={tm['profit_factor']:.2f} worst={tm['worst_bp']:.0f}bp")
    fig.text(0.012, 0.012, foot, fontsize=8.5, color="#444444", va="bottom")

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(out)
