"""The self-check is the test suite: if the referee cannot get the known cases
right, nothing it says about an unknown one is worth reading."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples"))

import falsifier as F  # noqa: E402
import selfcheck as S  # noqa: E402


@pytest.fixture(scope="module")
def panel():
    return S.build_panel()


@pytest.mark.parametrize("horizon", [1, 5, 20])
@pytest.mark.parametrize("offset,expect,boundary", [
    (0, "PASS", "+0->+1"),
    (1, "FAIL", "-1->+0"),
    (2, "FAIL", "-2->-1"),
])
def test_a2_locates_the_boundary(panel, horizon, offset, expect, boundary):
    ret, mask, _, _ = panel
    fwd = F.forward_returns(ret, horizon)
    c = F.a2_feature_shift(S.trailing_mean(ret, S.WINDOW, end_offset=offset), fwd, mask)
    assert c.outcome == expect
    assert c.evidence["boundary_at"] == boundary


@pytest.mark.parametrize("horizon", [1, 5, 20])
def test_a2_declines_to_guess(panel, horizon):
    """No boundary to find must read as INCONCLUSIVE, never as a leak."""
    ret, mask, z, _ = panel
    fwd = F.forward_returns(ret, horizon)
    g = np.random.default_rng(7)
    for sig in (g.standard_normal(ret.shape), np.repeat(z[None, :], len(ret), 0)):
        assert F.a2_feature_shift(sig, fwd, mask).outcome == "INCONCLUSIVE"


def test_prereg_is_write_once(tmp_path):
    pre = S.survivor_prereg()
    path = tmp_path / "prereg.json"
    pre.freeze(str(path))
    with pytest.raises(FileExistsError):
        pre.freeze(str(path))
    assert F.Prereg.load(str(path)).id == pre.id


def test_seal_refuses_a_second_configuration(tmp_path):
    seal = F.SealedSplit(train_end=20200101, valid_end=20230101,
                         ledger=str(tmp_path / "ledger.json"))
    dates = np.array([20190101, 20210101, 20240101])
    with pytest.raises(F.SealError):
        seal.mask_for(dates, "test")
    seal.unseal({"spec": "v1"}, reason="frozen")
    seal.unseal({"spec": "v1"}, reason="re-read same spec")   # allowed, counted
    with pytest.raises(F.SealError):
        seal.unseal({"spec": "v2"}, reason="tuned after looking")


@pytest.mark.slow
def test_full_gauntlet_on_all_targets():
    assert S.main(n_draws=80) == 0


@pytest.mark.slow
def test_matched_null_false_positive_rate(panel):
    """SM1 must reject skill-free books at close to its nominal rate.

    A single blind book landing above the threshold proves nothing either way.
    What has to hold is the rate: books drawn from the null distribution should
    clear the 95th percentile roughly one time in twenty. Much more often and
    the null is too easy; much less and it is throttling real findings.
    """
    ret, mask, _, _ = panel
    T_, N_ = ret.shape
    pcts = []
    for k in range(30):
        g = np.random.default_rng(500 + k)
        score = g.standard_normal((T_, N_))
        sel = np.zeros((T_, N_), bool)
        cur = None
        for t in range(T_):
            if t % 20 == 0 and mask[t].sum() > 25:
                idx = np.flatnonzero(mask[t])
                cur = idx[np.argsort(score[t, idx])[-25:]]
            if cur is not None:
                sel[t, cur] = True
        study = F.StrategyStudy(claim="blind", selection=sel, ret=ret, mask=mask, cost_bp=2.0)
        rep = F.run_strategy(study, n_draws=100, seed=k, metric="sharpe", verbose=False)
        pcts.append(next(c for c in rep.checks if c.id == "SM1").statistic)
    rate = float(np.mean(np.array(pcts) >= 95.0))
    assert rate <= 0.20, f"SM1 clears skill-free books {rate:.0%} of the time (nominal 5%)"
    assert np.median(pcts) < 80.0, f"median percentile of skill-free books is {np.median(pcts):.0f}"


# --------------------------------------------------------------------------
# priors + taxonomy
# --------------------------------------------------------------------------

def test_taxonomy_is_consistent():
    from falsifier import taxonomy as T
    assert len({m.id for m in T.MODES}) == len(T.MODES), "duplicate mode id"
    known = {"P0", "P1", "P2", "P3", "P4", "P5", "P6",
             "A0", "A1", "A2", "A3",
             "S4", "S5", "S6", "S7", "S8", "S9",
             "M0", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9", "M10", "M11",
             "E1", "E2", "E3", "E4", "E5", "I1", "I2",
             "SM0", "SM1", "SE1", "SE2"}
    for m in T.MODES:
        assert m.family in T.FAMILIES
        assert set(m.caught_by) <= known, f"{m.id} names an unknown check"
        assert m.looks_like and m.why, f"{m.id} is missing its description"
    # The uncovered list was the roadmap while it had entries. Now that it is
    # empty, what has to be guarded is the claim that empty means something:
    # every check a mode points at must actually carry a veto somewhere, and
    # every veto-carrying check must have a target built to trip it. Without
    # that, "44 of 44 covered" is a mapping exercise.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import selfcheck as SC

    claimed = {c for m in T.MODES for c in m.caught_by}
    advisory = {"M2", "M4", "M8", "M9", "E2", "SE2", "SM0", "S6", "A3", "P1"}  # informative, no veto
    # M11 is the borderline one: it carries a veto only where the claim was
    # asserted to hold at another cadence, so it is a target's cause of death
    # (frequency_flip declares `claimed_strides`) and advisory everywhere else.
    # Checks that carry a veto but are not any target's *primary* cause of
    # death. Each needs a reason, so that adding to this set is a decision
    # rather than a way to make the test go quiet.
    corroborating = {
        "M3": "fires as a secondary killer on size_proxy; M1 is the primary there",
        "S4": "rejects on weak significance, which every deliberately-broken target "
              "fails earlier and every honest one passes",
        "SE1": "portfolio-vs-benchmark, exercised by a replayed real study rather "
               "than by a synthetic target",
        "P2": "the seal is enforced where it can be -- SealedSplit.unseal raises, see "
              "test_seal_refuses_a_second_configuration. The gauntlet check only "
              "reports whether the seal is still intact",
    }
    declared = SC.declared_killers()
    for cid in sorted(claimed - advisory):
        assert cid in declared or cid in corroborating, (
            f"{cid} is claimed to catch a failure mode, but no target names it as the "
            "cause of death and it is not listed as corroborating")


def test_validate_rejects_a_record_that_cannot_be_keyed(tmp_path):
    """Keying `killed_by` to the taxonomy is the point: a record that cannot be
    keyed means either the record is vague or the taxonomy is short a mode."""
    import json
    from falsifier import priors as PR

    recs = [
        dict(id="ok", claim="c", verdict="REJECTED", killed_by=["cost-eats-it"],
             evidence="break-even 1.3bp", lesson="report break-even cost, not IC",
             applies_to=["x"], project="p"),
        dict(id="unkeyed", claim="c", verdict="REJECTED", killed_by=[],
             evidence="it did not work", lesson="a lesson long enough to pass",
             applies_to=["x"], project="p"),
        dict(id="invented", claim="c", verdict="REJECTED", killed_by=["not-a-real-mode"],
             evidence="it did not work", lesson="a lesson long enough to pass",
             applies_to=["x"], project="p"),
    ]
    f = tmp_path / "v.jsonl"
    f.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs), encoding="utf-8")

    problems = PR.validate(path=str(tmp_path))
    assert problems["no_cause"] == ["unkeyed"]
    assert any("not-a-real-mode" in x for x in problems["unknown_mode"])
    assert "ok" not in str(problems)


def test_prior_roundtrip_and_search(tmp_path):
    import json
    from falsifier import priors as PR
    recs = [
        dict(id="a", claim="momentum rotation over ETFs beats holding the index",
             verdict="REJECTED", killed_by=["null-turnover-unmatched"],
             evidence="21st percentile", lesson="match the null on turnover",
             applies_to=["rotation", "ETF"], project="p"),
        dict(id="b", claim="一个图神经网络学出来的邻接矩阵能预测波动率",
             verdict="REJECTED", killed_by=["null-random-structure-wins"],
             evidence="random adjacency wins", lesson="比随机邻接",
             applies_to=["图网络"], project="q"),
    ]
    f = tmp_path / "v.jsonl"
    f.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs), encoding="utf-8")

    ps = PR.load(str(tmp_path))
    assert len(ps) == 2

    idx = PR.Index(ps)
    en = idx.search("rotation strategy over ETFs, monthly rebalance", k=2)
    assert en and en[0][1].id == "a"
    zh = idx.search("用图神经网络学邻接矩阵做预测", k=2)
    assert zh and zh[0][1].id == "b", "CJK bigram retrieval failed"

    modes = PR.checklist(en)
    assert modes and modes[0].id == "null-turnover-unmatched"
    assert "PRIOR CHECK" in PR.render("q", en)
    assert "no precedent" in PR.render("q", []).lower()


def test_rolling_fit_supplies_both_callables(panel):
    """A clean rolling fit must reproduce itself from truncated history and
    score nothing on destroyed labels."""
    ret, mask, _, _ = panel
    T_, N_ = ret.shape
    r0 = np.nan_to_num(ret, nan=0.0)
    c = np.cumsum(np.vstack([np.zeros((1, N_)), r0]), 0)
    feat = np.stack([np.vstack([np.full((w - 1, N_), np.nan), (c[w:] - c[:-w]) / w])
                     for w in (2, 5, 10)], axis=2)
    m = mask & np.isfinite(feat).all(axis=2)
    rf = F.RollingFit(feat, ret, m, horizon=5, fitwin=200, stride=25, min_n=50)
    pred = rf.predict()
    assert np.isfinite(pred).any()
    a0 = F.a0_truncation(pred, rf.recompute_at, rf.probe_dates(3))
    assert a0.outcome == "PASS", a0.detail


@pytest.mark.slow
def test_a0_catches_a_full_sample_statistic(panel):
    ret, mask, _, _ = panel
    T_, N_ = ret.shape
    r0 = np.nan_to_num(ret, nan=0.0)
    c = np.cumsum(np.vstack([np.zeros((1, N_)), r0]), 0)
    feat = np.stack([np.vstack([np.full((w - 1, N_), np.nan), (c[w:] - c[:-w]) / w])
                     for w in (2, 5, 10)], axis=2)
    m = mask & np.isfinite(feat).all(axis=2)
    rf = F.RollingFit(feat, ret, m, horizon=5, fitwin=200, stride=25, min_n=50,
                      standardize="full-sample")
    a0 = F.a0_truncation(rf.predict(), rf.recompute_at, rf.probe_dates(3))
    assert a0.outcome == "FAIL", "A0 missed a scaling constant fitted on the whole sample"


# --------------------------------------------------------------------------
# S5 / S6 / S7
# --------------------------------------------------------------------------

def test_s7_separates_static_from_frozen(panel):
    """A characteristic that never moves is not a feed that died."""
    ret, _, z, _ = panel
    T_, N_ = ret.shape
    static = np.repeat(z[None, :], T_, axis=0)
    frozen = ret.copy()
    frozen[700:] = frozen[699]
    encoded = ret.copy()
    encoded[200:400] = 0.0                      # a filter that began matching nothing

    assert F.s7_input_staleness({"ret": ret}).outcome == "PASS"
    assert F.s7_input_staleness({"size": static}).outcome == "NA"
    assert F.s7_input_staleness({"ret": ret, "size": static}).outcome == "PASS"
    for bad in (frozen, encoded):
        c = F.s7_input_staleness({"x": bad})
        assert c.outcome == "FAIL" and c.statistic >= 190


def test_s6_distinguishes_no_signal_from_no_power(panel):
    ret, mask, _, _ = panel
    live = F.forward_returns(ret, 5)
    assert F.s6_positive_control(F.default_control(ret), live, mask, horizon=5).outcome == "PASS"

    dead = np.random.default_rng(3).standard_normal(ret.shape) * 0.02
    c = F.s6_positive_control(F.default_control(dead), F.forward_returns(dead, 5),
                              np.isfinite(dead), horizon=5)
    assert c.outcome == "FAIL"
    assert "uninterpretable" in c.detail


def test_s5_flags_a_picked_seed():
    stable = lambda s: float(np.random.default_rng(s).normal(0.5, 0.15))
    assert F.s5_seed_stability(stable, n_seeds=40, seed=0).outcome == "PASS"

    noisy = lambda s: float(np.random.default_rng(s).normal(-0.1, 0.4))
    assert F.s5_seed_stability(noisy, n_seeds=40, seed=0).outcome == "FAIL"

    # Even a genuinely positive distribution fails when the number being
    # claimed is drawn from its top decile.
    picked = F.s5_seed_stability(stable, n_seeds=40, seed=0,
                                 reported=max(stable(s) for s in range(40)) + 0.2)
    assert picked.outcome == "FAIL" and "picked" in picked.detail


def test_dead_panel_is_inconclusive_not_rejected(panel):
    """The distinction S6 exists for, end to end."""
    ret, mask, _, _ = panel
    g = np.random.default_rng(11)
    dead = g.standard_normal(ret.shape) * 0.02
    rep = F.run(F.Study(claim="nothing, on a panel with nothing in it",
                        signal=g.standard_normal(ret.shape), ret=dead, mask=mask,
                        horizon=5, covariates={"vol": np.abs(dead)}),
                n_draws=40, seed=0, verbose=False)
    assert rep.outcome == "INCONCLUSIVE", rep.render()
    assert not rep.killers, "a panel with no demonstrated power must not produce a killer"


# --------------------------------------------------------------------------
# E3 / S8 / P3
# --------------------------------------------------------------------------

def test_e3_separates_bounce_from_a_real_edge(panel):
    """A2 and E3 catch different halves of the price-artefact family."""
    ret, mask, _, _ = panel
    T_, N_ = ret.shape
    g = np.random.default_rng(5)
    fair = g.standard_normal((T_, N_)) * 0.012
    side = g.choice([-1.0, 1.0], size=(T_, N_))
    obs = fair + 0.004 * (side - np.vstack([side[:1], side[:-1]]))

    c = F.e3_execution_delay(-obs, obs, mask, horizon=1, tradable_ret=fair,
                             price_source="synthetic")
    assert c.outcome == "FAIL" and c.statistic < 50
    # A2 does not see it: the signal never overlaps its own label.
    a2 = F.a2_feature_shift(-obs, F.forward_returns(obs, 1), mask)
    assert a2.outcome == "PASS"

    # And a genuinely tradable edge is not flagged.
    honest = np.full_like(ret, np.nan)
    honest[1:] = ret[:-1]
    assert F.e3_execution_delay(honest, ret, mask, horizon=1,
                                tradable_ret=ret, price_source="index").outcome != "FAIL"


def test_e3_is_off_unless_the_source_is_declared(panel):
    ret, mask, _, _ = panel
    c = F.e3_execution_delay(ret, ret, mask, horizon=1)
    assert c.outcome == "NA" and "price_source" in c.detail


def test_s8_sees_the_overfitting_signature():
    def diverging(k, seg):
        return float(k) if seg == "train" else float(-k)

    def flat(k, seg):
        return 0.1 + 0.001 * float(k)

    params = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
    assert F.s8_knob_monotonicity(diverging, params).outcome == "FAIL"
    assert F.s8_knob_monotonicity(flat, params).outcome == "PASS"
    assert F.s8_knob_monotonicity(diverging, (0.0, 1.0)).outcome == "INCONCLUSIVE"


def test_p3_binds_the_frozen_file(tmp_path):
    import json
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"q_low": 0.33, "window": 5}), encoding="utf-8")

    assert F.p3_frozen_config({"q_low": 0.33, "window": 5}, str(path)).outcome == "PASS"
    drifted = F.p3_frozen_config({"q_low": 0.3412, "window": 5}, str(path))
    assert drifted.outcome == "FAIL" and "q_low" in drifted.detail
    assert F.p3_frozen_config({"window": 5}, str(path)).outcome == "FAIL"
    assert F.p3_frozen_config({}, "").outcome == "NA"


# --------------------------------------------------------------------------
# M5 / M6 / S9
# --------------------------------------------------------------------------

def test_m5_threshold_sits_between_real_prints_and_the_incident(panel):
    """Calibrated, not guessed: clean one-minute index data runs about 0.05%
    spike-and-revert, and the incident this was written from ran about 1%."""
    ret, _, _, _ = panel
    assert F.m5_print_quality(ret).outcome == "PASS"

    g = np.random.default_rng(0)
    spiked = ret.copy()
    sd = np.nanstd(ret)
    T_, N_ = ret.shape
    for i in g.choice(np.arange(1, T_ - 1), size=int(T_ * 0.01), replace=False):
        j = g.choice(np.arange(N_), size=max(1, N_ // 3), replace=False)
        jump = sd * g.uniform(8, 40) * g.choice([-1.0, 1.0])
        spiked[i, j] = jump
        spiked[i + 1, j] = -jump * 0.95
    bad = F.m5_print_quality(spiked)
    assert bad.outcome == "FAIL" and bad.statistic > 0.2


def test_m6_catches_an_input_that_ends_early(panel):
    ret, _, _, _ = panel
    early = np.abs(ret).copy()
    early[-40:] = np.nan
    assert F.m6_input_freshness({"ret": ret, "vol": np.abs(ret)}).outcome == "PASS"
    c = F.m6_input_freshness({"ret": ret, "vol": early})
    assert c.outcome == "FAIL" and c.statistic >= 40


def test_s9_threshold_scales_with_the_overlap_floor(panel):
    """Persistence is bounded at 1, so the margin has to be a share of the
    headroom above the floor. A constant added to it made the check unable to
    fire at all past h=5."""
    ret, mask, _, _ = panel
    for h in (1, 5, 20):
        c = F.s9_label_persistence(F.forward_returns(ret, h), mask, horizon=h)
        assert c.outcome == "PASS", (h, c.detail)
        assert c.threshold < 1.0, f"h={h} threshold {c.threshold} is unreachable"

    g = np.random.default_rng(0)
    T_, N_ = ret.shape
    theta = g.standard_normal(N_) * 0.004
    sticky = theta[None, :] + g.standard_normal((T_, N_)) * 0.004
    c = F.s9_label_persistence(F.forward_returns(sticky, 5), np.ones((T_, N_), bool), horizon=5)
    assert c.outcome == "FAIL" and c.statistic > 0.9


# --------------------------------------------------------------------------
# M7 / M8 / P4 / P5 / E4 / E5
# --------------------------------------------------------------------------

def test_m8_tells_a_threshold_from_a_gradient(panel):
    """Advisory, so it has no self-check target and is pinned here instead."""
    ret, mask, _, _ = panel
    T_, N_ = ret.shape
    g = np.random.default_rng(2)
    fwd = F.forward_returns(ret, 1)

    graded = np.tile(np.linspace(-1, 1, N_), (T_, 1)) + g.standard_normal((T_, N_)) * 0.1
    lift = np.where(graded > 0.9, 0.02, 0.0)
    assert F.m8_threshold_or_slope(graded, graded * 0.01 + g.standard_normal((T_, N_)) * 0.001,
                                   mask).outcome == "PASS"
    c = F.m8_threshold_or_slope(graded, lift + g.standard_normal((T_, N_)) * 0.0005, mask)
    assert c.outcome == "FAIL" and "threshold, not a gradient" in c.detail


def test_p4_requires_the_convention_to_be_stated():
    assert F.p4_fill_convention().outcome == "NA"
    assert F.p4_fill_convention("next-open").outcome == "PASS"
    assert F.p4_fill_convention("same-close").outcome == "FAIL"
    assert F.p4_fill_convention("next-open", bar_includes_signal_period=True).outcome == "FAIL"
    assert F.p4_fill_convention("nonsense").outcome == "INCONCLUSIVE"


def test_p6_falsifies_the_story_without_falsifying_the_number():
    pre = F.Prereg(claim="c", mechanism="slow diffusion",
                   implications=["stronger where coverage is thin", "decays with horizon"])
    assert F.p6_mechanism_implications(None).outcome == "NA"
    assert F.p6_mechanism_implications(pre).outcome == "INCONCLUSIVE"
    held = {i: "held" for i in pre.implications}
    assert F.p6_mechanism_implications(pre, held).outcome == "PASS"
    broken = dict(held); broken["stronger where coverage is thin"] = "failed"
    c = F.p6_mechanism_implications(pre, broken)
    assert c.outcome == "FAIL" and "not the one producing it" in c.detail


def test_p5_checks_against_something_the_pipeline_never_saw():
    assert F.p5_external_facts().outcome == "NA"
    ok = [{"what": "limit-down count", "expected": 3000, "observed": 2980, "tol": 0.05}]
    assert F.p5_external_facts(ok).outcome == "PASS"
    bad = [{"what": "limit-down count", "expected": 3000, "observed": 964, "tol": 0.05}]
    c = F.p5_external_facts(bad)
    assert c.outcome == "FAIL" and "964" in c.detail


def test_e4_catches_the_spread_both_ways():
    assert F.e4_cost_convention().outcome == "NA"
    assert F.e4_cost_convention("touch", ["commission"]).outcome == "PASS"
    twice = F.e4_cost_convention("touch", ["commission", "spread"])
    assert twice.outcome == "FAIL" and "twice" in twice.detail
    free = F.e4_cost_convention("mid", ["commission"])
    assert free.outcome == "FAIL" and "crossing for free" in free.detail


def test_e5_reports_capacity_and_judges_a_stated_size(panel):
    ret, mask, _, _ = panel
    T_, N_ = ret.shape
    sig = np.random.default_rng(4).standard_normal((T_, N_))
    dv = np.full((T_, N_), 1e6)
    open_ended = F.e5_capacity(sig, mask, dv, q=0.1, hold=5)
    assert open_ended.outcome == "PASS" and open_ended.statistic > 0
    assert F.e5_capacity(sig, mask, dv, q=0.1, hold=5, capital=1e10).outcome == "FAIL"
    assert F.e5_capacity(sig, mask, dv, q=0.1, hold=5, capital=1e4).outcome == "PASS"


def test_m7_rejects_a_trigger_list_built_from_the_answer(panel):
    ret, mask, _, _ = panel
    T_, N_ = ret.shape
    fwd = F.forward_returns(ret, 5)
    g = np.random.default_rng(6)
    # Three trigger lists: one chosen by the outcome, one chosen by information
    # available when it fired, one chosen at random. Only the middle one is a
    # detector, and the third failing is the check working -- a trigger list no
    # better than random events at the same rate is not finding anything.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
    import selfcheck as SC

    informed_sig = SC.trailing_mean(ret, SC.WINDOW, end_offset=0)
    filtered = np.zeros((T_, N_), bool)
    informed = np.zeros((T_, N_), bool)
    at_random = np.zeros((T_, N_), bool)
    for t in range(SC.WINDOW + 4, T_ - 6):
        alive = np.flatnonzero(mask[t] & np.isfinite(fwd[t]) & np.isfinite(informed_sig[t]))
        if alive.size < 10:
            continue
        good = alive[fwd[t, alive] > 0]
        if good.size >= 5:
            filtered[t, g.choice(good, size=5, replace=False)] = True
        informed[t, alive[np.argsort(informed_sig[t, alive])[-5:]]] = True
        at_random[t, g.choice(alive, size=5, replace=False)] = True

    picked = F.m7_event_integrity(filtered, fwd, mask, n_draws=40)
    assert picked.outcome == "FAIL" and "outcome it claims to predict" in picked.detail
    assert F.m7_event_integrity(informed, fwd, mask, n_draws=40).outcome == "PASS"
    assert F.m7_event_integrity(at_random, fwd, mask, n_draws=40).outcome == "FAIL"


def test_m6_allows_a_signal_its_own_cadence_and_horizon(panel):
    """A forecast published every k steps ends k+h-1 short of the panel by
    construction. Reading only the fixed lag flagged four correct pipelines as
    dead feeds; reading only the cadence still flagged three."""
    ret, _, _, _ = panel
    strided = np.full_like(ret, np.nan)
    strided[::10] = ret[::10]
    assert F.m6_input_freshness({"ret": ret, "sig": strided}, horizon=5).outcome == "PASS"

    stale = np.abs(ret).copy()
    stale[-40:] = np.nan
    assert F.m6_input_freshness({"ret": ret, "vol": stale}, horizon=5).outcome == "FAIL"


def test_scaffold_lays_out_a_runnable_study(tmp_path):
    from falsifier import scaffold

    root = scaffold.create(str(tmp_path / "proj"), "demo")
    for rel in ("run_study.sh", "_pick_py.sh", "code/study.py", "code/build.py",
                "verify/health_check.py", "verify/reproduce.py", "config/prereg.json",
                "README.md", "docs/STATUS.md"):
        assert (root / rel).exists(), rel
    # The pre-registration gate is the point of the layout: no criterion, no run.
    study_src = (root / "code" / "study.py").read_text()
    assert "Write the pre-registration before the first result" in study_src
    assert "raise SystemExit" in study_src
    assert scaffold.PREREG_JSON["n_candidates_searched"] == 1
    with pytest.raises(FileExistsError):
        scaffold.create(str(root), "demo")


# --------------------------------------------------------------------------
# I1 / I2 -- incremental contribution
# --------------------------------------------------------------------------

def _inc_fixture(kind: str):
    g = np.random.default_rng(3)
    T_ = 252 * 8
    dates = np.array([int(f"{y}{1 + i // 21:02d}{1 + i % 21:02d}")
                      for y in range(2017, 2025) for i in range(252)][:T_])
    yr = np.array([int(str(d)[:4]) for d in dates])
    bench = g.standard_normal(T_) * 0.011
    base = bench + g.standard_normal(T_) * 0.006 + 0.0002
    if kind == "real":
        comb = base + g.standard_normal(T_) * 0.0016 + 0.0003
    elif kind == "one_year":
        comb = base + np.where(yr == 2020, 0.003, -0.00002)
    else:
        comb = base - 0.0002
    return F.Increment(dates=dates, baseline=base, combined=comb, benchmark=bench)


def test_i1_reads_the_year_by_year_delta():
    good = F.i1_incremental_contribution(_inc_fixture("real"))
    assert good.outcome == "PASS" and good.statistic > 0

    one = F.i1_incremental_contribution(_inc_fixture("one_year"))
    assert one.outcome == "FAIL" and "2020 alone" in one.detail

    down = F.i1_incremental_contribution(_inc_fixture("down"))
    assert down.outcome == "FAIL"


def test_increment_accounting_is_compounded():
    inc = _inc_fixture("real")
    rows = inc.yearly()
    c = inc.curves()
    # The curve is a chained product, so its final value is the compounded total
    # rather than a sum of yearly numbers.
    chained = 1.0
    for r in rows:
        chained *= (1.0 + r["combined"])
    assert abs(chained - c["nav_combined"][-1]) < 1e-6
    assert set(rows[0]) == {"year", "n", "baseline", "combined", "delta"}
    tm = inc.per_trade()
    assert all(k in next(iter(tm.values())) for k in
               ("n_trades", "win_rate", "avg_bp", "payoff_ratio", "profit_factor"))


def test_i2_rejects_an_increment_a_coin_would_match():
    """The case the yearly table cannot see."""
    g = np.random.default_rng(11)
    T_ = 252 * 8
    dates = np.array([int(f"{y}{1 + i // 21:02d}{1 + i % 21:02d}")
                      for y in range(2017, 2025) for i in range(252)][:T_])
    bench = g.standard_normal(T_) * 0.011
    base = bench + g.standard_normal(T_) * 0.006 + 0.0002

    def null_of(s):
        return base + np.random.default_rng(s).standard_normal(T_) * 0.004

    noise = next(c for c in (null_of(900 + k) for k in range(200))
                 if F.i1_incremental_contribution(
                     F.Increment(dates=dates, baseline=base, combined=c, benchmark=bench)
                 ).outcome == "PASS")
    inc = F.Increment(dates=dates, baseline=base, combined=noise, benchmark=bench)
    assert F.i1_incremental_contribution(inc).outcome == "PASS"
    assert F.i2_increment_null(inc, null_of, n_draws=80, seed=0).outcome == "FAIL"


def test_chart_writes_a_png_and_refuses_tofu(tmp_path):
    pytest.importorskip("matplotlib", reason="charts are an optional extra")
    inc = _inc_fixture("real")
    out = F.plot_increment(inc, str(tmp_path / "inc.png"), title="English title")
    assert Path(out).exists() and Path(out).stat().st_size > 10_000

    import falsifier.charts as ch
    real_finder = ch._find_cjk_fonts
    ch._find_cjk_fonts = lambda: []          # pretend nothing is installed
    try:
        with pytest.raises(RuntimeError, match="CJK"):
            F.plot_increment(inc, str(tmp_path / "cjk.png"), title="中文标题")
    finally:
        ch._find_cjk_fonts = real_finder
