from datetime import date

def test_reconcile_within_tolerance():
    from nasdaq_agent.validation import reconcile_pct
    r = reconcile_pct(reported_pct=11.3, prev_close=10.0, close=11.1)
    assert r.ok and abs(r.computed_pct - 11.0) < 0.01 and abs(r.diff - 0.3) < 0.01

def test_reconcile_tolerates_a_cent_of_rounding_on_a_large_move():
    """CAST on 2026-06-12: Massive's raw closes 0.644 -> 1.55 (+140.68%), Yahoo's 0.64 -> 1.55 (+142.19%). Half a cent
    on a sub-dollar price is a 1.5-point gap on a 141% move, yet the two agree within 1% on the price ratio, so the
    stock is real and must not be skipped (run 20260929T143527Z-20d731 named the wrong gainer over this)."""
    from nasdaq_agent.metrics import gainer_pct
    from nasdaq_agent.validation import reconcile_pct
    r = reconcile_pct(reported_pct=gainer_pct(0.644, 1.55), prev_close=0.64, close=1.55)
    assert r.ok is True
    assert abs(r.diff - 1.504) < 0.01  # the gap in points is still reported for the skip message


def test_reconcile_rejects_reverse_split_artifact():
    from nasdaq_agent.validation import reconcile_pct
    r = reconcile_pct(reported_pct=1190.0, prev_close=0.10, close=1.20)
    assert r.ok is False

def test_reconcile_rejects_a_two_for_one_artifact_on_a_small_move():
    """The smallest split artefact: the source's raw closes straddle a 2-for-1 reverse split (+120%) while our bars
    show the real +10%. In points that gap shrank to nothing under the old rule only for tiny moves; on the price
    ratio it is always 100%, far past the tolerance."""
    from nasdaq_agent.validation import RECONCILE_TOLERANCE_RATIO, reconcile_pct
    r = reconcile_pct(reported_pct=120.0, prev_close=1.0, close=1.10)
    assert r.ok is False
    assert abs(r.ratio_gap - 1.0) < 1e-9 and r.ratio_gap > RECONCILE_TOLERANCE_RATIO


def test_reconcile_not_ok_when_gainer_pct_raises():
    import math
    from nasdaq_agent.validation import reconcile_pct
    r = reconcile_pct(reported_pct=5.0, prev_close=0.0, close=5.0)
    assert r.ok is False and math.isnan(r.computed_pct)

def test_detect_split_on_session():
    from nasdaq_agent.validation import detect_split
    from nasdaq_agent.sources.models import CorporateAction
    acts = [CorporateAction(date=date(2026, 9, 24), kind="split", ratio=0.1)]
    assert detect_split(acts, date(2026, 9, 24)) is True
    assert detect_split(acts, date(2026, 9, 23)) is False
    assert detect_split([CorporateAction(date=date(2026, 9, 24), kind="dividend", ratio=0.5)], date(2026, 9, 24)) is False

def test_completeness(canonical_rows):
    from nasdaq_agent.validation import check_completeness
    from nasdaq_agent.sources.models import Bar, BarSeries
    bars = [Bar(date=date.fromisoformat(r["date"]), open=r["open"], high=r["high"], low=r["low"],
                close=r["close"], adj_close=r["adj_close"], volume=r["volume"]) for r in canonical_rows]
    series = BarSeries(symbol="ACME", bars=bars, source="test")
    expected = [b.date for b in bars]
    assert check_completeness(series, expected).ok
    short = BarSeries(symbol="ACME", bars=bars[:-1], source="test")
    res = check_completeness(short, expected)
    assert res.ok is False and res.missing == [date(2026, 9, 24)]

def test_completeness_rejects_a_repeated_session(canonical_rows):
    """A provider that repeats a day passes a set comparison, but the metrics need exactly one bar per session."""
    from nasdaq_agent.validation import check_completeness
    from nasdaq_agent.sources.models import Bar, BarSeries
    bars = [Bar(date=date.fromisoformat(r["date"]), open=r["open"], high=r["high"], low=r["low"],
                close=r["close"], adj_close=r["adj_close"], volume=r["volume"]) for r in canonical_rows]
    repeated = BarSeries(symbol="ACME", bars=bars + bars[-1:], source="test")
    res = check_completeness(repeated, [b.date for b in bars])
    assert res.ok is False and res.duplicates == [date(2026, 9, 24)] and res.missing == []
