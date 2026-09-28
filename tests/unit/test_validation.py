from datetime import date

def test_reconcile_within_tolerance():
    from nasdaq_agent.validation import reconcile_pct
    r = reconcile_pct(reported_pct=11.3, prev_close=10.0, close=11.1)
    assert r.ok and abs(r.computed_pct - 11.0) < 0.01 and abs(r.diff - 0.3) < 0.01

def test_reconcile_rejects_reverse_split_artifact():
    from nasdaq_agent.validation import reconcile_pct
    r = reconcile_pct(reported_pct=1190.0, prev_close=0.10, close=1.20)
    assert r.ok is False

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
