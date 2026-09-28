import math
import pytest

TOL = 0.01

def test_canonical_fixture(canonical_closes, canonical_bench_closes):
    from nasdaq_agent import metrics as m
    r = m.compute_all(canonical_closes, canonical_bench_closes)
    assert [round(x, 3) for x in r.daily_changes_pct] == [2.0, -2.451, 6.533, 0.943, 3.738]
    assert abs(r.avg_daily_change_pct - 2.153) < TOL
    assert abs(r.cumulative_return_pct - 11.0) < TOL
    assert abs(r.volatility_annualized_pct - 52.87) < TOL
    assert abs(r.max_drawdown_pct - (-2.451)) < TOL
    assert abs(r.relative_vs_spy_pct - 10.2) < TOL
    assert r.trend == m.Trend.uptrend

@pytest.mark.parametrize("closes, expected", [
    ([10, 9.8, 9.6, 9.5, 9.7, 9.3], "downtrend"),
    ([10, 10.1, 9.9, 10.05, 9.95, 10.1], "sideways"),
    ([10, 10.5, 9.8, 10.6, 10.0, 10.9], "mixed"),
])
def test_trend_rule(closes, expected):
    from nasdaq_agent import metrics as m
    assert m.trend_label(closes) == m.Trend(expected)

def test_needs_exactly_six_closes():
    from nasdaq_agent import metrics as m
    with pytest.raises(ValueError):
        m.daily_changes_pct([10, 10.2, 9.95, 10.6, 10.7])

def test_rejects_non_finite_and_non_positive():
    from nasdaq_agent import metrics as m
    with pytest.raises(ValueError):
        m.cumulative_return_pct([10, 10.2, float("nan"), 10.6, 10.7, 11.1])
    with pytest.raises(ValueError):
        m.cumulative_return_pct([10, 10.2, 0.0, 10.6, 10.7, 11.1])

def test_result_model_rejects_extra_and_short_lists():
    from nasdaq_agent.metrics import AnalysisResult
    base = dict(daily_changes_pct=[1, 1, 1, 1, 1], avg_daily_change_pct=1, cumulative_return_pct=5,
                volatility_annualized_pct=1, max_drawdown_pct=0, relative_vs_spy_pct=1, trend="uptrend")
    AnalysisResult(**base)
    with pytest.raises(Exception):
        AnalysisResult(**base, extra_field=1)
    with pytest.raises(Exception):
        AnalysisResult(**{**base, "daily_changes_pct": [1, 1, 1, 1]})
    with pytest.raises(Exception):
        AnalysisResult(**{**base, "avg_daily_change_pct": float("inf")})

def test_gainer_pct():
    from nasdaq_agent.metrics import gainer_pct
    assert abs(gainer_pct(10.0, 11.1) - 11.0) < TOL

def test_gainer_pct_rejects_non_positive_and_non_finite():
    from nasdaq_agent.metrics import gainer_pct
    for prev, close in [(10.0, 0.0), (10.0, -5.0), (0.0, 10.0), (10.0, float("nan")), (float("inf"), 10.0)]:
        with pytest.raises(ValueError):
            gainer_pct(prev, close)

def test_definitions_text_mentions_every_metric():
    from nasdaq_agent.metrics import DEFINITIONS_TEXT, METRIC_NAMES
    for name in METRIC_NAMES:
        assert name in DEFINITIONS_TEXT
