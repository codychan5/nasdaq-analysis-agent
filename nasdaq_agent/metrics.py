"""Reviewed reference implementations. The prompt and the verifier both use these definitions."""
import math
from enum import StrEnum
from statistics import mean, stdev
from typing import Sequence

from pydantic import BaseModel, ConfigDict, field_validator

REQUIRED_CLOSES = 6
RETURNS_PER_WINDOW = REQUIRED_CLOSES - 1
TRADING_DAYS_PER_YEAR = 252
SIDEWAYS_THRESHOLD_PCT = 2.0
UPTREND_MIN_UP_DAYS = 4
DOWNTREND_MAX_UP_DAYS = 1
PCT_TOLERANCE = 0.01
METRIC_NAMES = (
    "daily_changes_pct", "avg_daily_change_pct", "cumulative_return_pct",
    "volatility_annualized_pct", "max_drawdown_pct", "relative_vs_spy_pct", "trend",
)


class Trend(StrEnum):
    uptrend = "uptrend"
    downtrend = "downtrend"
    sideways = "sideways"
    mixed = "mixed"


def _finite(value: float, name: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    return float(value)


class AnalysisResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    daily_changes_pct: list[float]
    avg_daily_change_pct: float
    cumulative_return_pct: float
    volatility_annualized_pct: float
    max_drawdown_pct: float
    relative_vs_spy_pct: float
    trend: Trend
    notes: str | None = None

    @field_validator("daily_changes_pct")
    @classmethod
    def _five_finite(cls, v: list[float]) -> list[float]:
        if len(v) != RETURNS_PER_WINDOW:
            raise ValueError(f"daily_changes_pct must have {RETURNS_PER_WINDOW} entries, got {len(v)}")
        return [_finite(x, "daily_changes_pct") for x in v]

    @field_validator("avg_daily_change_pct", "cumulative_return_pct", "volatility_annualized_pct",
                     "max_drawdown_pct", "relative_vs_spy_pct")
    @classmethod
    def _finite_scalar(cls, v: float, info) -> float:
        return _finite(v, info.field_name)


def _validate_closes(closes: Sequence[float]) -> list[float]:
    if len(closes) != REQUIRED_CLOSES:
        raise ValueError(f"need exactly {REQUIRED_CLOSES} closes, got {len(closes)}")
    out = []
    for c in closes:
        c = float(c)
        if not math.isfinite(c) or c <= 0:
            raise ValueError(f"closes must be finite and positive, got {c}")
        out.append(c)
    return out


def daily_changes_pct(closes: Sequence[float]) -> list[float]:
    c = _validate_closes(closes)
    return [(c[i] / c[i - 1] - 1.0) * 100.0 for i in range(1, len(c))]


def avg_daily_change_pct(closes: Sequence[float]) -> float:
    return mean(daily_changes_pct(closes))


def cumulative_return_pct(closes: Sequence[float]) -> float:
    c = _validate_closes(closes)
    return (c[-1] / c[0] - 1.0) * 100.0


def volatility_annualized_pct(closes: Sequence[float]) -> float:
    return stdev(daily_changes_pct(closes)) * math.sqrt(TRADING_DAYS_PER_YEAR)


def max_drawdown_pct(closes: Sequence[float]) -> float:
    c = _validate_closes(closes)
    peak, worst = c[0], 0.0
    for price in c:
        peak = max(peak, price)
        worst = min(worst, (price / peak - 1.0) * 100.0)
    return worst


def relative_vs_benchmark_pct(closes: Sequence[float], bench_closes: Sequence[float]) -> float:
    return cumulative_return_pct(closes) - cumulative_return_pct(bench_closes)


def trend_label(closes: Sequence[float]) -> Trend:
    c = _validate_closes(closes)
    up_days = sum(1 for r in daily_changes_pct(c) if r > 0)
    window_mean = mean(c)
    if up_days >= UPTREND_MIN_UP_DAYS and c[-1] > window_mean:
        return Trend.uptrend
    if up_days <= DOWNTREND_MAX_UP_DAYS and c[-1] < window_mean:
        return Trend.downtrend
    if abs(cumulative_return_pct(c)) < SIDEWAYS_THRESHOLD_PCT:
        return Trend.sideways
    return Trend.mixed


def compute_all(closes: Sequence[float], bench_closes: Sequence[float]) -> AnalysisResult:
    return AnalysisResult(
        daily_changes_pct=daily_changes_pct(closes),
        avg_daily_change_pct=avg_daily_change_pct(closes),
        cumulative_return_pct=cumulative_return_pct(closes),
        volatility_annualized_pct=volatility_annualized_pct(closes),
        max_drawdown_pct=max_drawdown_pct(closes),
        relative_vs_spy_pct=relative_vs_benchmark_pct(closes, bench_closes),
        trend=trend_label(closes),
    )


def gainer_pct(prev_close: float, close: float) -> float:
    """Close-to-close percentage change on raw closes."""
    if not math.isfinite(prev_close) or prev_close <= 0:
        raise ValueError(f"prev_close must be finite and positive, got {prev_close}")
    if not math.isfinite(close) or close <= 0:
        raise ValueError(f"close must be finite and positive, got {close}")
    return (close / prev_close - 1.0) * 100.0


DEFINITIONS_TEXT = f"""Given adjusted closes c0..c5 for six consecutive completed sessions, and daily change
r_i = (c_i / c_(i-1) - 1) * 100 for i = 1..5:
- daily_changes_pct: the list [r_1, ..., r_5].
- avg_daily_change_pct: the arithmetic mean of r.
- cumulative_return_pct: (c5 / c0 - 1) * 100.
- volatility_annualized_pct: the SAMPLE standard deviation of r (ddof=1) times sqrt({TRADING_DAYS_PER_YEAR}).
- max_drawdown_pct: the minimum over i of (c_i / max(c_0..c_i) - 1) * 100.
- relative_vs_spy_pct: the ticker's cumulative_return_pct minus SPY's cumulative_return_pct over the same dates.
- trend: let up_days = count of r_i > 0 and m = mean(c0..c5). 'uptrend' if up_days >= {UPTREND_MIN_UP_DAYS} and c5 > m;
  'downtrend' if up_days <= {DOWNTREND_MAX_UP_DAYS} and c5 < m; otherwise 'sideways' if |cumulative_return_pct| < {SIDEWAYS_THRESHOLD_PCT};
  otherwise 'mixed'.
"""
