from collections import Counter
from dataclasses import dataclass, field
from datetime import date

from .metrics import gainer_pct
from .sources.models import BarSeries, CorporateAction

# How far the source's price ratio (close over previous close) may sit from the one our own bars give, as a fraction
# of it. The two things this check must tell apart sit at very different scales: a split artefact (the source's raw
# closes straddle a split, our bars do not) is a ratio gap of 100% or more, while providers quoting a sub-dollar close
# to a different number of decimals (Yahoo 0.64 against Massive 0.644 for CAST on 2026-06-12) is a gap under 1% of the
# ratio. A fixed gap in percentage points cannot separate them, because rounding grows with the move: that half cent
# was 1.5 points on a 141% day, and the real top gainer was skipped.
RECONCILE_TOLERANCE_RATIO = 0.02


@dataclass(frozen=True)
class ReconcileResult:
    ok: bool
    reported_pct: float
    computed_pct: float
    diff: float  # the gap in percentage points, for the skip message
    ratio_gap: float  # the gap between the two price ratios as a fraction of ours, what ok is decided on


def reconcile_pct(reported_pct: float, prev_close: float, close: float,
                  tolerance: float = RECONCILE_TOLERANCE_RATIO) -> ReconcileResult:
    """Compare a source's percentage with our own raw close-to-close computation: ok when the two price ratios agree
    within `tolerance`, a fraction. A prev_close or close that cannot give a percentage is not ok."""
    try:
        computed = gainer_pct(prev_close, close)
    except ValueError:
        return ReconcileResult(False, reported_pct, float("nan"), float("nan"), float("nan"))
    diff = abs(reported_pct - computed)
    ratio_gap = abs((1.0 + reported_pct / 100.0) / (1.0 + computed / 100.0) - 1.0)
    return ReconcileResult(ratio_gap <= tolerance, reported_pct, computed, diff, ratio_gap)


def detect_split(actions: list[CorporateAction], session: date) -> bool:
    """True when a split or reverse split is effective on the session date."""
    return any(a.kind == "split" and a.date == session for a in actions)


@dataclass(frozen=True)
class CompletenessResult:
    ok: bool
    missing: list[date]
    extra: list[date]
    duplicates: list[date] = field(default_factory=list)  # sessions the series holds more than once


def check_completeness(series: BarSeries, expected_dates: list[date]) -> CompletenessResult:
    dates = series.dates()
    have = set(dates)
    want = set(expected_dates)
    missing = sorted(want - have)
    extra = sorted(have - want)
    # A set comparison alone passes a repeated day, but the metrics need exactly one bar per session.
    duplicates = sorted(day for day, count in Counter(dates).items() if count > 1)
    return CompletenessResult(ok=not missing and not extra and not duplicates, missing=missing, extra=extra,
                              duplicates=duplicates)
