from dataclasses import dataclass
from datetime import date

from .metrics import gainer_pct
from .sources.models import BarSeries, CorporateAction

RECONCILE_TOLERANCE_PCT_POINTS = 1.0


@dataclass(frozen=True)
class ReconcileResult:
    ok: bool
    reported_pct: float
    computed_pct: float
    diff: float


def reconcile_pct(reported_pct: float, prev_close: float, close: float,
                  tolerance: float = RECONCILE_TOLERANCE_PCT_POINTS) -> ReconcileResult:
    """Compare a source's percentage with our own raw close-to-close computation."""
    try:
        computed = gainer_pct(prev_close, close)
    except ValueError:
        return ReconcileResult(False, reported_pct, float("nan"), float("nan"))
    diff = abs(reported_pct - computed)
    return ReconcileResult(diff <= tolerance, reported_pct, computed, diff)


def detect_split(actions: list[CorporateAction], session: date) -> bool:
    """True when a split or reverse split is effective on the session date."""
    return any(a.kind == "split" and a.date == session for a in actions)


@dataclass(frozen=True)
class CompletenessResult:
    ok: bool
    missing: list[date]
    extra: list[date]


def check_completeness(series: BarSeries, expected_dates: list[date]) -> CompletenessResult:
    have = set(series.dates())
    want = set(expected_dates)
    missing = sorted(want - have)
    extra = sorted(have - want)
    return CompletenessResult(ok=not missing and not extra, missing=missing, extra=extra)
