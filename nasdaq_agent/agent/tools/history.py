from datetime import date, timedelta
from typing import Literal

from langchain_core.tools import tool

from ...calendar import previous_session_dates
from ...metrics import DEFINITIONS_TEXT, METRIC_NAMES, REQUIRED_CLOSES
from ...report.render import source_label
from ...sandbox.runner import prepare_sandbox_dir
from ...sources.errors import SourceError, SourceNoData
from ...sources.models import BarSeries
from ...validation import check_completeness
from ..context import HistoryInfo, PriceCheck, RunContext, SourceAttempt
from .common import Deps, err, ok, log, precondition, record_call

PRICE_CHECK_TOLERANCE = 0.005  # a close disagrees when the sources differ by more than 0.5% of the larger one
QUOTED_TO_THE_CENT_FROM = 1.0  # a close under a dollar is quoted to four decimals, as markets quote it
WIDEN_WINDOW_DAYS = 7
HEAD_ROWS = 3
COLUMNS = ["date", "open", "high", "low", "close", "adj_close", "volume"]
SourceChoice = Literal["auto", "yfinance", "massive"]


def _to_csv(series: BarSeries) -> str:
    rows = [",".join(COLUMNS)]
    for b in series.bars:
        rows.append(f"{b.date.isoformat()},{b.open},{b.high},{b.low},{b.close},{b.adj_close},{b.volume}")
    return "\n".join(rows) + "\n"


def _fetch_complete(src, symbol: str, expected: list[date], notes: list[str]) -> BarSeries | None:
    start, end = expected[0], expected[-1]
    for attempt, s in enumerate((start, start - timedelta(days=WIDEN_WINDOW_DAYS))):
        series = src.bars(symbol, s, end)
        wanted = BarSeries(symbol=symbol, source=series.source, bars=[b for b in series.bars if b.date in set(expected)])
        res = check_completeness(wanted, expected)
        if res.ok:
            return wanted
        problems = []
        if res.missing:
            problems.append(f"missing {', '.join(d.isoformat() for d in res.missing)}")
        if res.duplicates:
            problems.append(f"repeats {', '.join(d.isoformat() for d in res.duplicates)}")
        notes.append(f"{src.name}: {symbol} {'; '.join(problems)} (attempt {attempt + 1})")
    return None


def _price_text(price: float) -> str:
    return f"{price:.2f}" if price >= QUOTED_TO_THE_CENT_FROM else f"{price:.4f}"


def _closes_differ(ours: float, theirs: float) -> bool:
    return abs(ours - theirs) > PRICE_CHECK_TOLERANCE * max(abs(ours), abs(theirs))


# How far each day's factor may sit from the middle one for the six to count as one factor. The restated closes are
# rounded to the cent, so a factor carries up to half a cent of noise relative to the restated close: 0.5% at a dollar,
# less above it (AIXI's real factors ran 19.93 to 20.05, a 0.6% spread). 2% leaves room and is still far from anything
# a one-day data error produces.
RESTATEMENT_FACTOR_SPREAD = 0.02
# The smallest factor that reads as a restatement: a 10% stock dividend is the smallest common one. A source that is
# off by the same 1% every day is not reporting a split.
MIN_RESTATEMENT_FACTOR = 1.1


def _constant_factor(pairs: list[tuple[float, float]]) -> float | None:
    """The one factor by which every `ours` close exceeds its `theirs` close: the middle factor when every day's
    factor sits within RESTATEMENT_FACTOR_SPREAD of it and it is at least MIN_RESTATEMENT_FACTOR from 1, either way.
    None otherwise. A split after the sessions makes a provider restate every earlier close by the split ratio, so
    the factor is the same on every day; a data error never is."""
    factors = sorted(ours / theirs for ours, theirs in pairs if theirs > 0)
    if len(factors) != len(pairs) or not factors:
        return None
    middle = factors[len(factors) // 2]
    if any(abs(f - middle) > RESTATEMENT_FACTOR_SPREAD * middle for f in factors):
        return None
    if 1 / MIN_RESTATEMENT_FACTOR < middle < MIN_RESTATEMENT_FACTOR:
        return None
    return middle


def _factor_text(factor: float) -> str:
    return f"{factor:.3g} times" if factor >= 1 else f"1/{1 / factor:.3g} of"


def _restatement(primary, second, ticker: BarSeries, theirs: dict[date, float],
                 disagreements: list) -> PriceCheck | None:
    """When every close disagrees by one factor and one source's closes are the traded prices, the other has restated
    its history for a later split: report that, with the traded closes for the headline. None otherwise."""
    if len(disagreements) != len(ticker.bars):
        return None
    ours_raw, theirs_raw = getattr(primary, "closes_are_raw", False), getattr(second, "closes_are_raw", False)
    if ours_raw == theirs_raw:
        return None
    factor = _constant_factor([(b.close, theirs[b.date]) for b in ticker.bars])
    if factor is None:
        return None
    raw, restated = (primary, second) if ours_raw else (second, primary)
    factor = 1 / factor if ours_raw else factor  # the restated source's closes over the raw source's
    traded = [theirs[b.date] for b in ticker.bars] if theirs_raw else [b.close for b in ticker.bars]
    return PriceCheck(status="restated", source=second.name,
                      detail=f"{source_label(restated.name)}'s closes are {_factor_text(factor)} "
                             f"{source_label(raw.name)}'s on all {len(ticker.bars)} sessions: a split after these "
                             f"sessions restated them, so the day's prices shown are as traded, from "
                             f"{source_label(raw.name)}.",
                      traded_prev_close=traded[-2], traded_close=traded[-1])


def _price_check(deps: Deps, primary, symbol: str, expected: list[date], ticker: BarSeries,
                 notes: list[str]) -> PriceCheck:
    """Compare the stock's closes with the next history source; the benchmark is not checked. The check never
    blocks the run: when it cannot be made, the report says so and why. The report shows detail as it is, so it names
    the sources as the email does ("Massive"); source keeps the internal name."""
    second = next((s for s in deps.history_sources if s.name != primary.name), None)
    if second is None:
        return PriceCheck(status="not_checked", detail="Prices were not cross-checked: no second price source is configured.")
    sessions = len(expected)
    second_label, primary_label = source_label(second.name), source_label(primary.name)
    lacks_sessions = PriceCheck(status="not_checked", source=second.name,
                                detail=f"Prices were not cross-checked: {second_label} does not have all {sessions} sessions.")
    try:
        other = _fetch_complete(second, symbol, expected, notes)
        if other is None:
            return lacks_sessions
        # yfinance's close is split-adjusted and Massive's is raw, so across a split they differ by the split ratio.
        splits = [a for src in (primary, second) for a in src.corporate_actions(symbol, expected[0], expected[-1])
                  if a.kind == "split"]
    except SourceNoData:
        return lacks_sessions
    except Exception as e:  # rate limited, down, over quota or broken
        if not isinstance(e, SourceError):
            log.warning("price check: %s raised %s", second.name, type(e).__name__)
        return PriceCheck(status="not_checked", source=second.name,
                          detail=f"Prices were not cross-checked: {second_label} did not answer ({type(e).__name__}).")
    if splits:
        return PriceCheck(status="not_checked", source=second.name,
                          detail=f"Prices were not cross-checked: a split falls inside the {sessions} sessions, and the "
                                 "sources adjust for it differently.")
    theirs = {b.date: b.close for b in other.bars}
    tolerance = f"{PRICE_CHECK_TOLERANCE:.1%}"
    disagreements = [(b.date, b.close, theirs[b.date]) for b in ticker.bars if _closes_differ(b.close, theirs[b.date])]
    if not disagreements:
        return PriceCheck(status="agree", source=second.name,
                          detail=f"Closes cross-checked against {second_label}: all {len(ticker.bars)} agree within "
                                 f"{tolerance}.")
    restated = _restatement(primary, second, ticker, theirs, disagreements)
    if restated is not None:
        return restated
    listed = "; ".join(f"{day.isoformat()}: {_price_text(ours)} vs {_price_text(other_close)}"
                       for day, ours, other_close in disagreements)
    return PriceCheck(status="disagree", source=second.name,
                      detail=f"Price check: {second_label} disagrees with {primary_label} on {len(disagreements)} of "
                             f"{len(ticker.bars)} closes by more than {tolerance} ({listed}).")


def make_get_price_history(ctx: RunContext, deps: Deps):
    @tool("get_price_history")
    def get_price_history(source: SourceChoice = "auto") -> str:
        """Fetch six completed sessions of daily bars for the chosen gainer and for the benchmark, validate them
        against the calendar, and write them into the sandbox as ticker.csv, benchmark.csv and meta.json."""
        blocked = precondition(ctx, "gainer_chosen", "get_price_history", "call find_top_gainer first")
        if blocked:
            record_call(ctx, deps, "get_price_history", {"source": source}, blocked)
            return blocked
        # A finished stage must close. Once history is ready, re-fetching it could only pair the verified five-day
        # metrics with different bars than the ones run_python and verify_analysis actually used.
        if ctx.progress.history_ready:
            outcome = err("precondition for get_price_history not met: history has already been fetched")
            record_call(ctx, deps, "get_price_history", {"source": source}, outcome)
            return outcome
        session = date.fromisoformat(ctx.session.date)
        expected = previous_session_dates(session, deps.settings.lookback_sessions + 1)
        tried = {a.source for a in ctx.history_sources_tried}
        chain = [s for s in deps.history_sources if s.name not in tried and (source == "auto" or s.name == source)]
        for src in chain:
            try:
                ticker = _fetch_complete(src, ctx.gainer.symbol, expected, ctx.notes)
                bench = _fetch_complete(src, deps.settings.benchmark_symbol, expected, ctx.notes) if ticker else None
            except SourceError as e:
                ctx.history_sources_tried.append(SourceAttempt(source=src.name, ok=False, detail=str(e)[:200]))
                continue
            except Exception as e:
                ctx.history_sources_tried.append(SourceAttempt(source=src.name, ok=False, detail=f"{type(e).__name__}: {e}"[:200]))
                log.warning("get_price_history: %s raised %s", src.name, type(e).__name__)
                continue
            if ticker is None or bench is None:
                ctx.history_sources_tried.append(SourceAttempt(source=src.name, ok=False, detail="incomplete after widening"))
                continue
            meta = {"symbol": ctx.gainer.symbol, "session_date": ctx.session.date, "benchmark": deps.settings.benchmark_symbol,
                    "required_keys": list(METRIC_NAMES), "definitions": DEFINITIONS_TEXT, "close_column": "adj_close",
                    "rows": len(ticker.bars)}
            prepare_sandbox_dir(deps.run_dir.path, _to_csv(ticker), _to_csv(bench), meta)
            price_check = _price_check(deps, src, ctx.gainer.symbol, expected, ticker, ctx.notes)
            ctx.history = HistoryInfo(ticker=ticker, benchmark=bench, source=src.name, expected_dates=[d.isoformat() for d in expected],
                                      price_check=price_check)
            as_traded = None
            if price_check.status == "restated":
                # The headline and the cheap-stock warning show the prices that traded, not a later restatement; the
                # move is the same on either basis, so pct_change stands.
                ctx.gainer.prev_close, ctx.gainer.close = price_check.traded_prev_close, price_check.traded_close
                as_traded = {"prev_close": price_check.traded_prev_close, "close": price_check.traded_close}
            ctx.history_sources_tried.append(SourceAttempt(source=src.name, ok=True, detail=f"{len(ticker.bars)} bars"))
            ctx.progress.history_ready = True
            head = [dict(zip(COLUMNS, [b.date.isoformat(), b.open, b.high, b.low, b.close, b.adj_close, b.volume])) for b in ticker.bars[:HEAD_ROWS]]
            outcome = ok({"source": src.name, "rows": len(ticker.bars), "columns": COLUMNS, "date_range": [expected[0].isoformat(), expected[-1].isoformat()],
                          "head": head, "variables_in_sandbox": ["df (ticker bars)", "bench (benchmark bars)", "meta"],
                          "close_column": "adj_close", "required_result_keys": list(METRIC_NAMES),
                          **({"prices_as_traded": as_traded,
                              "note": f"{price_check.detail} Quote these as the session's prices; prev_close and close "
                                      "now hold them, and the bars' last two closes are declarable as "
                                      "prev_close_restated and close_restated."} if as_traded else {})})
            record_call(ctx, deps, "get_price_history", {"source": source}, outcome)
            return outcome
        remaining = [s.name for s in deps.history_sources if s.name not in {a.source for a in ctx.history_sources_tried}]
        outcome = err(f"no complete {REQUIRED_CLOSES}-session history; remaining sources: {', '.join(remaining) or 'none'}")
        record_call(ctx, deps, "get_price_history", {"source": source}, outcome)
        return outcome
    return get_price_history
