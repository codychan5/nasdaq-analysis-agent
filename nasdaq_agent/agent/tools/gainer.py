from datetime import date
from typing import Literal

from langchain_core.tools import tool

from ...calendar import is_market_closed, previous_session_dates
from ...sources.errors import SourceError, SourceNoData
from ...universe import UniverseError
from ...validation import detect_split, reconcile_pct
from ..context import ExcludedCandidate, GainerInfo, ListingInfo, RunContext, SkippedCandidate, SourceAttempt
from .common import Deps, err, ok, log, precondition, record_call

CANDIDATES_TO_CHECK_PER_SOURCE = 5
SourceChoice = Literal["auto", "massive", "alphavantage", "yahoo", "nasdaqcom"]


def _remaining(ctx: RunContext, deps: Deps) -> list[str]:
    tried = {a.source for a in ctx.gainer_sources_tried}
    return [s.name for s in deps.gainer_sources if s.name not in tried]


class CandidateUncheckable(Exception):
    """No history source had a candidate's two closes, and at least one gave no answer: rate limited, down, over
    quota or broken. The candidate is unverified rather than missing, so the tool must not pass over it."""


def _two_closes(deps: Deps, symbol: str, prev: date, session: date):
    """The candidate's closes on the two sessions and its corporate actions, from the first history source that has
    both bars. None when every source answered without them, which is a real skip. CandidateUncheckable when any
    source gave no answer, because that source might have had them."""
    no_answer: list[str] = []
    for src in deps.history_sources:
        try:
            series = src.bars(symbol, prev, session)
            by_date = {b.date: b for b in series.bars}
            if prev in by_date and session in by_date:
                actions = src.corporate_actions(symbol, prev, session)
                return by_date[prev].close, by_date[session].close, actions
        except SourceNoData:
            continue
        except Exception as e:
            if not isinstance(e, SourceError):
                log.warning("_two_closes: %s raised %s", src.name, type(e).__name__)
            no_answer.append(f"{src.name}: {type(e).__name__}")
    if no_answer:
        raise CandidateUncheckable("; ".join(no_answer))
    return None


def make_find_top_gainer(ctx: RunContext, deps: Deps):
    @tool("find_top_gainer")
    def find_top_gainer(source: SourceChoice = "auto") -> str:
        """Find the NASDAQ common stock with the highest close-to-close percentage gain for the resolved session.
        Use source='auto' to walk the source chain, or name a source after a failure. Each source runs at most once."""
        blocked = precondition(ctx, "session_resolved", "find_top_gainer", "call resolve_session first")
        if blocked:
            record_call(ctx, deps, "find_top_gainer", {"source": source}, blocked)
            return blocked
        # A finished stage must close. Once a gainer has been chosen, re-running this tool could only pair the verified
        # five-day metrics with a different symbol than the one the model already committed to.
        if ctx.progress.gainer_chosen:
            outcome = err("precondition for find_top_gainer not met: a gainer has already been chosen")
            record_call(ctx, deps, "find_top_gainer", {"source": source}, outcome)
            return outcome
        session = date.fromisoformat(ctx.session.date)
        # Which stocks count is settled before any source is asked: NASDAQ's file for a current session, Massive's
        # records for an earlier one. Without that list no ranking can be trusted, so no source is tried.
        try:
            listing = deps.universe.for_session(session)
        except UniverseError as e:
            message = f"could not tell which stocks were listed on NASDAQ on {session}: {e}"
            if message not in ctx.errors:
                ctx.errors.append(message)
            outcome = err(message)
            record_call(ctx, deps, "find_top_gainer", {"source": source}, outcome)
            return outcome
        ctx.listing = ListingInfo(source=listing.source,
                                  listed_on=listing.listed_on.isoformat() if listing.listed_on else None,
                                  common_stocks=listing.common_stock_count())
        prev = previous_session_dates(session, 2)[0]
        market_closed = is_market_closed(deps.clock())
        remaining = _remaining(ctx, deps)
        if source != "auto" and source not in remaining:
            outcome = err(f"{source} was already tried or is not configured; remaining sources: {', '.join(remaining) or 'none'}")
            record_call(ctx, deps, "find_top_gainer", {"source": source}, outcome)
            return outcome
        chain = [s for s in deps.gainer_sources if s.name in remaining and (source == "auto" or s.name == source)]
        for src in chain:
            if src.requires_market_closed and not market_closed:
                ctx.gainer_sources_tried.append(SourceAttempt(source=src.name, ok=False, detail="not allowed while the market is open"))
                if source != "auto":
                    break
                continue
            try:
                candidates = src.top_candidates(session, prev, limit=CANDIDATES_TO_CHECK_PER_SOURCE)
            except SourceError as e:
                ctx.gainer_sources_tried.append(SourceAttempt(source=src.name, ok=False, detail=str(e)[:200]))
                if source != "auto":
                    break
                continue
            except Exception as e:
                ctx.gainer_sources_tried.append(SourceAttempt(source=src.name, ok=False, detail=f"{type(e).__name__}: {e}"[:200]))
                log.warning("find_top_gainer: %s raised %s", src.name, type(e).__name__)
                if source != "auto":
                    break
                continue
            unchecked: str | None = None
            for cand in candidates:
                if cand.excluded is not None:
                    # Ranked above the stocks being checked, but it does not count that day: recorded, never checked.
                    ctx.excluded_candidates.append(ExcludedCandidate(symbol=cand.symbol, source=src.name,
                                                                     pct_change=cand.pct_change, reason=cand.excluded))
                    continue
                try:
                    closes = _two_closes(deps, cand.symbol, prev, session)
                except CandidateUncheckable as e:
                    # Passing over it would name a lower-ranked stock because a source was busy, not because of the
                    # data, so this source's ranking ends here and the chain moves on.
                    unchecked = f"could not check {cand.symbol} ({e}), so no lower-ranked stock was chosen"
                    break
                if closes is None:
                    ctx.skipped_candidates.append(SkippedCandidate(symbol=cand.symbol, source=src.name, reason="no bars to reconcile against"))
                    continue
                prev_close, close, actions = closes
                rec = reconcile_pct(cand.pct_change, prev_close, close)
                if not rec.ok:
                    ctx.skipped_candidates.append(SkippedCandidate(symbol=cand.symbol, source=src.name,
                        reason=f"failed to reconcile: source {cand.pct_change:.2f}% vs own bars {rec.computed_pct:.2f}%"))
                    continue
                if detect_split(actions, session):
                    ctx.skipped_candidates.append(SkippedCandidate(symbol=cand.symbol, source=src.name, reason="split effective in session"))
                    continue
                ctx.gainer = GainerInfo(symbol=cand.symbol, company=cand.name, prev_close=prev_close, close=close,
                                        pct_change=rec.computed_pct, source=src.name, reconciled_pct=rec.computed_pct)
                ctx.gainer_sources_tried.append(SourceAttempt(source=src.name, ok=True, detail=f"chose {cand.symbol}"))
                ctx.progress.gainer_chosen = True
                outcome = ok({"symbol": cand.symbol, "company": cand.name, "pct_change": round(rec.computed_pct, 3),
                              "prev_close": prev_close, "close": close, "source": src.name,
                              "skipped": [s.model_dump() for s in ctx.skipped_candidates]})
                record_call(ctx, deps, "find_top_gainer", {"source": source}, outcome)
                return outcome
            detail = unchecked or "no candidate passed integrity checks"
            ctx.gainer_sources_tried.append(SourceAttempt(source=src.name, ok=False, detail=detail[:200]))
            if source != "auto":
                break
        last = ctx.gainer_sources_tried[-1] if ctx.gainer_sources_tried else None
        detail = f"{last.source} failed: {last.detail}" if last else "no sources configured"
        if last and "market is open" in last.detail:
            detail = f"{last.source} is not allowed because the market is open"
        outcome = err(f"{detail}. remaining sources: {', '.join(_remaining(ctx, deps)) or 'none'}")
        record_call(ctx, deps, "find_top_gainer", {"source": source}, outcome)
        return outcome
    return find_top_gainer
