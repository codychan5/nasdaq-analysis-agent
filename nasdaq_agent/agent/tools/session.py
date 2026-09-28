from langchain_core.tools import tool

from ...calendar import CalendarError, is_market_closed, resolve_last_completed_session
from ..context import RunContext, SessionInfo
from .common import Deps, err, ok, record_call


def make_resolve_session(ctx: RunContext, deps: Deps):
    @tool("resolve_session")
    def resolve_session() -> str:
        """Resolve the last completed NASDAQ trading session from the trading calendar. Call this first."""
        try:
            now = deps.clock()
            session = resolve_last_completed_session(now)
            ctx.session = SessionInfo(date=session.date.isoformat(), label=session.label,
                                      early_close=session.early_close, market_closed_at_resolution=is_market_closed(now))
            ctx.progress.session_resolved = True
            outcome = ok({"session_date": ctx.session.date, "label": ctx.session.label,
                          "early_close": ctx.session.early_close, "market_closed_now": ctx.session.market_closed_at_resolution})
        except CalendarError as e:
            ctx.errors.append(f"calendar: {e}")
            outcome = err(f"calendar could not resolve a session: {e}")
        record_call(ctx, deps, "resolve_session", {}, outcome)
        return outcome
    return resolve_session
