import logging
import threading
import time
from collections.abc import Callable

from langchain.agents.middleware import (AgentMiddleware, ModelCallLimitMiddleware, ToolCallLimitMiddleware,
                                         hook_config)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ..config import Settings
from ..sources.errors import CassetteMiss
from .context import Progress, RunContext

log = logging.getLogger("nasdaq_agent.middleware")
RUN_DEADLINE_REACHED = "run deadline of {seconds} s reached"


def allowed_tool_names(progress: Progress, has_headlines: bool = True) -> set[str]:
    """Which tools the model may see right now. Tools re-check these flags themselves. has_headlines is False when
    get_news returned no headlines: record_sentiment then has nothing to score and is not offered."""
    if progress.sent:
        return set()
    allowed = {"give_up"}
    if progress.gave_up:
        return allowed
    if not progress.session_resolved:
        return allowed | {"resolve_session"}
    if not progress.gainer_chosen:
        return allowed | {"find_top_gainer"}
    if not progress.history_ready:
        allowed.add("get_price_history")
    if not progress.news_fetched:
        allowed.add("get_news")
    if progress.news_fetched and not progress.sentiment_recorded and has_headlines:
        allowed.add("record_sentiment")
    if progress.history_ready and not progress.verified and not progress.analysis_terminal:
        allowed.add("run_python")
        if progress.successful_run:
            allowed.add("verify_analysis")
    if progress.verified and not progress.composed:
        allowed.add("compose_report")
    if progress.composed:
        allowed.add("send_email")
    return allowed


class ToolGatingMiddleware(AgentMiddleware):
    """Offer the model only the tools whose preconditions hold, by filtering the request's tool list."""

    def __init__(self, ctx: RunContext):
        super().__init__()
        self.ctx = ctx

    def wrap_model_call(self, request, handler):
        allowed = allowed_tool_names(self.ctx.progress, has_headlines=bool(self.ctx.news and self.ctx.news.headlines))
        tools = [t for t in request.tools if t.name in allowed]
        log.info("offering tools: %s", [t.name for t in tools], extra={"event": "gating", "run_id": self.ctx.run_id})
        # request.override(tools=...) was confirmed end to end against the installed langchain 1.4.2
        # during the initial spike, before any tool code was written.
        return handler(request.override(tools=tools))


class RunDeadlineMiddleware(AgentMiddleware):
    """Ends the agent loop once this invocation has run for `deadline_seconds` (spec section 5's whole-run deadline,
    AGENT_RUN_DEADLINE_SECONDS; final fix wave A2).

    Checked before each model call, and the loop is ended the way ModelCallLimitMiddleware ends it: before_model
    returns a jump to the end with a closing AI message. The reason goes into ctx.errors, so the failure notice names
    it.

    The deadline counts per invocation. The clock starts when this middleware is built, which _setup does once per
    run_once or resume_run call, so a resume starts a fresh deadline. What bounds the total work across resumes is the
    per-run call caps: thread limits, see build_middleware. A tool call already running is not interrupted; the
    sandbox and the HTTP clients have their own time limits.
    """

    def __init__(self, ctx: RunContext, deadline_seconds: int, clock: Callable[[], float] | None = None):
        super().__init__()
        self.ctx = ctx
        self.deadline_seconds = deadline_seconds
        # Seconds on a monotonic clock. Resolved when built rather than at import, so a test can replace this
        # module's `time` without touching the process-wide clock.
        self._clock = clock if clock is not None else time.monotonic
        self._started = self._clock()

    @hook_config(can_jump_to=["end"])
    def before_model(self, state, runtime):
        if self._clock() - self._started < self.deadline_seconds:
            return None
        reason = RUN_DEADLINE_REACHED.format(seconds=self.deadline_seconds)
        log.error("%s; ending the agent loop before the next model call", reason,
                  extra={"event": "run_deadline", "run_id": self.ctx.run_id})
        self.ctx.errors.append(reason)
        self.ctx.save()
        return {"jump_to": "end", "messages": [AIMessage(content=f"Stopped: {reason} before the next model call.")]}


class PacingMiddleware(AgentMiddleware):
    """Sleep between model calls so a free-tier key is not rate limited."""

    def __init__(self, delay_seconds: float):
        super().__init__()
        self.delay_seconds = delay_seconds
        self._calls = 0

    def before_model(self, state, runtime):
        self._calls += 1
        if self.delay_seconds > 0 and self._calls > 1:
            time.sleep(self.delay_seconds)
        return None


class ToolCrashMiddleware(AgentMiddleware):
    """A bug inside a tool becomes an error message for the model instead of a crashed run. A replay CassetteMiss is
    the exception: it propagates (final fix wave A3).

    Also serializes tool execution for the run. create_agent's ToolNode dispatches one model
    turn's tool calls concurrently on a real thread pool (langgraph's `get_executor_for_config`
    returns a `ContextThreadPoolExecutor`; confirmed in `.venv/lib/.../langgraph/prebuilt/
    tool_node.py`), but RunContext is a single mutable object shared by every tool (the
    `tool_calls` counter, `analysis.attempts`, every `progress` flag, ...). Two tool bodies --
    most importantly two run_python calls -- running at once would race on that shared state
    and could blow past the per-run caps, which assume one call finishes before the next
    starts. The lock is created once in `__init__`, so one `ToolCrashMiddleware` instance (built
    once per `build_middleware` / `build_orchestrator` call, i.e. once per run) serializes every
    tool call of that run behind it, whichever thread the executor runs it on.
    """

    def __init__(self):
        super().__init__()
        self._lock = threading.Lock()

    def wrap_tool_call(self, request, handler):
        with self._lock:
            try:
                return handler(request)
            except CassetteMiss:
                # Not a tool bug for the model to work around: the recording cannot serve this replay. Propagating it
                # ends the run at the first miss -- agent_node records the error, finalize sends the failure notice --
                # instead of feeding the model an error it would answer with a prompt the cassette also lacks.
                raise
            except Exception as e:  # noqa: BLE001  deliberate: any tool bug must surface to the model, not kill the run
                log.exception("tool crashed", extra={"event": "tool_crash"})
                return ToolMessage(content=f"ERROR: tool crashed with {type(e).__name__}: {e}", tool_call_id=request.tool_call["id"])


# How many times per invocation a text-only reply is sent back to the model before the loop may end.
MAX_NUDGES = 2
NUDGE_MESSAGE = ("You replied without calling a tool, and the report has not been sent. Do not describe what you will "
                 "do: call one of the tools available now ({tools}), or give_up with a reason.")


class FinishTheRunMiddleware(AgentMiddleware):
    """A reply with no tool call ends create_agent's loop. A live run's model wrote "Let me try an alternative news
    source" instead of calling a tool, and the run stopped unsent. Until the report is sent or the agent gives up, a
    text-only reply earns a reminder and another turn, at most MAX_NUDGES times per invocation; the call caps and the
    run deadline still bound the loop, and the reminder text is fixed, so replay stays deterministic."""

    def __init__(self, ctx: RunContext):
        super().__init__()
        self.ctx = ctx
        self.nudges = 0

    @hook_config(can_jump_to=["model"])
    def after_model(self, state, runtime):
        messages = state.get("messages") or []
        last = messages[-1] if messages else None
        if not isinstance(last, AIMessage) or last.tool_calls:
            return None
        if self.ctx.progress.sent or self.ctx.progress.gave_up or self.nudges >= MAX_NUDGES:
            return None
        self.nudges += 1
        log.info("model replied without a tool call before the report was sent; reminder %d of %d", self.nudges,
                 MAX_NUDGES, extra={"event": "nudge", "run_id": self.ctx.run_id})
        # Name the tools on offer right now: a live run's model kept promising a tool that was no longer offered.
        available = sorted(allowed_tool_names(self.ctx.progress,
                                              has_headlines=bool(self.ctx.news and self.ctx.news.headlines)) - {"give_up"})
        return {"messages": [HumanMessage(NUDGE_MESSAGE.format(tools=", ".join(available) or "none"))], "jump_to": "model"}


def build_middleware(ctx: RunContext, settings: Settings) -> list[AgentMiddleware]:
    """The orchestrator's middleware. before_model hooks run in list order, so PacingMiddleware comes last: when the
    deadline or a call cap ends the loop, it ends without a pacing sleep first. (after_model hooks run in reverse list
    order; the two caps keep their relative order, so the tool-call check still runs after the model-call count.)

    Final fix wave A1: the call caps are thread limits. Spec section 10 caps calls per run; the orchestrator's thread
    id is the run id, and a thread count is checkpointed with the thread, so a resume spends what is left of the run's
    budget. A run limit would reset on every invocation.
    """
    return [ToolGatingMiddleware(ctx), ToolCrashMiddleware(), RunDeadlineMiddleware(ctx, settings.run_deadline_seconds),
            ToolCallLimitMiddleware(thread_limit=settings.max_tool_calls, exit_behavior="end"),
            ModelCallLimitMiddleware(thread_limit=settings.max_model_calls, exit_behavior="end"),
            PacingMiddleware(settings.llm_call_delay_seconds), FinishTheRunMiddleware(ctx)]
