import time
import pytest

def progress(**flags):
    from nasdaq_agent.agent.context import Progress
    return Progress(**flags)

def test_allowed_tools_follow_the_state():
    from nasdaq_agent.agent.middleware import allowed_tool_names as allowed
    assert allowed(progress()) == {"resolve_session", "give_up"}
    assert allowed(progress(session_resolved=True)) == {"find_top_gainer", "give_up"}
    assert allowed(progress(session_resolved=True, gainer_chosen=True)) == {"get_price_history", "get_news", "give_up"}
    p = progress(session_resolved=True, gainer_chosen=True, history_ready=True)
    assert "run_python" in allowed(p) and "verify_analysis" not in allowed(p) and "send_email" not in allowed(p)
    p.successful_run = True
    assert {"run_python", "verify_analysis"} <= allowed(p)
    p.verified = True
    assert "compose_report" in allowed(p) and "run_python" not in allowed(p)
    p.news_fetched = True
    assert "record_sentiment" in allowed(p)
    p.composed = True
    assert "send_email" in allowed(p) and "compose_report" not in allowed(p)
    p.sent = True
    assert allowed(p) == set()
    assert allowed(progress(session_resolved=True, gainer_chosen=True, history_ready=True, analysis_terminal=True)) == {"get_news", "give_up"}

class FakeTool:
    def __init__(self, name): self.name = name

class FakeRequest:
    def __init__(self, tools): self.tools = tools
    def override(self, **kw):
        r = FakeRequest(kw.get("tools", self.tools)); return r

def test_gating_middleware_filters_tools(tmp_path):
    from nasdaq_agent.agent.context import RunContext
    from nasdaq_agent.agent.middleware import ToolGatingMiddleware
    ctx = RunContext.new("r", "h", str(tmp_path))
    ctx.progress.session_resolved = True
    mw = ToolGatingMiddleware(ctx)
    req = FakeRequest([FakeTool(n) for n in ["resolve_session", "find_top_gainer", "send_email", "give_up"]])
    seen = {}
    def handler(r): seen["names"] = [t.name for t in r.tools]; return "resp"
    assert mw.wrap_model_call(req, handler) == "resp"
    assert seen["names"] == ["find_top_gainer", "give_up"]

def test_record_sentiment_is_not_offered_when_no_headlines_were_fetched():
    from nasdaq_agent.agent.middleware import allowed_tool_names as allowed
    p = progress(session_resolved=True, gainer_chosen=True, history_ready=True, news_fetched=True)
    assert "record_sentiment" in allowed(p) and "record_sentiment" in allowed(p, has_headlines=True)
    assert "record_sentiment" not in allowed(p, has_headlines=False)

def test_gating_middleware_hides_record_sentiment_when_the_news_has_no_headlines(tmp_path):
    from nasdaq_agent.agent.context import NewsInfo, RunContext
    from nasdaq_agent.agent.middleware import ToolGatingMiddleware
    ctx = RunContext.new("r", "h", str(tmp_path))
    for flag in ("session_resolved", "gainer_chosen", "history_ready", "news_fetched"):
        setattr(ctx.progress, flag, True)
    ctx.news = NewsInfo(headlines=[], sources=["massive"])
    seen = {}
    def handler(r): seen["names"] = [t.name for t in r.tools]; return "resp"
    ToolGatingMiddleware(ctx).wrap_model_call(FakeRequest([FakeTool(n) for n in ["record_sentiment", "run_python", "give_up"]]), handler)
    assert "record_sentiment" not in seen["names"] and "run_python" in seen["names"]

def test_finish_guard_reminds_a_text_only_reply_until_the_report_is_sent(tmp_path):
    # A live run: the model wrote "Let me try an alternative news source" instead of calling a tool, and a reply with
    # no tool call ended the loop unsent. Until the report is sent or the agent gives up, such a reply earns a
    # reminder and another turn, at most MAX_NUDGES times.
    from langchain_core.messages import AIMessage, HumanMessage
    from nasdaq_agent.agent.context import RunContext
    from nasdaq_agent.agent.middleware import MAX_NUDGES, FinishTheRunMiddleware
    guard = FinishTheRunMiddleware(RunContext.new("r", "h", str(tmp_path)))
    text_only = {"messages": [AIMessage(content="Let me try an alternative news source.")]}
    for _ in range(MAX_NUDGES):
        out = guard.after_model(text_only, None)
        assert out["jump_to"] == "model" and isinstance(out["messages"][0], HumanMessage)
    assert guard.after_model(text_only, None) is None

def test_finish_guard_reminder_names_the_tools_available_now(tmp_path):
    # A live run: after verification the model kept promising to fetch news, but get_news was no longer offered, and a
    # generic reminder did not tell it what it could call. The reminder lists the tools on offer right now.
    from langchain_core.messages import AIMessage
    from nasdaq_agent.agent.context import NewsInfo, RunContext
    from nasdaq_agent.agent.middleware import FinishTheRunMiddleware
    ctx = RunContext.new("r", "h", str(tmp_path))
    for flag in ("session_resolved", "gainer_chosen", "history_ready", "news_fetched", "successful_run", "verified"):
        setattr(ctx.progress, flag, True)
    ctx.news = NewsInfo(headlines=[], sources=["yfinance"])
    out = FinishTheRunMiddleware(ctx).after_model({"messages": [AIMessage(content="I'll try another news source.")]}, None)
    reminder = out["messages"][0].content
    assert "compose_report" in reminder and "give_up" in reminder and "get_news" not in reminder

# How each supported provider's LangChain client marks a reply that stopped at the output-token limit: OpenRouter
# (the OpenAI API), Google and Anthropic.
CUT_OFF_METADATA = [{"finish_reason": "length"}, {"finish_reason": "MAX_TOKENS"}, {"stop_reason": "max_tokens"}]

@pytest.mark.parametrize("metadata", CUT_OFF_METADATA)
def test_finish_guard_tells_a_reply_cut_off_at_the_output_limit_what_happened(tmp_path, metadata):
    # A live run: at the sentiment step the model spent its whole 8,192-token reply thinking and was cut off before it
    # called a tool, three times. Each reminder told it not to "describe what you will do", which it had not done.
    from langchain_core.messages import AIMessage
    from nasdaq_agent.agent.context import RunContext
    from nasdaq_agent.agent.middleware import FinishTheRunMiddleware
    out = FinishTheRunMiddleware(RunContext.new("r", "h", str(tmp_path))).after_model(
        {"messages": [AIMessage(content="", response_metadata=metadata)]}, None)
    reminder = out["messages"][0].content
    assert out["jump_to"] == "model"
    assert "output limit" in reminder and "Do not describe what you will do" not in reminder

@pytest.mark.parametrize("metadata", CUT_OFF_METADATA)
def test_cut_off_replies_that_end_the_run_name_the_output_limit_for_the_failure_notice(tmp_path, metadata):
    # The same run's failure notice gave its reason as "the agent ended without sending a report". The last reason in
    # ctx.errors is what the notice shows, so it names the limit and the setting that raises it.
    from langchain_core.messages import AIMessage
    from nasdaq_agent.agent.context import RunContext
    from nasdaq_agent.agent.middleware import MAX_NUDGES, FinishTheRunMiddleware
    ctx = RunContext.new("r", "h", str(tmp_path))
    guard = FinishTheRunMiddleware(ctx)
    cut_off = {"messages": [AIMessage(content="", response_metadata=metadata)]}
    for _ in range(MAX_NUDGES):
        assert guard.after_model(cut_off, None)["jump_to"] == "model"
    assert ctx.errors == []
    assert guard.after_model(cut_off, None) is None
    assert len(ctx.errors) == 1
    assert "output limit" in ctx.errors[0] and f"{MAX_NUDGES + 1} times" in ctx.errors[0]
    assert "AGENT_LLM_MAX_OUTPUT_TOKENS" in ctx.errors[0]

def test_text_only_replies_that_end_the_run_add_no_output_limit_error(tmp_path):
    from langchain_core.messages import AIMessage
    from nasdaq_agent.agent.context import RunContext
    from nasdaq_agent.agent.middleware import MAX_NUDGES, FinishTheRunMiddleware
    ctx = RunContext.new("r", "h", str(tmp_path))
    guard = FinishTheRunMiddleware(ctx)
    text_only = {"messages": [AIMessage(content="I am done.", response_metadata={"finish_reason": "stop"})]}
    for _ in range(MAX_NUDGES + 1):
        guard.after_model(text_only, None)
    assert ctx.errors == []

def test_finish_guard_leaves_tool_calls_and_finished_runs_alone(tmp_path):
    from langchain_core.messages import AIMessage
    from nasdaq_agent.agent.context import RunContext
    from nasdaq_agent.agent.middleware import FinishTheRunMiddleware
    ctx = RunContext.new("r", "h", str(tmp_path))
    guard = FinishTheRunMiddleware(ctx)
    call = AIMessage(content="", tool_calls=[{"name": "get_news", "args": {}, "id": "c1", "type": "tool_call"}])
    assert guard.after_model({"messages": [call]}, None) is None
    ctx.progress.sent = True
    assert guard.after_model({"messages": [AIMessage(content="Sent the report.")]}, None) is None
    ctx.progress.sent, ctx.progress.gave_up = False, True
    assert guard.after_model({"messages": [AIMessage(content="Gave up.")]}, None) is None

def test_tool_crash_middleware_converts_exception_to_tool_message(caplog):
    """ToolCrashMiddleware's crash-to-ToolMessage branch: a RuntimeError raised by the handler
    must come back as a ToolMessage naming the exception and the original tool_call_id, and must
    be logged at ERROR with exception info.

    Isolation note: tests/unit/test_artifacts.py's configure_logging() calls mutate the
    process-global "nasdaq_agent" parent logger -- attaching a SecretRedactingFilter-bearing
    handler and setting propagate=False -- and never restore it. A stdlib LogRecord is one
    mutable object shared by every handler in the propagation chain; when test_artifacts.py
    runs first (true for the full `pytest tests/unit` suite, where it sorts before this file),
    that leftover filter clears record.exc_info to None (moving the traceback to record.exc_text
    instead) before caplog's own handler ever sees the record. Confirmed by running this
    assertion without the save/restore below as part of the full suite: it failed with
    "exc_info=None exc_text='Traceback ...'". Saving and restoring "nasdaq_agent"'s handlers and
    propagate flag around the call under test makes the assertion correct regardless of what ran
    before it, instead of silently depending on test order."""
    import logging
    from types import SimpleNamespace
    from langchain_core.messages import ToolMessage
    from nasdaq_agent.agent.middleware import ToolCrashMiddleware

    parent = logging.getLogger("nasdaq_agent")
    saved_handlers, saved_propagate = list(parent.handlers), parent.propagate
    for h in saved_handlers:
        parent.removeHandler(h)
    parent.propagate = True
    try:
        mw = ToolCrashMiddleware()
        request = SimpleNamespace(tool_call={"id": "call_42"})  # minimal: wrap_tool_call's except branch reads only request.tool_call["id"]

        def crashing_handler(r):
            raise RuntimeError("boom")

        with caplog.at_level(logging.ERROR, logger="nasdaq_agent.middleware"):
            result = mw.wrap_tool_call(request, crashing_handler)

        assert isinstance(result, ToolMessage)
        assert result.content.startswith("ERROR: tool crashed with RuntimeError")
        assert result.tool_call_id == "call_42"

        records = [r for r in caplog.records if r.name == "nasdaq_agent.middleware"]
        assert len(records) == 1, records
        record = records[0]
        assert record.levelno == logging.ERROR
        assert record.exc_info is not None and record.exc_info[0] is RuntimeError
    finally:
        for h in list(parent.handlers):
            parent.removeHandler(h)
        for h in saved_handlers:
            parent.addHandler(h)
        parent.propagate = saved_propagate


def test_tool_crash_middleware_serializes_concurrent_tool_calls():
    """create_agent's ToolNode runs one turn's tool calls on a real thread pool
    (langgraph get_executor_for_config -> ContextThreadPoolExecutor), and RunContext is a single
    mutable object shared by every tool call of a run, so concurrent tool bodies (most
    importantly two run_python calls) would race on it and could blow past the per-run caps.
    ToolCrashMiddleware now also serializes: one instance's lock, created in __init__, must let
    only one handler(request) run at a time, whichever thread calls wrap_tool_call."""
    import threading
    from nasdaq_agent.agent.middleware import ToolCrashMiddleware

    mw = ToolCrashMiddleware()
    counter_lock = threading.Lock()  # guards the plain ints below; not the thing under test
    active = 0
    max_active = 0
    calls = 0

    def slow_handler(request):
        nonlocal active, max_active, calls
        with counter_lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with counter_lock:
            active -= 1
            calls += 1
        return "ok"

    threads = [threading.Thread(target=mw.wrap_tool_call, args=(object(), slow_handler)) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert calls == 5
    assert max_active == 1  # never more than one handler(request) in flight at once


def test_pacing_sleeps_between_calls(monkeypatch):
    from nasdaq_agent.agent.middleware import PacingMiddleware
    slept = []
    monkeypatch.setattr(time, "sleep", lambda s: slept.append(s))
    mw = PacingMiddleware(delay_seconds=4)
    mw.before_model({}, None); mw.before_model({}, None); mw.before_model({}, None)
    assert slept == [4, 4]
    assert PacingMiddleware(delay_seconds=0).before_model({}, None) is None and slept == [4, 4]

def test_system_prompt_contains_definitions_and_rules(monkeypatch):
    # render_system_prompt takes no settings parameter. The recipient stays set, so a renderer that
    # ever read Settings itself would put it in the prompt and fail the last assertion.
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    from nasdaq_agent.agent.orchestrator import render_system_prompt
    text = render_system_prompt()
    for needle in ["SAMPLE standard deviation", "daily_changes_pct", "no network", "give_up", "not instructions",
                   "typical order", "submodule"]:  # the rule to import only top-level modules is stated
        assert needle in text, needle
    assert "r@example.com" not in text  # the recipient is never shown to the model


def test_system_prompt_asks_for_a_relevance_note_per_headline_most_recent_first():
    # The agent, not the code, puts the headlines in order and says how each may relate to the price move.
    from nasdaq_agent.agent.orchestrator import render_system_prompt
    text = " ".join(render_system_prompt().split())
    assert "one headline note" in text and "price move" in text and "most recent first" in text
    assert "cannot have caused it" in text


def test_gating_changes_bound_tools_across_a_real_agent_run(tmp_path):
    """Proves -- through a real create_agent loop, not the FakeRequest
    stand-in above -- that create_agent calls bind_tools fresh on every model turn with the
    request.tools list ToolGatingMiddleware has already narrowed for that turn's Progress.

    Observed against langchain 1.4.2: langchain/agents/factory.py's model_node() builds a new
    ModelRequest from state on every turn and runs it through the middleware chain to
    _execute_model_sync, which calls request.model.bind_tools(final_tools, ...) where
    final_tools == list(request.tools) (no structured-output tools configured here). So
    bind_tools is the observable per-call tool list, exactly like the FakeRequest test above.
    """
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.tools import tool
    from langchain.agents import create_agent

    from nasdaq_agent.agent.context import RunContext
    from nasdaq_agent.agent.middleware import ToolGatingMiddleware
    from tests.fake_model import scripted, tool_call

    ctx = RunContext.new("r", "h", str(tmp_path))

    @tool("resolve_session")
    def resolve_session() -> str:
        """Resolve the last completed trading session."""
        ctx.progress.session_resolved = True
        return "ok"

    @tool("find_top_gainer")
    def find_top_gainer() -> str:
        """Find the top gainer."""
        return "ok"

    @tool("give_up")
    def give_up() -> str:
        """Stop the run."""
        return "ok"

    model = scripted([tool_call("resolve_session", {}, "1"), AIMessage(content="done")])
    agent = create_agent(model=model, tools=[resolve_session, find_top_gainer, give_up],
                         middleware=[ToolGatingMiddleware(ctx)])
    agent.invoke({"messages": [HumanMessage(content="go")]})

    assert len(model.bound_tool_names) == 2
    assert set(model.bound_tool_names[0]) == {"resolve_session", "give_up"}
    assert set(model.bound_tool_names[1]) == {"find_top_gainer", "give_up"}


class _SteppedClock:
    """A monotonic clock for tests: each read returns the next reading, and the last reading repeats."""

    def __init__(self, *readings: float):
        self._readings = list(readings)

    def __call__(self) -> float:
        return self._readings.pop(0) if len(self._readings) > 1 else self._readings[0]


def test_run_deadline_lets_model_calls_through_until_the_deadline(tmp_path):
    from nasdaq_agent.agent.context import RunContext
    from nasdaq_agent.agent.middleware import RunDeadlineMiddleware
    ctx = RunContext.new("r", "h", str(tmp_path))
    mw = RunDeadlineMiddleware(ctx, 600, clock=_SteppedClock(1000.0, 1599.9, 1600.0))
    assert mw.before_model({}, None) is None and ctx.errors == []  # 599.9 s into the invocation
    ended = mw.before_model({}, None)  # 600 s: the deadline is reached
    assert ended["jump_to"] == "end"
    assert "run deadline of 600 s reached" in ended["messages"][0].content
    assert ctx.errors == ["run deadline of 600 s reached"]


def test_run_deadline_is_checked_before_pacing_so_a_late_run_never_sleeps(tmp_path, monkeypatch):
    """Through build_middleware's own list and a real create_agent loop: the deadline passes after
    the first model call, so the second before_model must end the loop without PacingMiddleware sleeping first --
    before_model hooks run in list order. Only this module's `time` is replaced, never the process-wide clock."""
    from types import SimpleNamespace

    from langchain.agents import create_agent
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.tools import tool

    from nasdaq_agent.agent import middleware
    from nasdaq_agent.agent.context import RunContext
    from nasdaq_agent.config import Settings
    from tests.fake_model import scripted, tool_call

    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    settings = Settings(_env_file=None)
    assert settings.llm_call_delay_seconds > 0 and settings.run_deadline_seconds == 600
    slept: list[float] = []
    # Readings: the middleware is built (0), the first check (1), the second check (well past 600 s).
    monkeypatch.setattr(middleware, "time", SimpleNamespace(monotonic=_SteppedClock(0.0, 1.0, 10_000.0),
                                                            sleep=slept.append))
    ctx = RunContext.new("r", "h", str(tmp_path))

    @tool("resolve_session")
    def resolve_session() -> str:
        """Resolve the last completed trading session."""
        ctx.progress.session_resolved = True
        return "ok"

    @tool("give_up")
    def give_up() -> str:
        """Stop the run."""
        return "ok"

    model = scripted([tool_call("resolve_session", {}, "1"), AIMessage(content="must not be reached")])
    agent = create_agent(model=model, tools=[resolve_session, give_up], middleware=middleware.build_middleware(ctx, settings))
    result = agent.invoke({"messages": [HumanMessage(content="go")]})

    assert len(model.bound_tool_names) == 1  # one model call, then the deadline ended the loop
    assert slept == []
    assert "run deadline of 600 s reached" in result["messages"][-1].content
    assert ctx.errors == ["run deadline of 600 s reached"]


def test_a_model_call_cap_ends_the_loop_without_a_pacing_sleep_first(tmp_path, monkeypatch):
    """The deadline and both call caps sit before PacingMiddleware in build_middleware,
    so when max_model_calls is reached the loop ends without sleeping first. With max_model_calls=1 the cap ends the
    second before_model pass -- exactly where PacingMiddleware, if it came first, would sleep (it sleeps before every
    model call after the first). Only this module's `time` is replaced."""
    from types import SimpleNamespace

    from langchain.agents import create_agent
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.tools import tool

    from nasdaq_agent.agent import middleware
    from nasdaq_agent.agent.context import RunContext
    from nasdaq_agent.config import Settings
    from tests.fake_model import scripted, tool_call

    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    monkeypatch.setenv("AGENT_MAX_MODEL_CALLS", "1")
    settings = Settings(_env_file=None)
    assert settings.llm_call_delay_seconds > 0
    slept: list[float] = []
    monkeypatch.setattr(middleware, "time", SimpleNamespace(monotonic=time.monotonic, sleep=slept.append))
    ctx = RunContext.new("r", "h", str(tmp_path))

    @tool("resolve_session")
    def resolve_session() -> str:
        """Resolve the last completed trading session."""
        ctx.progress.session_resolved = True
        return "ok"

    @tool("give_up")
    def give_up() -> str:
        """Stop the run."""
        return "ok"

    model = scripted([tool_call("resolve_session", {}, "1"), AIMessage(content="must not be reached")])
    agent = create_agent(model=model, tools=[resolve_session, give_up], middleware=middleware.build_middleware(ctx, settings))
    result = agent.invoke({"messages": [HumanMessage(content="go")]})

    assert len(model.bound_tool_names) == 1
    assert "Model call limits exceeded" in result["messages"][-1].content
    assert slept == []


def test_tool_crash_middleware_lets_a_replay_cassette_miss_escape():
    """A CassetteMiss is not a tool bug for the model to work around -- the recording cannot serve
    this replay -- so it is re-raised, not turned into a tool error message, and the lock is released."""
    from types import SimpleNamespace

    from nasdaq_agent.agent.middleware import ToolCrashMiddleware
    from nasdaq_agent.sources.errors import CassetteMiss

    mw = ToolCrashMiddleware()
    request = SimpleNamespace(tool_call={"id": "call_7"})

    def missing(r):
        raise CassetteMiss("no recording for GET https://example.test/quote")

    with pytest.raises(CassetteMiss, match="no recording"):
        mw.wrap_tool_call(request, missing)
    assert mw.wrap_tool_call(request, lambda r: "next call runs") == "next call runs"


def test_a_cassette_miss_inside_a_tool_ends_the_agent_loop_at_the_first_miss():
    """Through a real create_agent loop: a CassetteMiss inside a tool propagates out of agent.invoke, and the model is
    never asked to react to it."""
    from langchain.agents import create_agent
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.tools import tool

    from nasdaq_agent.agent.middleware import ToolCrashMiddleware
    from nasdaq_agent.sources.errors import CassetteMiss
    from tests.fake_model import scripted, tool_call

    @tool("resolve_session")
    def resolve_session() -> str:
        """Resolve the last completed trading session."""
        raise CassetteMiss("no recording for GET https://example.test/calendar")

    model = scripted([tool_call("resolve_session", {}, "1"), AIMessage(content="must not be reached")])
    agent = create_agent(model=model, tools=[resolve_session], middleware=[ToolCrashMiddleware()])
    with pytest.raises(CassetteMiss, match="calendar"):
        agent.invoke({"messages": [HumanMessage(content="go")]})
    assert len(model.bound_tool_names) == 1
