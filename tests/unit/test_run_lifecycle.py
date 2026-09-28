"""Fix round 1, item 5: run_once/resume_run lifecycle tests. All offline -- no network, no real
model. The orchestrator model is always tests/fake_model.py's scripted([...]) (or a model that
raises if it's ever touched); deps come from a fakes-backed deps_factory using a real, local-disk
FileTransport. Sqlite connections opened by graph.py are recorded via a wrapper around
nasdaq_agent.agent.graph.sqlite3.connect, and "closed" is verified the way the task plan specifies:
conn.execute("select 1") raises sqlite3.ProgrammingError once closed.
"""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from tests.fake_model import scripted, tool_call



def _ends_without_tools(text):
    """A model that only ever replies with text: the finish guard reminds it MAX_NUDGES times, then the loop ends."""
    from nasdaq_agent.agent.middleware import MAX_NUDGES
    return scripted([AIMessage(content=f"{text} ({i})") for i in range(MAX_NUDGES + 1)])

def _settings(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    monkeypatch.setenv("AGENT_ARTIFACTS_DIR", str(tmp_path / "runs"))
    # Several tests here script more than one model turn, and PacingMiddleware sleeps before every
    # model call after the first -- no pacing delay keeps them fast.
    monkeypatch.setenv("AGENT_LLM_CALL_DELAY_SECONDS", "0")
    from nasdaq_agent.config import Settings
    return Settings(_env_file=None)


def _fake_deps_factory(settings, ctx, run_dir, clock, cassette=None):
    from nasdaq_agent.agent.tools.common import Deps
    from nasdaq_agent.email.transports import FileTransport
    from nasdaq_agent.universe import Universe
    from tests.unit.test_universe import SAMPLE
    return Deps(settings=settings, universe=Universe.from_text(SAMPLE), gainer_sources=[], history_sources=[],
                news_sources=[], runner=None, transport=FileTransport(run_dir.outbox_dir),
                judge=lambda *a, **k: None, clock=clock, run_dir=run_dir)


def _record_connections(monkeypatch):
    from nasdaq_agent.agent import graph as graph_module
    connections: list[sqlite3.Connection] = []
    real_connect = graph_module.sqlite3.connect

    def wrapper(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        connections.append(conn)
        return conn

    monkeypatch.setattr(graph_module.sqlite3, "connect", wrapper)
    return connections


def _is_closed(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("select 1")
        return False
    except sqlite3.ProgrammingError:
        return True


# Task 23 correction h: resume_run accepts only new_run_id's format (and only a run whose context.json exists), so
# the hand-made runs resumed below use well-formed ids.
ALREADY_SENT_RUN_ID = "20260924T220000Z-0a0001"
CRASH_WINDOW_RUN_ID = "20260924T220000Z-0a0002"
STALE_SUMMARY_RUN_ID = "20260924T220000Z-0a0003"


class _ExplodingModel:
    """A model that fails loudly if the agent loop is ever entered, for tests where it must not
    be -- an already-resolved resume should never touch the model at all."""

    def bind_tools(self, tools, **kwargs):
        raise AssertionError("model must not be used: the run was already resolved")

    def invoke(self, *args, **kwargs):
        raise AssertionError("model must not be used: the run was already resolved")

    def with_structured_output(self, *args, **kwargs):
        raise AssertionError("model must not be used: the run was already resolved")


# --- Item 1 -----------------------------------------------------------------------------------

def test_run_once_model_ends_without_tools_sends_one_failure_notice(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    connections = _record_connections(monkeypatch)
    from nasdaq_agent.agent.graph import run_once

    model = _ends_without_tools("Nothing notable today.")
    outcome = run_once(settings, model=model, deps_factory=_fake_deps_factory)

    assert outcome.exit_code == 1
    run_dir_path = Path(outcome.artifacts_path)
    eml_files = list((run_dir_path / "outbox").glob("*.eml"))
    assert len(eml_files) == 1
    summary = json.loads((run_dir_path / "summary.json").read_text())
    assert summary["exit_code"] == 1
    assert connections and all(_is_closed(c) for c in connections)


# --- Item 2 -----------------------------------------------------------------------------------

def test_resume_run_already_sent_never_calls_model(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    connections = _record_connections(monkeypatch)
    from nasdaq_agent.agent.context import RunContext
    from nasdaq_agent.agent.graph import resume_run
    from nasdaq_agent.artifacts import RunDir

    run_id = ALREADY_SENT_RUN_ID
    run_dir = RunDir(tmp_path / "runs", run_id)
    ctx = RunContext.new(run_id, "h", str(run_dir.path))
    ctx.progress.sent = True
    ctx.save()

    outcome = resume_run(settings, run_id, model=_ExplodingModel(), deps_factory=_fake_deps_factory)

    assert outcome.exit_code == 0
    assert list(run_dir.outbox_dir.glob("*.eml")) == []
    # _setup (which would build/use the model) is never reached for an already-resolved resume,
    # so no sqlite connection is ever opened either.
    assert connections == []


# --- Item 3 -----------------------------------------------------------------------------------

def test_resume_run_reconciles_sent_marker_without_reentering_agent_loop(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    connections = _record_connections(monkeypatch)
    from nasdaq_agent.agent.context import RunContext
    from nasdaq_agent.agent.graph import resume_run
    from nasdaq_agent.agent.tools.send import SENT_MARKER
    from nasdaq_agent.artifacts import RunDir
    from nasdaq_agent.email.idempotency import SendMarker
    from nasdaq_agent.email.transports import SendResult

    run_id = CRASH_WINDOW_RUN_ID
    run_dir = RunDir(tmp_path / "runs", run_id)
    ctx = RunContext.new(run_id, "h", str(run_dir.path))
    ctx.save()  # progress.sent is False: the crash happened before this was ever persisted

    marker = SendMarker(run_dir.path / SENT_MARKER)
    marker.begin()
    marker.complete(SendResult(transport="file", message_id="<abc@run>", location=str(run_dir.path / "sent.eml")))

    outcome = resume_run(settings, run_id, model=_ExplodingModel(), deps_factory=_fake_deps_factory)

    assert outcome.exit_code == 0
    assert list(run_dir.outbox_dir.glob("*.eml")) == []  # no failure notice
    summary = json.loads((run_dir.path / "summary.json").read_text())
    assert summary["sent"] is True

    reloaded = RunContext.load(run_dir.path)
    assert reloaded.progress.sent is True
    assert "delivery reconciled from the send marker after an interrupted run" in reloaded.notes
    assert reloaded.email is not None and reloaded.email.message_id == "<abc@run>"
    assert connections == []


# --- Item 5 -----------------------------------------------------------------------------------

def test_run_once_setup_failure_sends_notice_without_raising(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    connections = _record_connections(monkeypatch)
    from nasdaq_agent.agent.graph import run_once
    from nasdaq_agent.universe import UniverseError

    def _raising_deps_factory(settings, ctx, run_dir, clock, cassette=None):
        raise UniverseError("nasdaqtrader unreachable")

    outcome = run_once(settings, deps_factory=_raising_deps_factory)

    assert outcome.exit_code == 1
    run_dir_path = Path(outcome.artifacts_path)
    eml_files = list((run_dir_path / "outbox").glob("*.eml"))
    assert len(eml_files) == 1
    assert "setup failed: UniverseError" in eml_files[0].read_bytes().decode()
    summary = json.loads((run_dir_path / "summary.json").read_text())
    assert summary["exit_code"] == 1
    # deps_factory raised before the sqlite connection would have been opened (it's the last
    # setup step, opened only after deps and the model exist).
    assert connections == []


# --- Item 6 -----------------------------------------------------------------------------------

def test_run_once_orchestrator_construction_failure_closes_connection(tmp_path, monkeypatch):
    settings = _settings(tmp_path, monkeypatch)
    connections = _record_connections(monkeypatch)
    from nasdaq_agent.agent import graph as graph_module

    def _raising_build_orchestrator(ctx, deps, model, checkpointer=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(graph_module, "build_orchestrator", _raising_build_orchestrator)

    model = scripted([AIMessage(content="unused")])
    outcome = graph_module.run_once(settings, model=model, deps_factory=_fake_deps_factory)

    assert outcome.exit_code == 1
    run_dir_path = Path(outcome.artifacts_path)
    eml_files = list((run_dir_path / "outbox").glob("*.eml"))
    assert len(eml_files) == 1
    assert "setup failed: RuntimeError" in eml_files[0].read_bytes().decode()
    assert connections and all(_is_closed(c) for c in connections)


# --- Fix round 2, item 1: an outcome recorded during THIS invocation, not summary.json ---------

def test_stale_summary_json_never_misreports_a_report_delivered_this_invocation(tmp_path, monkeypatch):
    """Fix round 2, item 1(a): _handle_graph_invoke_failure must never read summary.json, because
    on a resume, summary.json from an EARLIER invocation of the same run already exists. Simulates
    the task plan's failing sequence directly: a stale summary.json (exit_code 1) already exists; THIS
    invocation's agent step marks the report sent; graph.invoke then raises before finalize_node
    can record anything in the outcome holder. The fallback (recorded as "run failed", which calls
    finalize_with) must still land on the correct, CURRENT outcome -- 0, no notice, summary.json
    rewritten -- by reconciling delivery, never by trusting the stale file.
    """
    settings = _settings(tmp_path, monkeypatch)
    from nasdaq_agent.agent import graph as graph_module
    from nasdaq_agent.agent.context import RunContext
    from nasdaq_agent.agent.tools.send import SENT_MARKER
    from nasdaq_agent.artifacts import RunDir
    from nasdaq_agent.email.idempotency import SendMarker
    from nasdaq_agent.email.transports import SendResult

    run_id = STALE_SUMMARY_RUN_ID
    seed_run_dir = RunDir(tmp_path / "runs", run_id)
    RunContext.new(run_id, "h", str(seed_run_dir.path)).save()  # correction h: only an existing run can be resumed
    seed_run_dir.write_json("summary.json", {"run_id": run_id, "exit_code": 1, "sent": False, "gainer": None,
                                             "degradations": [], "errors": [], "tool_calls": 0, "give_up_reason": None})

    def _fake_build_orchestrator(ctx, deps, model, checkpointer=None):
        class _SendsThenReturns:
            def invoke(self, inp, config=None, **kwargs):
                marker = SendMarker(deps.run_dir.path / SENT_MARKER)
                marker.begin()
                marker.complete(SendResult(transport="file", message_id="<sent-this-invocation@run>", location="x"))
                ctx.progress.sent = True
                return {"messages": []}
        return _SendsThenReturns()

    def _raising_finalize(ctx, deps):
        raise RuntimeError("crash right at finalize")

    monkeypatch.setattr(graph_module, "build_orchestrator", _fake_build_orchestrator)
    monkeypatch.setattr(graph_module, "finalize", _raising_finalize)

    outcome = graph_module.resume_run(settings, run_id, model=object(), deps_factory=_fake_deps_factory)

    assert outcome.exit_code == 0
    run_dir_path = Path(outcome.artifacts_path)
    assert list((run_dir_path / "outbox").glob("*.eml")) == []
    summary = json.loads((run_dir_path / "summary.json").read_text())
    assert summary["exit_code"] == 0


def test_graph_invoke_failure_after_finalize_returns_the_recorded_exit_code(tmp_path, monkeypatch):
    """Fix round 2, item 1(b): once finalize_node has recorded an exit code in the outcome holder,
    a LATER graph.invoke failure -- here, the checkpoint write for finalize's own step-completion,
    simulating the shape of the round-1 failure mode without needing a broken package -- must
    return that recorded code directly, and must never fall through to the "run failed" fallback
    (which would call finalize_with a redundant second time; asserted directly by counting calls
    to graph_module.finalize_with, which run_once reaches only through its failure paths).
    """
    settings = _settings(tmp_path, monkeypatch)
    connections = _record_connections(monkeypatch)
    from nasdaq_agent.agent import graph as graph_module
    from langgraph.checkpoint.sqlite import SqliteSaver as RealSqliteSaver

    finalize_ran = {"done": False}
    real_finalize = graph_module.finalize

    def _finalize_then_flag(ctx, deps):
        code = real_finalize(ctx, deps)
        finalize_ran["done"] = True
        return code

    finalize_with_calls = {"count": 0}
    real_finalize_with = graph_module.finalize_with

    def _counting_finalize_with(*args, **kwargs):
        finalize_with_calls["count"] += 1
        return real_finalize_with(*args, **kwargs)

    def _raising_after_finalize_saver(conn, *, serde=None):
        # A thin wrapper around the REAL SqliteSaver: everything before finalize completes goes
        # through untouched (so this run's own checkpoints are genuinely persisted), and only
        # `put` calls made after finalize_ran["done"] becomes True raise -- simulating a
        # checkpoint-persistence failure for finalize's own step, landing after its Python code
        # (and the outcome holder write) already completed.
        saver = RealSqliteSaver(conn, serde=serde)
        original_put = saver.put

        def _put(*args, **kwargs):
            if finalize_ran["done"]:
                raise RuntimeError("simulated checkpoint persistence failure after finalize")
            return original_put(*args, **kwargs)

        saver.put = _put
        return saver

    monkeypatch.setattr(graph_module, "finalize", _finalize_then_flag)
    monkeypatch.setattr(graph_module, "finalize_with", _counting_finalize_with)
    monkeypatch.setattr(graph_module, "SqliteSaver", _raising_after_finalize_saver)

    model = _ends_without_tools("Nothing notable today.")
    outcome = graph_module.run_once(settings, model=model, deps_factory=_fake_deps_factory)

    assert outcome.exit_code == 1  # the model never sends -> finalize's own real computation is 1
    assert finalize_with_calls["count"] == 0  # never fell through to the "run failed" fallback
    assert connections and all(_is_closed(c) for c in connections)


# --- Fix round 2, item 2: checkpoints genuinely persist -----------------------------------------

def test_checkpoints_persist_for_both_the_outer_graph_and_the_orchestrator_threads(tmp_path, monkeypatch):
    """Fix round 2, item 2: after a real run_once, open a FRESH sqlite connection to the run's
    checkpoints.sqlite (not the one graph.py itself used -- proving the data is genuinely durable
    on disk, not merely visible through the same in-process connection/object), build a
    SqliteSaver on it, and list checkpoints for both threads a run touches: the outer graph's
    "graph-<run_id>" and the orchestrator's own "<run_id>".
    """
    settings = _settings(tmp_path, monkeypatch)
    from langgraph.checkpoint.sqlite import SqliteSaver
    from nasdaq_agent.agent.graph import CHECKPOINT_DB, run_once

    model = _ends_without_tools("Nothing notable today.")
    outcome = run_once(settings, model=model, deps_factory=_fake_deps_factory)

    db_path = Path(outcome.artifacts_path) / CHECKPOINT_DB
    conn = sqlite3.connect(str(db_path))
    try:
        saver = SqliteSaver(conn)
        outer_checkpoints = list(saver.list({"configurable": {"thread_id": f"graph-{outcome.run_id}"}}))
        orchestrator_checkpoints = list(saver.list({"configurable": {"thread_id": outcome.run_id}}))
    finally:
        conn.close()

    assert len(outer_checkpoints) >= 1
    assert len(orchestrator_checkpoints) >= 1


# --- Fix round 2, item 3: resume from the checkpoint after a simulated crash --------------------

class _KilledByCrash(BaseException):
    """Simulates an external kill signal (SystemExit/KeyboardInterrupt-like): deliberately a
    BaseException subclass, not an Exception, so nothing in the normal error-handling chain --
    ToolCrashMiddleware, agent_node, run_once -- catches it, exactly as a real process kill would
    not be "caught" by any of them.
    """


class _CrashesOnWideWindowOnly:
    """Wraps a real, working history source but raises _KilledByCrash for a WIDE query (more than
    5 calendar days between start and end) -- get_price_history's own multi-session fetch -- while
    delegating narrow queries to the real source. find_top_gainer ALSO consults history_sources
    (_two_closes, reconciling a candidate's reported pct_change against real close-to-close bars,
    for a narrow 1-3 day window), and a source that crashed unconditionally would crash there
    first instead, before get_price_history is ever reached. Isolating the simulated crash to the
    wide query is what actually reproduces the spec's scenario: resolve_session and
    find_top_gainer complete normally, and it is specifically get_price_history's tool call that
    is killed mid-execution.
    """
    def __init__(self, working_source):
        self._working = working_source
        self.name = working_source.name

    def bars(self, symbol, start, end):
        if (end - start).days > 5:
            raise _KilledByCrash("simulated process kill during get_price_history")
        return self._working.bars(symbol, start, end)

    def corporate_actions(self, symbol, start, end):
        return self._working.corporate_actions(symbol, start, end)


def _agent_flow_universe():
    from nasdaq_agent.universe import Universe
    return Universe.from_text("Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size|ETF|NextShares\n"
                              "ACME|Acme Corp - Common Stock|Q|N|N|100|N|N\nSPY|SPDR S&P 500|G|N|N|100|Y|N\n")


def _agent_flow_judge():
    from nasdaq_agent.report.schemas import JudgeVerdict
    return lambda v, h, n, extra_facts=None, session_facts=None: JudgeVerdict(faithful=True)


def _crashing_deps_factory(settings, ctx, run_dir, clock, cassette=None):
    from nasdaq_agent.agent.tools.common import Deps
    from nasdaq_agent.email.transports import FileTransport
    from tests import fakes
    from tests.unit.test_grounding import headlines
    working = fakes.FakeHistorySource(data={"ACME": fakes.series("ACME", fakes.CLOSES), "SPY": fakes.series("SPY", fakes.BENCH)})
    return Deps(settings=settings, universe=_agent_flow_universe(),
                gainer_sources=[fakes.FakeGainerSource("massive", [fakes.acme_candidate()])],
                history_sources=[_CrashesOnWideWindowOnly(working)], news_sources=[fakes.FakeNewsSource(headlines=headlines())],
                runner=None, transport=FileTransport(run_dir.outbox_dir), judge=_agent_flow_judge(),
                clock=clock, run_dir=run_dir)


def _working_deps_factory(settings, ctx, run_dir, clock, cassette=None):
    from nasdaq_agent.agent.tools.common import Deps
    from nasdaq_agent.email.transports import FileTransport
    from tests import fakes
    from tests.unit.test_grounding import headlines
    from tests.unit.test_tools_agent import FakeRunner, success
    hist = fakes.FakeHistorySource(data={"ACME": fakes.series("ACME", fakes.CLOSES), "SPY": fakes.series("SPY", fakes.BENCH)})
    return Deps(settings=settings, universe=_agent_flow_universe(),
                gainer_sources=[fakes.FakeGainerSource("massive", [fakes.acme_candidate()])],
                history_sources=[hist], news_sources=[fakes.FakeNewsSource(headlines=headlines())],
                runner=FakeRunner([success()]), transport=FileTransport(run_dir.outbox_dir), judge=_agent_flow_judge(),
                clock=clock, run_dir=run_dir)


def test_resume_continues_the_checkpoint_after_a_simulated_crash(tmp_path, monkeypatch):
    """Fix round 2, item 3 (spec section 11): a simulated crash mid-tool-call, then a resume that
    continues from the checkpoint rather than restarting the conversation from "init". Exactly one
    email must ever be sent (and no failure notice), resolve_session/find_top_gainer must each
    show up exactly once in tool_log.jsonl (never replayed on resume), and get_price_history must
    have a successful entry (its retry, after the resume, against a working history source).
    """
    settings = _settings(tmp_path, monkeypatch)
    from nasdaq_agent.agent.graph import resume_run, run_once
    from tests.unit.test_grounding import good_narrative

    now = datetime(2026, 9, 24, 22, 0, tzinfo=timezone.utc)

    first_model = scripted([
        tool_call("resolve_session", {}, "1"),
        tool_call("find_top_gainer", {"source": "auto"}, "2"),
        tool_call("get_price_history", {"source": "auto"}, "3"),
    ])

    with pytest.raises(_KilledByCrash):
        run_once(settings, now=now, model=first_model, deps_factory=_crashing_deps_factory)

    # run_once always mints its own fresh run_id; recover the one it just used.
    runs_root = tmp_path / "runs"
    run_id = next(p.name for p in runs_root.iterdir() if p.is_dir())

    second_model = scripted([
        tool_call("run_python", {"code": "result = {}"}, "4"),
        tool_call("verify_analysis", {}, "5"),
        tool_call("get_news", {"source": "auto"}, "6"),
        tool_call("record_sentiment", {"score": 0.5, "label": "positive", "rationale": "guidance raised",
                                        "per_headline": [{"headline_id": 1, "score": 0.5}]}, "7"),
        tool_call("compose_report", good_narrative().model_dump(), "8"),
        tool_call("send_email", {}, "9"),
        AIMessage(content="Report sent."),
    ])

    outcome = resume_run(settings, run_id, model=second_model, deps_factory=_working_deps_factory)

    assert outcome.exit_code == 0
    run_dir_path = Path(outcome.artifacts_path)
    eml_files = list((run_dir_path / "outbox").glob("*.eml"))
    assert len(eml_files) == 1  # exactly one report email, ever
    assert not (run_dir_path / "failure_sent.json").exists()

    entries = [json.loads(line) for line in (run_dir_path / "tool_log.jsonl").read_text().splitlines()]
    tool_names = [e["tool"] for e in entries]
    assert tool_names.count("resolve_session") == 1
    assert tool_names.count("find_top_gainer") == 1
    successful_history_calls = [e for e in entries if e["tool"] == "get_price_history" and not e["outcome"].startswith("ERROR")]
    assert len(successful_history_calls) == 1


# --- Final fix wave, part A ------------------------------------------------------------------------------------------

def _only_run_id(tmp_path) -> str:
    """run_once mints its own run id; recover it from the one run directory a test created."""
    (run_id,) = [p.name for p in (tmp_path / "runs").iterdir() if p.is_dir()]
    return run_id


def _notice_body(run_dir_path: Path) -> str:
    """The plain part of the run's one failure notice, decoded: a body line longer than 78 characters is sent
    quoted-printable, whose soft line breaks can split a string in the raw bytes."""
    import email
    from email import policy
    (eml_path,) = (run_dir_path / "outbox").glob("*.eml")
    return email.message_from_bytes(eml_path.read_bytes(), policy=policy.default).get_body(("plain",)).get_content()


class _GainerSourceKilledMidCall:
    """A gainer source whose call is killed, so find_top_gainer's tool call never finishes."""
    name = "massive"
    requires_market_closed = False

    def top_candidates(self, session_date, prev_session_date, limit):
        raise _KilledByCrash("simulated process kill during find_top_gainer")


def _budget_deps_factory(gainer_source_factory):
    """Deps for the call-budget test: a working history and news chain, no code runner (the test never reaches
    run_python), and the gainer source the caller chooses."""
    def factory(settings, ctx, run_dir, clock, cassette=None):
        from nasdaq_agent.agent.tools.common import Deps
        from nasdaq_agent.email.transports import FileTransport
        from tests import fakes
        from tests.unit.test_grounding import headlines
        hist = fakes.FakeHistorySource(data={"ACME": fakes.series("ACME", fakes.CLOSES),
                                             "SPY": fakes.series("SPY", fakes.BENCH)})
        return Deps(settings=settings, universe=_agent_flow_universe(), gainer_sources=[gainer_source_factory()],
                    history_sources=[hist], news_sources=[fakes.FakeNewsSource(headlines=headlines())], runner=None,
                    transport=FileTransport(run_dir.outbox_dir), judge=_agent_flow_judge(), clock=clock,
                    run_dir=run_dir)
    return factory


def test_a_resume_shares_the_runs_model_call_budget(tmp_path, monkeypatch):
    """Final fix wave A1: the caps are per run (spec section 10), so they are thread limits on the orchestrator's
    thread, whose id is the run id; a run limit would reset on every resume. With max_model_calls=3, the first
    invocation spends 2 model calls and is killed mid-tool-call. The resume re-runs the pending find_top_gainer call,
    gets exactly 1 more model call, and the loop then ends at the cap."""
    from tests import fakes
    settings = _settings(tmp_path, monkeypatch).model_copy(update={"max_model_calls": 3})
    from nasdaq_agent.agent.graph import resume_run, run_once

    first_model = scripted([tool_call("resolve_session", {}, "1"), tool_call("find_top_gainer", {"source": "auto"}, "2")])
    with pytest.raises(_KilledByCrash):
        run_once(settings, now=datetime(2026, 9, 24, 22, 0, tzinfo=timezone.utc), model=first_model,
                 deps_factory=_budget_deps_factory(_GainerSourceKilledMidCall))
    assert len(first_model.bound_tool_names) == 2

    second_model = scripted([
        tool_call("get_price_history", {"source": "auto"}, "3"),
        tool_call("get_news", {"source": "auto"}, "4"),
        AIMessage(content="must not be reached: the run's model-call budget is spent"),
    ])
    working_gainers = lambda: fakes.FakeGainerSource("massive", [fakes.acme_candidate()])  # noqa: E731
    outcome = resume_run(settings, _only_run_id(tmp_path), model=second_model,
                         deps_factory=_budget_deps_factory(working_gainers))

    assert len(second_model.bound_tool_names) == 1
    run_dir_path = Path(outcome.artifacts_path)
    assert "thread limit (3/3)" in (run_dir_path / "closing_summary.txt").read_text()
    entries = [json.loads(line) for line in (run_dir_path / "tool_log.jsonl").read_text().splitlines()]
    assert [e["tool"] for e in entries] == ["resolve_session", "find_top_gainer", "get_price_history"]
    assert outcome.exit_code == 1  # the budget ran out before a report was sent


def test_a_resume_shares_the_runs_tool_call_budget(tmp_path, monkeypatch):
    """Final fix wave A1, the tool-call cap: ToolCallLimitMiddleware counts the tool calls a model turn proposes. With
    max_tool_calls=3, the first invocation proposes 2 and is killed while running the second. The resume re-runs that
    pending call without counting it again, allows get_price_history (the 3rd), and ends the loop instead of running
    get_news (the 4th)."""
    from tests import fakes
    settings = _settings(tmp_path, monkeypatch).model_copy(update={"max_tool_calls": 3})
    from nasdaq_agent.agent.graph import resume_run, run_once

    first_model = scripted([tool_call("resolve_session", {}, "1"), tool_call("find_top_gainer", {"source": "auto"}, "2")])
    with pytest.raises(_KilledByCrash):
        run_once(settings, now=datetime(2026, 9, 24, 22, 0, tzinfo=timezone.utc), model=first_model,
                 deps_factory=_budget_deps_factory(_GainerSourceKilledMidCall))

    second_model = scripted([
        tool_call("get_price_history", {"source": "auto"}, "3"),
        tool_call("get_news", {"source": "auto"}, "4"),
        AIMessage(content="must not be reached: the run's tool-call budget is spent"),
    ])
    working_gainers = lambda: fakes.FakeGainerSource("massive", [fakes.acme_candidate()])  # noqa: E731
    outcome = resume_run(settings, _only_run_id(tmp_path), model=second_model,
                         deps_factory=_budget_deps_factory(working_gainers))

    run_dir_path = Path(outcome.artifacts_path)
    entries = [json.loads(line) for line in (run_dir_path / "tool_log.jsonl").read_text().splitlines()]
    assert [e["tool"] for e in entries] == ["resolve_session", "find_top_gainer", "get_price_history"]
    assert "thread limit exceeded (4/3 calls)" in (run_dir_path / "closing_summary.txt").read_text()
    assert outcome.exit_code == 1


def test_a_run_past_its_deadline_ends_before_the_first_model_call(tmp_path, monkeypatch):
    """Final fix wave A2: the deadline is checked before each model call. Here it has already passed at the first
    check, so the model is never called; the run exits 1 and the failure notice names the deadline. Only the
    middleware module's `time` is replaced (built at 0 s, checked at 10,000 s), never the process-wide clock."""
    import time
    from types import SimpleNamespace
    settings = _settings(tmp_path, monkeypatch)
    from nasdaq_agent.agent import middleware
    from nasdaq_agent.agent.graph import run_once
    readings = iter([0.0])
    monkeypatch.setattr(middleware, "time", SimpleNamespace(monotonic=lambda: next(readings, 10_000.0),
                                                            sleep=time.sleep))
    model = scripted([AIMessage(content="must not be reached: the deadline has passed")])

    outcome = run_once(settings, model=model, deps_factory=_fake_deps_factory)

    assert outcome.exit_code == 1
    assert model.bound_tool_names == []
    run_dir_path = Path(outcome.artifacts_path)
    assert "Reason: run deadline of 600 s reached" in _notice_body(run_dir_path)
    summary = json.loads((run_dir_path / "summary.json").read_text())
    assert summary["exit_code"] == 1 and "run deadline of 600 s reached" in summary["errors"]


def test_agent_node_detects_a_pending_orchestrator_task_by_tasks_not_next():
    """Final fix wave A4: StateSnapshot.next leaves out a task whose writes were saved before a kill, so the
    orchestrator thread's pending work is read from .tasks. A thread with a pending task is continued with
    invoke(None); a finished thread gets the kickoff; a fake without get_state counts as finished."""
    from types import SimpleNamespace

    from nasdaq_agent.agent.graph import build_graph
    from nasdaq_agent.agent.orchestrator import KICKOFF_MESSAGE

    class _Orchestrator:
        def __init__(self):
            self.inputs = []

        def invoke(self, inp, config=None, **kwargs):
            self.inputs.append(inp)
            return {"messages": []}

    class _WithState(_Orchestrator):
        def __init__(self, tasks):
            super().__init__()
            self._tasks = tasks

        def get_state(self, config):
            return SimpleNamespace(next=(), tasks=self._tasks)

    pending, finished, stateless = _WithState(tasks=("tools",)), _WithState(tasks=()), _Orchestrator()
    for orchestrator in (pending, finished, stateless):
        build_graph(orchestrator, None, None).invoke({"run_id": "r"})
    assert pending.inputs == [None]
    for kicked_off in (finished, stateless):
        (inp,) = kicked_off.inputs
        assert inp["messages"][0].content == KICKOFF_MESSAGE


class _KilledTransport:
    """A transport killed mid-send: raises _KilledByCrash, which nothing in finalize catches (it catches SendError,
    then Exception), exactly as a real kill during the failure notice would not be caught."""
    name = "killed"

    def send(self, msg):
        raise _KilledByCrash("simulated process kill while the failure notice is being sent")


def _killed_in_finalize_deps_factory(settings, ctx, run_dir, clock, cassette=None):
    deps = _fake_deps_factory(settings, ctx, run_dir, clock, cassette)
    deps.transport = _KilledTransport()
    return deps


def _run_killed_inside_finalize(settings, tmp_path) -> str:
    """run_once whose model ends without a report, killed while finalize sends the failure notice: the outer graph's
    "agent" step is checkpointed, its "finalize" step is not, and the failure marker is left at "sending"."""
    from nasdaq_agent.agent.graph import run_once
    with pytest.raises(_KilledByCrash):
        run_once(settings, model=_ends_without_tools("Nothing notable today."),
                 deps_factory=_killed_in_finalize_deps_factory)
    return _only_run_id(tmp_path)


def test_resume_after_a_kill_inside_finalize_runs_only_finalize(tmp_path, monkeypatch):
    """Final fix wave A8: the outer graph stopped in "finalize", so resume_run continues it with invoke(None) and only
    finalize runs. The model is never called -- restarting from "init" would re-enter the agent loop, which gives the
    finished orchestrator thread a new kickoff and calls the model -- and no second notice is sent: the failure
    marker from the killed attempt is still "sending", so that notice's outcome is unknown."""
    settings = _settings(tmp_path, monkeypatch)
    from nasdaq_agent.agent.finalize import FAILURE_MARKER
    from nasdaq_agent.agent.graph import resume_run
    from nasdaq_agent.email.idempotency import SendMarker
    run_id = _run_killed_inside_finalize(settings, tmp_path)
    run_dir_path = tmp_path / "runs" / run_id
    assert SendMarker(run_dir_path / FAILURE_MARKER).state == "sending"

    resume_model = scripted([AIMessage(content="must not be reached: the resume runs only finalize")])
    outcome = resume_run(settings, run_id, model=resume_model, deps_factory=_fake_deps_factory)

    assert outcome.exit_code == 1
    assert resume_model.bound_tool_names == []  # the agent loop was not re-entered
    assert list((run_dir_path / "outbox").glob("*.eml")) == []  # no second failure notice
    assert SendMarker(run_dir_path / FAILURE_MARKER).state == "sending"
    assert json.loads((run_dir_path / "summary.json").read_text())["exit_code"] == 1


class _NextHidesPendingTasks:
    """The compiled outer graph, except that get_state reports .next as empty while .tasks keeps the pending tasks:
    the StateSnapshot langgraph returns when a pending task's writes were saved before the kill. Records each real
    snapshot's tasks so the test can confirm the scenario really had a pending task to hide."""

    def __init__(self, graph):
        self._graph = graph
        self.real_tasks: list[tuple] = []

    def get_state(self, config):
        snapshot = self._graph.get_state(config)
        self.real_tasks.append(snapshot.tasks)
        return snapshot._replace(next=())

    def invoke(self, *args, **kwargs):
        return self._graph.invoke(*args, **kwargs)


def test_resume_detects_the_outer_graphs_pending_work_by_tasks_not_next(tmp_path, monkeypatch):
    """Final fix wave A4, the outer graph in resume_run: with .next empty but a task still pending, resume_run must
    continue with invoke(None) -- here, finalize alone -- and never restart the agent loop."""
    settings = _settings(tmp_path, monkeypatch)
    from nasdaq_agent.agent import graph as graph_module
    run_id = _run_killed_inside_finalize(settings, tmp_path)
    real_build_graph = graph_module.build_graph
    stand_ins: list[_NextHidesPendingTasks] = []

    def build_stand_in(*args, **kwargs):
        stand_ins.append(_NextHidesPendingTasks(real_build_graph(*args, **kwargs)))
        return stand_ins[-1]

    monkeypatch.setattr(graph_module, "build_graph", build_stand_in)
    resume_model = scripted([AIMessage(content="must not be reached: the resume runs only finalize")])
    outcome = graph_module.resume_run(settings, run_id, model=resume_model, deps_factory=_fake_deps_factory)

    (stand_in,) = stand_ins
    assert [len(tasks) for tasks in stand_in.real_tasks] == [1]  # finalize was pending, hidden from .next
    assert outcome.exit_code == 1 and resume_model.bound_tool_names == []


def test_a_graph_invoke_failure_before_finalize_is_labelled_run_failed(tmp_path, monkeypatch):
    """Final fix wave A7: setup succeeded, so a graph.invoke failure before finalize recorded an exit code is a run
    failure, not a setup failure, in the notice and in summary.json."""
    settings = _settings(tmp_path, monkeypatch)
    from nasdaq_agent.agent import graph as graph_module

    def _raising_finalize(ctx, deps):
        raise RuntimeError("crash right at finalize")

    monkeypatch.setattr(graph_module, "finalize", _raising_finalize)
    outcome = graph_module.run_once(settings, model=_ends_without_tools("Nothing notable today."),
                                    deps_factory=_fake_deps_factory)

    assert outcome.exit_code == 1
    run_dir_path = Path(outcome.artifacts_path)
    body = _notice_body(run_dir_path)
    assert "run failed: RuntimeError: crash right at finalize" in body and "setup failed" not in body
    summary = json.loads((run_dir_path / "summary.json").read_text())
    assert summary["errors"] == ["run failed: RuntimeError: crash right at finalize"]
