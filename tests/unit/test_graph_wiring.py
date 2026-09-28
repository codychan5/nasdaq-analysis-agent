import time
from datetime import date, datetime, timezone

import httpx
import pytest
import respx


def test_build_deps_notes_sandbox_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    from nasdaq_agent.config import Settings
    from nasdaq_agent.artifacts import RunDir
    from nasdaq_agent.agent.context import RunContext
    from nasdaq_agent.agent import graph
    from nasdaq_agent.sandbox import runner as sandbox_runner
    from nasdaq_agent.universe import Universe
    from tests.unit.test_universe import SAMPLE
    # Controller correction c: build_deps must call select_runner with select_runner's own
    # default availability check (docker_backend_available), not a daemon-only check imported
    # into graph.py -- a daemon-only check would let "auto" pick Docker and then fail every run
    # with "Unable to find image". docker_backend_available looks up `docker_available` as a
    # module global in nasdaq_agent.sandbox.runner at call time, so patching it there (rather
    # than on `graph`, which no longer imports it) is what actually takes effect.
    monkeypatch.setattr(sandbox_runner, "docker_available", lambda: False)
    monkeypatch.setattr(graph, "load_universe", lambda fetch, cache: Universe.from_text(SAMPLE))
    run_dir = RunDir(tmp_path, "run-w")
    ctx = RunContext.new("run-w", "h", str(run_dir.path))
    deps = graph.build_deps(Settings(_env_file=None), ctx, run_dir, clock=lambda: datetime.now(timezone.utc), judge=lambda *a: None)
    assert deps.runner.name == "subprocess" and any("Docker" in n for n in ctx.notes)
    assert [s.name for s in deps.gainer_sources] == ["yahoo", "nasdaqcom"] and deps.transport.name == "file"


@respx.mock
def test_every_massive_source_draws_on_one_rate_limit(tmp_path, monkeypatch):
    """Massive counts calls per API key. With room for two calls per window, the third Massive call of a run waits,
    whichever of the gainer, history and news sources makes it."""
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    monkeypatch.setenv("MASSIVE_API_KEY", "test-key")
    from nasdaq_agent.config import Settings
    from nasdaq_agent.artifacts import RunDir
    from nasdaq_agent.agent.context import RunContext
    from nasdaq_agent.agent import graph
    from nasdaq_agent.sandbox import runner as sandbox_runner
    from nasdaq_agent.sources import registry
    from nasdaq_agent.sources.errors import SourceError
    from nasdaq_agent.universe import Universe
    from tests.unit.test_universe import SAMPLE
    window_seconds = 0.3
    monkeypatch.setattr(sandbox_runner, "docker_available", lambda: False)
    monkeypatch.setattr(graph, "load_universe", lambda fetch, cache: Universe.from_text(SAMPLE))
    monkeypatch.setattr(registry, "MASSIVE_CALLS_PER_MINUTE", 2)
    monkeypatch.setattr(registry, "SECONDS_PER_MINUTE", window_seconds)
    respx.get(url__startswith="https://api.massive.com/").mock(return_value=httpx.Response(200, json={"results": []}))
    run_dir = RunDir(tmp_path, "run-m")
    ctx = RunContext.new("run-m", "h", str(run_dir.path))
    deps = graph.build_deps(Settings(_env_file=None), ctx, run_dir, clock=lambda: datetime.now(timezone.utc), judge=lambda *a: None)
    gainer, history, news = (next(s for s in sources if s.name == "massive")
                             for sources in (deps.gainer_sources, deps.history_sources, deps.news_sources))
    session, moment = date(2026, 9, 25), datetime(2026, 9, 26, tzinfo=timezone.utc)
    started = time.monotonic()
    with pytest.raises(SourceError):  # one call: an empty grouped-daily reply ends the lookup
        gainer.top_candidates(session, date(2026, 9, 24), limit=5)
    history.corporate_actions("AAPL", session, session)
    news.headlines("AAPL", moment, moment)  # the third call finds the shared window full
    assert time.monotonic() - started >= window_seconds


def test_graph_compiles(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    from nasdaq_agent.agent import graph

    class FakeOrchestrator:
        def invoke(self, inp, config=None, **kwargs): return {"messages": []}

    compiled = graph.build_graph(FakeOrchestrator(), ctx=None, deps=None, checkpointer=None)
    assert set(compiled.get_graph().nodes) >= {"init", "agent", "finalize"}


def test_agent_node_survives_orchestrator_exception_and_finalize_still_sends_notice(tmp_path, monkeypatch):
    """Controller correction e: whatever the orchestrator raises must not escape the "agent"
    node, or the graph aborts before "finalize" ever runs -- and finalize is what sends the
    failure notice and writes summary.json. Exercised through build_graph with a real ctx/deps
    (built from fakes, no network/model), not a mocked finalize.
    """
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    from nasdaq_agent.agent import graph
    from nasdaq_agent.agent.context import RunContext
    from nasdaq_agent.agent.tools.common import Deps
    from nasdaq_agent.artifacts import RunDir
    from nasdaq_agent.config import Settings
    from nasdaq_agent.email.transports import FileTransport
    from nasdaq_agent.universe import Universe
    from tests.unit.test_universe import SAMPLE

    class FakeOrchestrator:
        def invoke(self, inp, config=None, **kwargs):
            raise RuntimeError("boom")

    run_dir = RunDir(tmp_path, "run-crash")
    ctx = RunContext.new("run-crash", "h", str(run_dir.path))
    deps = Deps(settings=Settings(_env_file=None), universe=Universe.from_text(SAMPLE),
                gainer_sources=[], history_sources=[], news_sources=[], runner=None,
                transport=FileTransport(run_dir.outbox_dir), judge=None,
                clock=lambda: datetime.now(timezone.utc), run_dir=run_dir)

    compiled = graph.build_graph(FakeOrchestrator(), ctx=ctx, deps=deps, checkpointer=None)
    result = compiled.invoke({"run_id": "run-crash"}, config={"configurable": {"thread_id": "graph-run-crash"}})

    assert result["exit_code"] == 1
    assert any("agent loop failed" in e and "RuntimeError" in e for e in ctx.errors)
    eml_files = list(run_dir.outbox_dir.glob("*.eml"))
    assert len(eml_files) == 1


def test_agent_node_and_finalize_redact_secrets_everywhere(tmp_path, monkeypatch):
    """Fix round 1, item 4/9: a secret that ends up inside an orchestrator exception message must
    never reach ctx.errors, the failure-notice email, summary.json or the saved context.json --
    only the *** placeholder should. Settings reads the Google key via the un-prefixed
    GOOGLE_API_KEY alias.
    """
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    monkeypatch.setenv("GOOGLE_API_KEY", "sk-test-secret-123")
    from nasdaq_agent.agent import graph
    from nasdaq_agent.agent.context import RunContext
    from nasdaq_agent.agent.tools.common import Deps
    from nasdaq_agent.artifacts import RunDir
    from nasdaq_agent.config import Settings
    from nasdaq_agent.email.transports import FileTransport
    from nasdaq_agent.universe import Universe
    from tests.unit.test_universe import SAMPLE

    secret = "sk-test-secret-123"

    class FakeOrchestrator:
        def invoke(self, inp, config=None, **kwargs):
            raise RuntimeError(f"auth failed for key {secret}")

    settings = Settings(_env_file=None)
    run_dir = RunDir(tmp_path, "run-secret")
    ctx = RunContext.new("run-secret", "h", str(run_dir.path))
    deps = Deps(settings=settings, universe=Universe.from_text(SAMPLE), gainer_sources=[], history_sources=[],
                news_sources=[], runner=None, transport=FileTransport(run_dir.outbox_dir), judge=None,
                clock=lambda: datetime.now(timezone.utc), run_dir=run_dir)

    compiled = graph.build_graph(FakeOrchestrator(), ctx=ctx, deps=deps, checkpointer=None)
    result = compiled.invoke({"run_id": "run-secret"}, config={"configurable": {"thread_id": "graph-run-secret"}})

    assert result["exit_code"] == 1
    assert ctx.errors and all(secret not in e for e in ctx.errors)
    assert any("***" in e for e in ctx.errors)

    eml = next(run_dir.outbox_dir.glob("*.eml")).read_bytes().decode()
    assert secret not in eml and "***" in eml

    assert secret not in (run_dir.path / "summary.json").read_text()
    assert secret not in (run_dir.path / "context.json").read_text()
