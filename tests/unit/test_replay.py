# tests/unit/test_replay.py
"""Task 22: record and replay. The first four tests are the task plan's, verbatim. The rest cover the controller's
corrections (a, b, c, e, f, j, k) and what the spec's "replay runs the bundled cassette with no keys" needs. All
offline: no keys, no network."""
import json
import re
import sys
import sysconfig
from datetime import date, datetime, timezone
from pathlib import Path

import pytest
from pydantic import SecretStr

from tests import fakes
from tests.unit.test_tools_data import deps, ctx, universe  # noqa: F401  (fixtures)
from tests.unit.test_tools_agent import GOOD, FakeRunner, _good_args, _verified, failure, ready, success  # noqa: F401

NOW_ISO = "2026-09-24T22:00:00+00:00"
PYTHON_LIB_IN_SANDBOX_IMAGE = f"/usr/local/lib/python{sys.version_info.major}.{sys.version_info.minor}"
CREDENTIAL_VARIABLES = ("MASSIVE_API_KEY", "ALPHAVANTAGE_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY",
                        "OPENROUTER_API_KEY", "LANGSMITH_API_KEY", "LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2")


@pytest.fixture(autouse=True)
def _clean_environment(monkeypatch):
    """Fix round 1, minor 2: a developer's exported keys or AGENT_* settings must not change what these tests see
    (a MASSIVE_API_KEY would add Massive to every source chain). Tests set what they need themselves."""
    import os
    for name in [*CREDENTIAL_VARIABLES, *(n for n in os.environ if n.startswith("AGENT_"))]:
        monkeypatch.delenv(name, raising=False)


def test_http_cassette_round_trip(tmp_path):
    from nasdaq_agent.replay.http_cassette import HttpCassette
    rec = HttpCassette(tmp_path, "record"); rec.store("k1", 200, '{"a":1}')
    rep = HttpCassette(tmp_path, "replay")
    assert rep.lookup("k1") == (200, '{"a":1}') and rep.lookup("nope") is None

def test_llm_cache_record_then_replay(tmp_path):
    from langchain_core.outputs import ChatGeneration
    from langchain_core.messages import AIMessage
    from nasdaq_agent.replay.llm_cache import JsonFileLLMCache
    from nasdaq_agent.sources.errors import CassetteMiss
    rec = JsonFileLLMCache(tmp_path, "record")
    gen = [ChatGeneration(message=AIMessage(content="hello", tool_calls=[{"name": "give_up", "args": {"reason": "x"}, "id": "c1", "type": "tool_call"}]))]
    assert rec.lookup("prompt", "llm") is None
    rec.update("prompt", "llm", gen)
    rep = JsonFileLLMCache(tmp_path, "replay")
    got = rep.lookup("prompt", "llm")
    assert got[0].message.content == "hello" and got[0].message.tool_calls[0]["name"] == "give_up"
    with pytest.raises(CassetteMiss):
        rep.lookup("other prompt", "llm")

def test_manifest_guards_prompt_changes(tmp_path, monkeypatch):
    from nasdaq_agent.replay import llm_cache as lc
    from nasdaq_agent.sources.errors import CassetteMiss
    lc.write_manifest(tmp_path, "2026-09-24T22:00:00+00:00")
    assert lc.check_manifest(tmp_path)["now"] == "2026-09-24T22:00:00+00:00"
    monkeypatch.setattr(lc, "prompt_fingerprint", lambda: "changed")
    with pytest.raises(CassetteMiss):
        lc.check_manifest(tmp_path)

def test_recorded_source_round_trip(tmp_path):
    from nasdaq_agent.replay.proxies import RecordedSource
    from tests import fakes
    inner = fakes.FakeHistorySource(data={"ACME": fakes.series("ACME", fakes.CLOSES)})
    rec = RecordedSource(inner, tmp_path, "record")
    s1 = rec.bars("ACME", date(2026, 9, 17), date(2026, 9, 24))
    dead = fakes.FakeHistorySource(error="offline")
    rep = RecordedSource(dead, tmp_path, "replay")
    s2 = rep.bars("ACME", date(2026, 9, 17), date(2026, 9, 24))
    assert s2.closes() == s1.closes() and rep.name == inner.name
    rec.corporate_actions("ACME", date(2026, 9, 17), date(2026, 9, 24))
    assert rep.corporate_actions("ACME", date(2026, 9, 17), date(2026, 9, 24)) == []


# --- Cassette stores: boundaries and malformed content --------------------------------------------

def test_http_cassette_rejects_entry_names_that_could_leave_its_directory(tmp_path):
    from nasdaq_agent.replay.http_cassette import HttpCassette
    cassette = HttpCassette(tmp_path, "record")
    for unsafe in ("../escape", "a/b", "", ".hidden", "x.json"):
        with pytest.raises(ValueError):
            cassette.store(unsafe, 200, "text")
        with pytest.raises(ValueError):
            cassette.lookup(unsafe)
    assert not (tmp_path / "escape.json").exists()


def test_http_cassette_malformed_entry_is_a_cassette_miss(tmp_path):
    from nasdaq_agent.replay.http_cassette import HttpCassette
    from nasdaq_agent.sources.errors import CassetteMiss
    HttpCassette(tmp_path, "record").store("k1", 200, "ok")
    (tmp_path / "http" / "k1.json").write_text('{"status": "200"}')
    with pytest.raises(CassetteMiss):
        HttpCassette(tmp_path, "replay").lookup("k1")


def test_cassette_stores_reject_an_unknown_mode(tmp_path):
    from nasdaq_agent.replay.http_cassette import HttpCassette
    from nasdaq_agent.replay.llm_cache import JsonFileLLMCache
    from nasdaq_agent.replay.proxies import RecordedSource
    for build in (lambda: HttpCassette(tmp_path, "live"), lambda: JsonFileLLMCache(tmp_path, "off"),
                  lambda: RecordedSource(fakes.FakeHistorySource(), tmp_path, "live")):
        with pytest.raises(ValueError):
            build()


def test_llm_cache_stores_plain_message_data(tmp_path):
    """Correction a: messages_to_dict for the message plus the generation's text and generation_info -- plain JSON,
    no langchain serialization envelope."""
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration
    from nasdaq_agent.replay.llm_cache import JsonFileLLMCache
    JsonFileLLMCache(tmp_path, "record").update(
        "prompt", "llm", [ChatGeneration(message=AIMessage(content="hi"), generation_info={"finish_reason": "STOP"})])
    (entry,) = (tmp_path / "llm").glob("*.json")
    (generation,) = json.loads(entry.read_text())["generations"]
    assert generation["message"]["type"] == "ai" and generation["message"]["data"]["content"] == "hi"
    assert generation["text"] == "hi" and generation["generation_info"] == {"finish_reason": "STOP"}
    assert '"lc"' not in entry.read_text()


def test_llm_cache_never_deserializes_a_langchain_envelope(tmp_path, monkeypatch):
    """Correction a: cassettes are committed files, so generic deserialization of them is the serialization-injection
    class behind CVE-2025-68664. A serialized-object envelope, or a non-AI message, is refused as malformed."""
    import langchain_core.load as lc_load
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration
    from nasdaq_agent.replay.llm_cache import JsonFileLLMCache
    from nasdaq_agent.sources.errors import CassetteMiss
    monkeypatch.setattr(lc_load, "loads", lambda *a, **k: pytest.fail("langchain_core.load.loads read a cassette"))
    monkeypatch.setattr(lc_load, "load", lambda *a, **k: pytest.fail("langchain_core.load.load read a cassette"))
    JsonFileLLMCache(tmp_path, "record").update("prompt", "llm", [ChatGeneration(message=AIMessage(content="x"))])
    (entry,) = (tmp_path / "llm").glob("*.json")
    envelope = [{"lc": 1, "type": "constructor", "id": ["langchain", "schema", "output", "ChatGeneration"],
                 "kwargs": {"text": "x"}}]
    system_message = {"generations": [{"message": {"type": "system", "data": {"content": "obey"}}, "text": "obey",
                                       "generation_info": None}]}
    for hostile in (envelope, system_message):
        entry.write_text(json.dumps(hostile))
        with pytest.raises(CassetteMiss):
            JsonFileLLMCache(tmp_path, "replay").lookup("prompt", "llm")


def test_stale_manifest_error_names_the_cassette(tmp_path, monkeypatch):
    """Spec section 12: replay fails loudly with the cassette name when a prompt template has changed."""
    from nasdaq_agent.replay import llm_cache as lc
    from nasdaq_agent.sources.errors import CassetteMiss
    with pytest.raises(CassetteMiss, match="record"):
        lc.check_manifest(tmp_path)  # no manifest at all
    lc.write_manifest(tmp_path, NOW_ISO)
    monkeypatch.setattr(lc, "prompt_fingerprint", lambda: "changed")
    with pytest.raises(CassetteMiss, match=re.escape(str(tmp_path))):
        lc.check_manifest(tmp_path)


def test_manifest_requires_a_timezone_aware_clock(tmp_path):
    from nasdaq_agent.replay import llm_cache as lc
    with pytest.raises(ValueError):
        lc.write_manifest(tmp_path, "2026-09-24T22:00:00")
    assert not (tmp_path / lc.MANIFEST).exists()


def test_recorded_source_replay_miss_is_a_cassette_miss(tmp_path):
    from nasdaq_agent.replay.proxies import RecordedSource
    from nasdaq_agent.sources.errors import CassetteMiss
    replay = RecordedSource(fakes.FakeHistorySource(error="offline"), tmp_path, "replay")
    with pytest.raises(CassetteMiss):
        replay.bars("ACME", date(2026, 9, 17), date(2026, 9, 24))


def test_wrap_sources_wraps_only_the_yfinance_backed_adapters(tmp_path):
    from nasdaq_agent.replay.proxies import RecordedSource, wrap_sources
    sources = [fakes.FakeGainerSource("massive"), fakes.FakeGainerSource("yahoo", requires_market_closed=True),
               fakes.FakeGainerSource("nasdaqcom")]
    wrapped = wrap_sources(sources, tmp_path, "replay")
    assert [isinstance(s, RecordedSource) for s in wrapped] == [False, True, False]
    assert wrapped[1].name == "yahoo" and wrapped[1].requires_market_closed is True


# --- Correction b: model-visible tool results carry no run-specific values ------------------------

def test_send_email_results_carry_no_run_specific_values(ready, deps):
    from nasdaq_agent.agent.tools.compose import make_compose_report
    from nasdaq_agent.agent.tools.send import make_send_email
    _verified(ready, deps); make_compose_report(ready, deps).invoke(_good_args())
    tool = make_send_email(ready, deps)
    first, again = tool.invoke({}), tool.invoke({})
    for out in (first, again):
        assert not {"message_id", "location", "transport"} & set(json.loads(out))
        assert str(deps.run_dir.path) not in out and ".eml" not in out
    assert json.loads(first) == {"sent": True} and json.loads(again) == {"sent": True, "already_sent": True}
    # The details stay in the run's own record: ctx.email and sent.json.
    sent = json.loads((deps.run_dir.path / "sent.json").read_text())
    assert ready.email.message_id and sent["message_id"] == ready.email.message_id
    assert ready.email.location and sent["location"] == ready.email.location


def test_send_email_transport_failure_result_omits_the_outbox_path(ready, deps):
    from nasdaq_agent.agent.tools.compose import make_compose_report
    from nasdaq_agent.agent.tools.send import make_send_email
    from nasdaq_agent.email.transports import SendError

    class RefusingTransport:
        name = "smtp"

        def send(self, msg):
            raise SendError("smtp send failed: SMTPAuthenticationError")

    _verified(ready, deps); make_compose_report(ready, deps).invoke(_good_args())
    deps.transport = RefusingTransport()
    out = make_send_email(ready, deps).invoke({})
    assert out.startswith("ERROR") and "written to the outbox" in out
    assert str(deps.run_dir.path) not in out and ".eml" not in out
    assert ready.email.location.endswith(".eml")  # the path stays in ctx.email


def test_run_python_results_omit_the_sandbox_backend(ready, deps):
    from nasdaq_agent.agent.tools.code import make_run_python
    deps.runner = FakeRunner([failure(), success()])
    tool = make_run_python(ready, deps)
    failed, passed = tool.invoke({"code": "x"}), tool.invoke({"code": "result = {}"})
    assert "backend" not in failed and "fake" not in failed
    assert "backend" not in json.loads(passed)
    assert [a.backend for a in ready.analysis.attempts] == ["fake", "fake"]  # kept in attempts for the report's footer


# --- Correction f: error tails show the Docker layout, never the host's ---------------------------

def test_run_python_error_text_uses_the_docker_layout(ready, deps):
    from nasdaq_agent.agent.tools.code import make_run_python
    from nasdaq_agent.sandbox.contract import RunResult
    site_packages = sysconfig.get_paths()["purelib"]
    seen = {}

    class HostTracebackRunner:
        name = "subprocess"

        def run(self, code, sandbox_dir):
            seen["dir"] = str(sandbox_dir)
            trace = (f'Traceback (most recent call last):\n  File "{sandbox_dir}/bootstrap.py", line 19, in <module>\n'
                     f'  File "{site_packages}/pandas/core/frame.py", line 4113, in __getitem__\n'
                     "KeyError: 'adj_close_typo'")
            return RunResult(exit_code=1, backend=self.name, wall_time_s=0.1, stderr_tail=trace,
                             error=f"sandbox setup failed: OSError: {sandbox_dir}/analysis_code.py")

    deps.runner = HostTracebackRunner()
    out = make_run_python(ready, deps).invoke({"code": "x"})
    assert '"/work/bootstrap.py"' in out and "/work/analysis_code.py" in out
    assert f"{PYTHON_LIB_IN_SANDBOX_IMAGE}/site-packages/pandas/core/frame.py" in out
    assert seen["dir"] not in out and site_packages not in out
    # The local artefact keeps the real traceback for whoever debugs the run.
    assert seen["dir"] in (deps.run_dir.attempts_dir / "attempt_1.err").read_text()


def test_run_python_works_with_a_relative_artifacts_directory(ctx, deps, tmp_path, monkeypatch):
    """The default AGENT_ARTIFACTS_DIR is ./runs. The subprocess backend runs its child inside the sandbox directory,
    so it must be handed an absolute path or the child resolves the bootstrap path against itself."""
    from nasdaq_agent.agent.tools.code import make_run_python
    from nasdaq_agent.artifacts import RunDir
    from nasdaq_agent.sandbox.runner import SubprocessRunner, prepare_sandbox_dir
    from nasdaq_agent.agent.tools.history import _to_csv
    monkeypatch.chdir(tmp_path)
    deps.run_dir = RunDir(Path("runs"), "run-relative")
    ticker = fakes.series("ACME", fakes.CLOSES)
    prepare_sandbox_dir(deps.run_dir.path, _to_csv(ticker), _to_csv(fakes.series("SPY", fakes.BENCH)),
                        {"required_keys": list(GOOD)})
    ctx.progress.history_ready = True
    deps.runner = SubprocessRunner()
    out = json.loads(make_run_python(ctx, deps).invoke({"code": f"result = {json.dumps(GOOD)}"}))
    assert out["exit_code"] == 0 and out["result"]["trend"] == "uptrend"


# --- Correction k: the symbol file comes through the cassette in record and replay -----------------

def _replay_settings_for(tmp_path, **update):
    from nasdaq_agent.config import Mode, Settings
    return Settings(_env_file=None).model_copy(update={"mode": Mode.replay, "cassette_dir": tmp_path / "cassette",
                                                       "artifacts_dir": tmp_path / "artifacts", **update})


def test_build_deps_in_replay_loads_the_universe_from_the_cassette_without_network(tmp_path, monkeypatch):
    import httpx
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    from nasdaq_agent.agent import graph
    from nasdaq_agent.agent.context import RunContext
    from nasdaq_agent.artifacts import RunDir
    from nasdaq_agent.replay.http_cassette import HttpCassette
    from nasdaq_agent.replay.proxies import RecordedSource
    from nasdaq_agent.sandbox import runner as sandbox_runner
    from nasdaq_agent.sources.http import request_key
    from nasdaq_agent.universe import SYMBOL_FILE_URL
    from tests.unit.test_universe import SAMPLE

    def no_network(*args, **kwargs):
        raise AssertionError("replay must not touch the network")

    monkeypatch.setattr(httpx.Client, "send", no_network)
    monkeypatch.setattr(sandbox_runner, "docker_available", lambda: False)
    settings = _replay_settings_for(tmp_path)
    HttpCassette(settings.cassette_dir, "record").store(request_key("GET", SYMBOL_FILE_URL, None), 200, SAMPLE)
    settings.artifacts_dir.mkdir()  # empty: no local universe cache on a clean machine
    run_dir = RunDir(tmp_path / "runs", "run-k")
    built = graph.build_deps(settings, RunContext.new("run-k", "h", str(run_dir.path)), run_dir,
                             clock=lambda: datetime(2026, 9, 24, 22, 0, tzinfo=timezone.utc), judge=lambda *a, **k: None)
    assert built.universe.is_common_stock("AAPL") and built.universe.record("TSLA") is not None
    assert list(settings.artifacts_dir.iterdir()) == []  # and it writes no local cache either
    assert [s.name for s in built.gainer_sources] == ["yahoo", "nasdaqcom"]
    assert isinstance(built.gainer_sources[0], RecordedSource) and isinstance(built.history_sources[0], RecordedSource)


def test_replay_ignores_a_fresh_local_universe_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    from nasdaq_agent.agent import graph
    from nasdaq_agent.agent.context import RunContext
    from nasdaq_agent.artifacts import RunDir
    from nasdaq_agent.replay.http_cassette import HttpCassette
    from nasdaq_agent.sandbox import runner as sandbox_runner
    from nasdaq_agent.sources.http import request_key
    from nasdaq_agent.universe import SYMBOL_FILE_URL
    from tests.unit.test_universe import SAMPLE
    monkeypatch.setattr(sandbox_runner, "docker_available", lambda: False)
    settings = _replay_settings_for(tmp_path)
    HttpCassette(settings.cassette_dir, "record").store(request_key("GET", SYMBOL_FILE_URL, None), 200, SAMPLE)
    settings.artifacts_dir.mkdir()
    (settings.artifacts_dir / graph.UNIVERSE_CACHE).write_text(SAMPLE.replace("TSLA|Tesla", "ZZZZ|Other"))
    run_dir = RunDir(tmp_path / "runs", "run-k2")
    built = graph.build_deps(settings, RunContext.new("run-k2", "h", str(run_dir.path)), run_dir,
                             clock=lambda: datetime(2026, 9, 24, 22, 0, tzinfo=timezone.utc), judge=lambda *a, **k: None)
    assert built.universe.record("TSLA") is not None and built.universe.record("ZZZZ") is None


# --- Correction j: replay never sends real mail ----------------------------------------------------

def test_replay_always_uses_the_file_outbox_whatever_smtp_settings_hold(tmp_path, monkeypatch):
    for name, value in {"AGENT_EMAIL_TO": "r@example.com", "AGENT_SMTP_HOST": "smtp.example.com",
                        "AGENT_SMTP_USERNAME": "u", "AGENT_SMTP_PASSWORD": "pw", "AGENT_SMTP_FROM": "b@example.com"}.items():
        monkeypatch.setenv(name, value)
    from nasdaq_agent.agent import graph
    from nasdaq_agent.artifacts import RunDir
    from nasdaq_agent.config import Mode
    run_dir = RunDir(tmp_path / "runs", "run-j")
    settings = _replay_settings_for(tmp_path)
    assert graph._transport_for_mode(settings, run_dir.outbox_dir).name == "file"
    assert graph._transport_for_finalize(settings, run_dir).name == "file"
    record = settings.model_copy(update={"mode": Mode.record})
    assert graph._transport_for_mode(record, run_dir.outbox_dir).name == "smtp"  # record is a live run


# --- Zero keys (spec: "replay runs the bundled cassette with no keys") -----------------------------

def test_replay_builds_the_default_model_with_no_key_and_the_recorded_llm_string(tmp_path, monkeypatch):
    """The provider class is still needed in replay -- bind_tools, with_structured_output and the cache key all come
    from it -- and ChatGoogleGenerativeAI cannot be constructed without a credential. Replay gives it a placeholder,
    which is never sent anywhere (every response comes from the cassette, a miss raises first); the llm_string, part
    of every cache key, names the key by reference only, so it matches the keyed build that recorded it."""
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    from nasdaq_agent.agent import graph
    from nasdaq_agent.agent.llm import build_chat_model
    replay = graph._replay_settings(_replay_settings_for(tmp_path), {"keyed_sources": []})
    recorded_with = replay.model_copy(update={"google_api_key": SecretStr("a-real-key-at-record-time")})
    assert build_chat_model(replay, "orchestrator")._get_llm_string() == \
        build_chat_model(recorded_with, "orchestrator")._get_llm_string()
    assert replay.llm_call_delay_seconds == 0  # correction c


def test_build_chat_model_passes_the_provider_key_from_settings(monkeypatch):
    """A key set only in .env reaches Settings but not os.environ (pydantic-settings does not export it), so the model
    must be handed the key explicitly; replay's placeholder travels the same way."""
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    monkeypatch.setenv("AGENT_LLM_MODEL", "google_genai:gemini-3.5-flash-lite")
    monkeypatch.setenv("AGENT_LLM_FALLBACK_MODEL", "anthropic:claude-haiku-5")
    monkeypatch.setenv("GOOGLE_API_KEY", "g-123")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    captured = {}

    def fake_init(model, **kw):
        captured[model] = kw

        class Model:
            def with_fallbacks(self, others):
                return self
        return Model()

    from nasdaq_agent.agent import llm
    from nasdaq_agent.config import Settings
    monkeypatch.setattr(llm, "init_chat_model", fake_init)
    llm.build_chat_model(Settings(_env_file=None), "orchestrator")
    assert captured["google_genai:gemini-3.5-flash-lite"]["api_key"].get_secret_value() == "g-123"
    assert "api_key" not in captured["anthropic:claude-haiku-5"]  # unset: left to the provider's own lookup


def test_replay_rebuilds_exactly_the_recorded_source_chain(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    monkeypatch.setenv("ALPHAVANTAGE_API_KEY", "av-key-on-this-machine")  # a key the recording did not use
    monkeypatch.setenv("AGENT_SEC_USER_AGENT", "Machine Owner owner@example.com")  # nor this SEC contact
    from nasdaq_agent.agent import graph
    from nasdaq_agent.sources.registry import (build_gainer_sources, build_history_sources, build_news_sources,
                                               massive_rate_limiter)
    from nasdaq_agent.universe import Universe
    from tests.unit.test_universe import SAMPLE
    replay = graph._replay_settings(_replay_settings_for(tmp_path), {"keyed_sources": ["massive"]})
    sample_universe = Universe.from_text(SAMPLE)
    limiter = massive_rate_limiter()
    assert [s.name for s in build_gainer_sources(replay, sample_universe, tmp_path, massive_limiter=limiter)] == [
        "massive", "yahoo", "nasdaqcom"]
    assert [s.name for s in build_history_sources(replay, sample_universe, massive_limiter=limiter)] == [
        "yfinance", "massive"]
    assert [s.name for s in build_news_sources(replay, massive_limiter=limiter)] == ["massive", "yfinance"]
    assert "av-key-on-this-machine" not in replay.secret_values()  # replay never holds a real key
    assert "Machine Owner owner@example.com" not in replay.secret_values()


def test_replay_rebuilds_a_recorded_sec_source_without_the_recording_contact(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    monkeypatch.delenv("AGENT_SEC_USER_AGENT", raising=False)
    from nasdaq_agent.agent import graph
    from nasdaq_agent.sources.registry import build_news_sources, massive_rate_limiter
    replay = graph._replay_settings(_replay_settings_for(tmp_path), {"keyed_sources": ["sec"]})
    assert [s.name for s in build_news_sources(replay, massive_limiter=massive_rate_limiter())] == ["yfinance", "sec"]
    assert replay.sec_user_agent.get_secret_value() == graph.REPLAY_PLACEHOLDER_KEY


def test_replay_refuses_a_manifest_naming_an_unknown_source(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    from nasdaq_agent.agent import graph
    from nasdaq_agent.sources.errors import CassetteMiss
    with pytest.raises(CassetteMiss):
        graph._replay_settings(_replay_settings_for(tmp_path), {"keyed_sources": ["mystery"]})


def test_request_hash_ignores_credential_parameters():
    """Cassettes recorded with keys must replay with none (or with other keys), so request hashes leave credentials out."""
    from nasdaq_agent.sources.http import request_key
    url = "https://api.massive.com/v2/aggs/grouped/locale/us/market/stocks/2026-09-24"
    assert request_key("GET", url, {"adjusted": "false", "apiKey": "real"}) == \
        request_key("GET", url, {"adjusted": "false", "apiKey": "placeholder"})
    assert request_key("GET", url, {"function": "TOP_GAINERS_LOSERS", "apikey": "a"}) == \
        request_key("GET", url, {"function": "TOP_GAINERS_LOSERS", "apikey": "b"})
    assert request_key("GET", url, {"adjusted": "false"}) != request_key("GET", url, {"adjusted": "true"})


# --- Correction e: cassette hygiene ------------------------------------------------------------------

def _previous_cassette(root: Path) -> dict[str, str]:
    """A small valid-looking cassette at root, returned as {relative path: content} for later comparison."""
    from nasdaq_agent.replay import llm_cache as lc
    from nasdaq_agent.replay.http_cassette import HttpCassette
    lc.write_manifest(root, "2026-01-02T22:00:00+00:00", exit_code=0)
    HttpCassette(root, "record").store("k-old", 200, "previous recording")
    return _tree(root)


def _tree(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): p.read_text() for p in sorted(root.rglob("*")) if p.is_file()}


def test_start_recording_stages_next_to_the_cassette_and_leaves_it_untouched(tmp_path):
    """K7: a recording goes into a fresh staging directory beside the cassette; the cassette itself is not touched."""
    from nasdaq_agent.replay.recording import start_recording
    final = tmp_path / "cassette"
    before = _previous_cassette(final)
    staging = start_recording(final, "20260924T220000Z-abc123")
    assert staging.is_dir() and staging.parent == final.resolve().parent and staging.name.startswith(".cassette.")
    assert list(staging.iterdir()) == [] and _tree(final) == before


def test_recording_guard_refuses_a_directory_that_is_not_a_cassette(tmp_path):
    """K7 guard: nothing outside a cassette directory may ever be moved or deleted."""
    from nasdaq_agent.replay.recording import CassetteDirectoryRefused, start_recording
    run_id = "20260924T220000Z-abc123"
    unrelated = tmp_path / "project"
    unrelated.mkdir()
    (unrelated / "README.md").write_text("not a cassette")
    a_file = tmp_path / "a-file"
    a_file.write_text("x")
    for refused in (unrelated, a_file):
        with pytest.raises(CassetteDirectoryRefused):
            start_recording(refused, run_id)
    assert (unrelated / "README.md").read_text() == "not a cassette" and a_file.read_text() == "x"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a-file", "project"]  # no staging directory was made
    empty = tmp_path / "empty"
    empty.mkdir()
    only_http = tmp_path / "only-http"
    (only_http / "http").mkdir(parents=True)
    for allowed in (tmp_path / "missing", empty, only_http):
        assert start_recording(allowed, run_id).is_dir()


def test_seal_promotes_the_staged_recording(tmp_path):
    from nasdaq_agent.replay import llm_cache as lc
    from nasdaq_agent.replay.http_cassette import HttpCassette
    from nasdaq_agent.replay.recording import seal_recording, start_recording
    final = tmp_path / "cassette"
    _previous_cassette(final)
    staging = start_recording(final, "20260924T220000Z-abc123")
    HttpCassette(staging, "record").store("k-new", 200, "new recording")
    seal_recording(staging, final, NOW_ISO, ["sk-live-123"], ["massive"], {"python": "3.12.0", "packages": {}}, 2)
    assert HttpCassette(final, "replay").lookup("k-new") == (200, "new recording")
    assert HttpCassette(final, "replay").lookup("k-old") is None
    manifest = lc.check_manifest(final)
    assert manifest["now"] == NOW_ISO and manifest["keyed_sources"] == ["massive"] and manifest["exit_code"] == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ["cassette"]  # no staging or backup left behind


def test_seal_refuses_a_cassette_holding_a_configured_secret(tmp_path):
    """Correction e with K7: the leak is named by path only, the manifest is never written, and the previous cassette
    is left exactly as it was."""
    from nasdaq_agent.replay.http_cassette import HttpCassette
    from nasdaq_agent.replay.recording import CassetteSecretLeak, seal_recording, start_recording
    final = tmp_path / "cassette"
    before = _previous_cassette(final)
    staging = start_recording(final, "20260924T220000Z-abc123")
    cassette = HttpCassette(staging, "record")
    cassette.store("k1", 200, '{"next_url": "https://api.massive.com/v2?cursor=1&apiKey=sk-live-123"}')
    cassette.store("k2", 200, '{"ok": true}')
    with pytest.raises(CassetteSecretLeak) as info:
        seal_recording(staging, final, NOW_ISO, ["sk-live-123", ""], [])
    assert "http/k1.json" in str(info.value) and "k2" not in str(info.value)
    assert "sk-live-123" not in str(info.value)
    assert _tree(final) == before


def test_secret_scan_sees_escaped_and_encoded_forms(tmp_path):
    """Correction e, widened in fix round 1 (minor 5): lowercase percent-encoding, "+" for spaces, and \\/-escaped
    slashes as well as raw, JSON-escaped and upper-case percent-encoded forms."""
    from nasdaq_agent.replay.http_cassette import HttpCassette
    from nasdaq_agent.replay.recording import find_secret_leaks
    password = 'pa"ss\\w/ord&x y'
    cassette = HttpCassette(tmp_path, "record")
    cassette.store("json", 200, json.dumps({"echo": password}))  # JSON inside our JSON: escaped twice
    cassette.store("url", 200, "https://example.com/?p=pa%22ss%5Cw%2Ford%26x%20y")
    cassette.store("url_lower", 200, "https://example.com/?p=pa%22ss%5cw%2ford%26x%20y")
    cassette.store("form", 200, "p=pa%22ss%5Cw%2Ford%26x+y")
    cassette.store("slashes", 200, 'pa"ss\\w\\/ord&x y')
    cassette.store("path_quote", 200, "https://example.com/pa%22ss%5Cw/ord%26x%20y")  # quote(): "/" left raw
    cassette.store("clean", 200, "nothing here")
    assert find_secret_leaks(tmp_path, [password]) == ["http/form.json", "http/json.json", "http/path_quote.json",
                                                        "http/slashes.json", "http/url.json", "http/url_lower.json"]


def test_llm_cache_key_ignores_what_a_cache_hit_changes(tmp_path):
    """Found by the record-then-replay round trip (tests/graph/test_record_replay.py): on a cache hit langchain-core
    returns the message with "total_cost": 0 added to usage_metadata, so from the second turn on the replayed prompt
    differs from the recorded one in that field alone. Message ids differ per call too (langchain-core 1.6.5 already
    nulls them; the key drops them as well). What the model said and was told must still decide the key."""
    from langchain_core.load import dumps
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langchain_core.outputs import ChatGeneration
    from nasdaq_agent.replay.llm_cache import JsonFileLLMCache
    from nasdaq_agent.sources.errors import CassetteMiss
    call = {"name": "resolve_session", "args": {}, "id": "c1", "type": "tool_call"}
    usage = {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}

    def conversation(message_id, usage_metadata, tool_output="{}"):
        return dumps([HumanMessage("Begin."), AIMessage(content="", id=message_id, tool_calls=[call], usage_metadata=usage_metadata),
                      ToolMessage(content=tool_output, tool_call_id="c1")])

    JsonFileLLMCache(tmp_path, "record").update(conversation("lc_run--aaa-0", usage), "llm",
                                                [ChatGeneration(message=AIMessage(content="next"))])
    replay = JsonFileLLMCache(tmp_path, "replay")
    assert replay.lookup(conversation("lc_run--bbb-0", {**usage, "total_cost": 0}), "llm")[0].message.content == "next"
    assert replay.lookup(conversation("lc_run--ccc-0", None), "llm")[0].message.content == "next"
    with pytest.raises(CassetteMiss):  # a different tool result is a different conversation
        replay.lookup(conversation("lc_run--aaa-0", usage, tool_output='{"other": 1}'), "llm")


def test_keyed_source_recorded_with_a_key_replays_with_none_through_the_real_adapters(tmp_path, monkeypatch):
    """The zero-key pieces together, through build_deps and the real Massive adapter: a request recorded with this
    machine's key (and the symbol file, correction k) replays from the cassette with no key held and no network."""
    import httpx
    import respx
    from nasdaq_agent.agent import graph
    from nasdaq_agent.agent.context import RunContext
    from nasdaq_agent.artifacts import RunDir
    from nasdaq_agent.config import Mode
    from nasdaq_agent.replay import llm_cache as lc
    from nasdaq_agent.replay.recording import seal_recording
    from nasdaq_agent.sandbox import runner as sandbox_runner
    from nasdaq_agent.sources.massive import MASSIVE_BASE_URL
    from nasdaq_agent.universe import SYMBOL_FILE_URL
    from tests.unit.test_universe import SAMPLE
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    monkeypatch.setattr(sandbox_runner, "docker_available", lambda: False)
    session, prev = date(2026, 9, 24), date(2026, 9, 23)

    def clock():
        return datetime(2026, 9, 24, 22, 0, tzinfo=timezone.utc)

    def grouped(closes):
        return {"results": [{"T": symbol, "c": close, "v": 1000} for symbol, close in closes.items()]}

    def build(settings, run_id):
        run_dir = RunDir(tmp_path / "runs", run_id)
        return graph.build_deps(settings, RunContext.new(run_id, "h", str(run_dir.path)), run_dir, clock=clock,
                                judge=lambda *a, **k: None)

    staging = tmp_path / ".cassette.staging"
    record = _replay_settings_for(tmp_path).model_copy(
        update={"mode": Mode.record, "cassette_dir": staging, "massive_api_key": SecretStr("key-at-record-time")})
    grouped_url = f"{MASSIVE_BASE_URL}/v2/aggs/grouped/locale/us/market/stocks"
    with respx.mock(assert_all_called=True) as live:
        live.get(SYMBOL_FILE_URL).mock(return_value=httpx.Response(200, text=SAMPLE))
        live.get(f"{grouped_url}/{session}").mock(return_value=httpx.Response(200, json=grouped({"AAPL": 110.0, "TSLA": 200.0})))
        live.get(f"{grouped_url}/{prev}").mock(return_value=httpx.Response(200, json=grouped({"AAPL": 100.0, "TSLA": 199.0})))
        recorded = build(record, "rec").gainer_sources[0].top_candidates(session, prev, 5)
    final = tmp_path / "cassette"
    seal_recording(staging, final, clock().isoformat(), record.secret_values(), graph._keyed_sources(record))
    assert [c.symbol for c in recorded] == ["AAPL", "TSLA"]

    for name in ("MASSIVE_API_KEY", "ALPHAVANTAGE_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    replay = graph._replay_settings(_replay_settings_for(tmp_path), lc.check_manifest(final))
    with respx.mock(assert_all_called=False) as offline:
        offline.route().mock(side_effect=AssertionError("replay must not touch the network"))
        source = build(replay, "rep").gainer_sources[0]
        assert source.name == "massive" and source.top_candidates(session, prev, 5) == recorded



def test_sec_filings_recorded_with_a_contact_replay_offline_without_it(tmp_path, monkeypatch):
    """The SEC source through build_deps and its real adapter: recorded with a contact in the User-Agent, sealed (the
    secret scan passes: the contact travels in a header, which cassettes never store) and replayed with no network."""
    import httpx
    import respx
    from nasdaq_agent.agent import graph
    from nasdaq_agent.agent.context import RunContext
    from nasdaq_agent.artifacts import RunDir
    from nasdaq_agent.config import Mode
    from nasdaq_agent.replay import llm_cache as lc
    from nasdaq_agent.replay.recording import seal_recording
    from nasdaq_agent.sandbox import runner as sandbox_runner
    from nasdaq_agent.universe import SYMBOL_FILE_URL
    from tests.unit import test_sec_news_source as sec
    from tests.unit.test_universe import SAMPLE
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    monkeypatch.delenv("MASSIVE_API_KEY", raising=False)
    monkeypatch.setattr(sandbox_runner, "docker_available", lambda: False)
    contact = "Jane Doe jane@example.com"

    def clock():
        return sec.UNTIL

    def build(settings, run_id):
        run_dir = RunDir(tmp_path / "runs", run_id)
        return graph.build_deps(settings, RunContext.new(run_id, "h", str(run_dir.path)), run_dir, clock=clock,
                                judge=lambda *a, **k: None)

    staging = tmp_path / ".cassette.staging"
    record = _replay_settings_for(tmp_path).model_copy(
        update={"mode": Mode.record, "cassette_dir": staging, "sec_user_agent": SecretStr(contact)})
    with respx.mock(assert_all_called=True) as live:
        live.get(SYMBOL_FILE_URL).mock(return_value=httpx.Response(200, text=SAMPLE))
        live.get(sec.SEARCH_URL, params={"keysTyped": "WETO"}).mock(return_value=httpx.Response(
            200, json=sec._search_payload([sec._entity("1941158", "Wetour Robotics Ltd", "WETO")])))
        live.get(sec.WETO_SUBMISSIONS_URL).mock(return_value=httpx.Response(200, json=sec._submissions([sec.PLACEMENT])))
        live.get(sec.SEARCH_URL, params={"ciks": "0001941158"}).mock(
            return_value=httpx.Response(200, json=sec.PLACEMENT_DOCUMENTS))
        recorded = build(record, "rec").news_sources[-1].headlines("WETO", sec.SINCE, sec.UNTIL)
    final = tmp_path / "cassette"
    seal_recording(staging, final, clock().isoformat(), record.secret_values(), graph._keyed_sources(record))
    assert [h.title for h in recorded] == [
        "Wetour Robotics Ltd filed a Form 6-K with the SEC: FORM OF SECURITIES PURCHASE AGREEMENT"]

    replay = graph._replay_settings(_replay_settings_for(tmp_path), lc.check_manifest(final))
    with respx.mock(assert_all_called=False) as offline:
        offline.route().mock(side_effect=AssertionError("replay must not touch the network"))
        source = build(replay, "rep").news_sources[-1]
        assert source.name == "sec" and source.headlines("WETO", sec.SINCE, sec.UNTIL) == recorded

# --- Fix round 1 --------------------------------------------------------------------------------------

def test_recorded_source_replays_a_recorded_failure_with_the_same_type_and_message(tmp_path):
    """K2: a failure seen while recording fails again on replay with the same text, so the model sees what it saw."""
    from nasdaq_agent.replay.proxies import RecordedSource
    from nasdaq_agent.sources.errors import SourceError, SourceUnavailable

    class Flaky:
        name = "yfinance"

        def bars(self, symbol, start, end):
            raise SourceUnavailable("GET https://query1.finance.yahoo.com/x: TransientHttpError: HTTP 429")

        def corporate_actions(self, symbol, start, end):
            raise ValueError("unexpected frame layout")

    start, end = date(2026, 9, 17), date(2026, 9, 24)
    recording = RecordedSource(Flaky(), tmp_path, "record")
    with pytest.raises(SourceUnavailable) as recorded_error:
        recording.bars("ACME", start, end)
    with pytest.raises(ValueError):
        recording.corporate_actions("ACME", start, end)
    replay = RecordedSource(fakes.FakeHistorySource(name="yfinance", data={"ACME": fakes.series("ACME", fakes.CLOSES)}),
                            tmp_path, "replay")
    with pytest.raises(SourceUnavailable) as replayed_error:
        replay.bars("ACME", start, end)
    assert type(replayed_error.value) is SourceUnavailable and str(replayed_error.value) == str(recorded_error.value)
    # A failure outside the SourceError family replays with the text the tools show for it ("Type: message").
    with pytest.raises(SourceError) as other:
        replay.corporate_actions("ACME", start, end)
    assert str(other.value) == "ValueError: unexpected frame layout"


def test_a_recorded_no_data_answer_replays_as_no_data(tmp_path):
    """A source that had no bars while recording says so again on replay. find_top_gainer skips a candidate on
    SourceNoData but stops on any other failure, so replaying it as a plain SourceError would change the run."""
    from nasdaq_agent.replay.proxies import RecordedSource
    from nasdaq_agent.sources.errors import SourceNoData

    class NoBars:
        name = "yfinance"

        def bars(self, symbol, start, end):
            raise SourceNoData(f"yfinance: no bars for {symbol} between {start} and {end}")

        def corporate_actions(self, symbol, start, end):
            return []

    start, end = date(2026, 9, 17), date(2026, 9, 24)
    with pytest.raises(SourceNoData):
        RecordedSource(NoBars(), tmp_path, "record").bars("ACME", start, end)
    replay = RecordedSource(fakes.FakeHistorySource(name="yfinance"), tmp_path, "replay")
    with pytest.raises(SourceNoData) as replayed:
        replay.bars("ACME", start, end)
    assert type(replayed.value) is SourceNoData


def test_http_client_replays_a_recorded_failure(tmp_path, monkeypatch):
    """K2 for our own HTTP client: a request that exhausted its retries while recording fails the same way on replay."""
    import httpx
    import respx
    from nasdaq_agent.replay.http_cassette import HttpCassette
    from nasdaq_agent.sources import http as h
    from nasdaq_agent.sources.errors import SourceUnavailable
    monkeypatch.setattr(h, "RETRY_WAIT_MAX_SECONDS", 0.01)
    url = "https://api.nasdaq.com/api/screener/stocks"
    with respx.mock(assert_all_called=True) as live:
        live.get(url).mock(side_effect=httpx.ConnectError("connection refused"))
        with pytest.raises(SourceUnavailable) as recorded_error:
            h.HttpClient(1, 1, cassette=HttpCassette(tmp_path, "record")).get_json(url, params={"limit": "25"})
    with respx.mock(assert_all_called=False) as offline:
        offline.route().mock(side_effect=AssertionError("replay must not touch the network"))
        with pytest.raises(SourceUnavailable) as replayed_error:
            h.HttpClient(1, 1, cassette=HttpCassette(tmp_path, "replay")).get_json(url, params={"limit": "25"})
    assert str(replayed_error.value) == str(recorded_error.value)


def test_recorded_failure_names_no_class_to_build(tmp_path):
    """A recorded failure's type is a name looked up in a fixed table, never a class the cassette can choose."""
    from nasdaq_agent.replay.http_cassette import HttpCassette
    from nasdaq_agent.sources.errors import CassetteMiss, SourceError
    cassette = HttpCassette(tmp_path, "record")
    cassette.store_failure("k1", SourceError("x"))
    entry = tmp_path / "http" / "k1.json"
    for hostile in ({"failure": {"type": "os.system", "message": "x", "source_error": True}},
                    {"failure": {"type": "SourceError", "message": 1, "source_error": True}},
                    {"failure": "SourceError"}):
        entry.write_text(json.dumps(hostile))
        with pytest.raises(CassetteMiss):
            HttpCassette(tmp_path, "replay").lookup("k1")


def test_recorded_source_rejects_an_entry_of_the_wrong_shape(tmp_path):
    """Fix round 1, minor 3: a list where the method returns a list, an object where it returns an object."""
    from nasdaq_agent.replay.proxies import RecordedSource
    from nasdaq_agent.sources.errors import CassetteMiss
    start, end = date(2026, 9, 17), date(2026, 9, 24)
    recording = RecordedSource(fakes.FakeHistorySource(data={"ACME": fakes.series("ACME", fakes.CLOSES)}), tmp_path, "record")
    recording.bars("ACME", start, end)
    recording.corporate_actions("ACME", start, end)
    entries = {json.loads(p.read_text()).__class__: p for p in (tmp_path / "sources").glob("*.json")}
    # Each entry holds data that would validate as the method's model, only in the other shape: without the check,
    # bars would return a list and corporate_actions a bare object.
    entries[dict].write_text(json.dumps([fakes.series("ACME", fakes.CLOSES).model_dump(mode="json")]))  # bars: object
    entries[list].write_text(json.dumps({"date": "2026-09-18", "kind": "split", "ratio": 2.0}))  # a list method
    replay = RecordedSource(fakes.FakeHistorySource(), tmp_path, "replay")
    with pytest.raises(CassetteMiss):
        replay.bars("ACME", start, end)
    with pytest.raises(CassetteMiss):
        replay.corporate_actions("ACME", start, end)


def test_alpha_vantage_quota_is_untouched_in_replay(tmp_path):
    """K4: replay makes no call, so it spends no quota -- quota.json is neither read nor written."""
    from nasdaq_agent.replay.http_cassette import HttpCassette
    from nasdaq_agent.sources.alphavantage import ALPHAVANTAGE_URL, AlphaVantageGainerSource
    from nasdaq_agent.sources.http import DailyQuota, HttpClient, request_key
    from nasdaq_agent.universe import Universe
    from tests.unit.test_universe import SAMPLE
    body = {"top_gainers": [{"ticker": "AAPL", "price": "110", "change_amount": "10", "change_percentage": "10%",
                             "volume": "1000"}]}
    params = {"function": "TOP_GAINERS_LOSERS", "apikey": "placeholder"}
    HttpCassette(tmp_path, "record").store(request_key("GET", ALPHAVANTAGE_URL, params), 200, json.dumps(body))
    quota_path = tmp_path / "state" / "quota.json"
    quota = DailyQuota("alphavantage", 1, quota_path)

    class Untouchable(DailyQuota):
        def _load(self):
            raise AssertionError("replay read the quota")

        def _store_atomic(self, data):
            raise AssertionError("replay wrote the quota")

    source = AlphaVantageGainerSource(HttpClient(1, 1, cassette=HttpCassette(tmp_path, "replay")), "placeholder",
                                      Universe.from_text(SAMPLE), Untouchable(quota.name, quota.limit, quota_path))
    assert [c.symbol for c in source.top_candidates(date(2026, 9, 24), date(2026, 9, 23), 5)] == ["AAPL"]
    assert not quota_path.exists()


def test_run_python_result_floats_are_rounded_for_the_model_only(ready, deps):
    """K1: numpy builds can differ in the last digit across platforms, which would change the next prompt; the model
    sees 6 decimal places, the run keeps full precision."""
    from nasdaq_agent.agent.tools.code import make_run_python
    precise = success(avg_daily_change_pct=2.15267912345678, volatility_annualized_pct=52.87400000000001,
                      daily_changes_pct=[2.0, -2.450980392156863, 6.532663316582915, 0.9433962264150943, 3.738317757009346])
    deps.runner = FakeRunner([precise])
    out = json.loads(make_run_python(ready, deps).invoke({"code": "result = {}"}))
    assert out["result"]["avg_daily_change_pct"] == 2.152679 and out["result"]["volatility_annualized_pct"] == 52.874
    assert out["result"]["daily_changes_pct"] == [2.0, -2.45098, 6.532663, 0.943396, 3.738318]
    assert ready.analysis.latest_result.avg_daily_change_pct == 2.15267912345678


def test_verify_rejection_message_rounds_floats(ready, deps):
    from nasdaq_agent.agent.tools.code import make_run_python
    from nasdaq_agent.agent.tools.verify import make_verify_analysis
    deps.runner = FakeRunner([success(volatility_annualized_pct=47.30123456789)])
    make_run_python(ready, deps).invoke({"code": "a"})
    message = make_verify_analysis(ready, deps).invoke({})
    assert message.startswith("ERROR") and "model 47.301235 vs verifier " in message
    assert "47.30123456789" not in message
    verifier_text = message.split("vs verifier ", 1)[1].split(";", 1)[0]
    assert len(verifier_text.split(".", 1)[1]) <= 6
    row = next(r for r in ready.analysis.verification if r.metric == "volatility_annualized_pct")
    assert row.model_value == "47.30123456789"  # the run's own record keeps full precision


def test_round_floats_normalises_negative_zero_and_leaves_other_values():
    from nasdaq_agent.agent.tools.common import round_floats
    assert round_floats({"a": -0.0000001, "b": [1.23456789, "x", True, 3], "c": None}) == \
        {"a": 0.0, "b": [1.234568, "x", True, 3], "c": None}
    assert str(round_floats(-0.0000001)) == "0.0"


def test_manifest_records_the_environment_and_the_exit_code(tmp_path):
    """Important 2 and K6: the manifest names the recording environment and the run's exit code."""
    import platform
    from importlib.metadata import version
    from nasdaq_agent.replay import llm_cache as lc
    from nasdaq_agent.replay.environment import ENVIRONMENT_PACKAGES, current_environment
    lc.write_manifest(tmp_path, NOW_ISO, environment=current_environment("subprocess"), exit_code=2)
    manifest = lc.check_manifest(tmp_path)
    assert manifest["exit_code"] == 2 and lc.recorded_exit_code(tmp_path) == 2
    environment = manifest["environment"]
    assert environment["python"] == platform.python_version() and environment["sandbox_backend"] == "subprocess"
    assert set(ENVIRONMENT_PACKAGES) == {"langchain-core", "langchain", "langgraph", "langchain-google-genai",
                                         "langchain-anthropic", "langchain-openai", "pydantic", "numpy", "pandas"}
    assert environment["packages"] == {name: version(name) for name in ENVIRONMENT_PACKAGES}
    assert lc.recorded_exit_code(tmp_path / "nowhere") is None


def test_environment_differences_name_each_difference():
    from nasdaq_agent.replay.environment import current_environment, environment_differences
    here = current_environment("subprocess")
    recorded = {**here, "python": "3.12.0", "sandbox_backend": "docker",
                "packages": {**here["packages"], "langchain-core": "0.0.1"}}
    differences = environment_differences(recorded, here)
    assert "python 3.12.0 recorded, " + here["python"] + " here" in differences
    assert f"langchain-core 0.0.1 recorded, {here['packages']['langchain-core']} here" in differences
    assert "sandbox backend docker recorded, subprocess here" in differences
    assert environment_differences(here, here) == []
    # The backend is compared only once this run knows its own.
    assert not any("backend" in d for d in environment_differences(recorded, current_environment(None)))
    assert environment_differences(None, here) == ["the cassette does not record its environment"]


def test_cassette_miss_leads_with_the_environment_differences():
    """Important 2: during a replay whose environment differs, every CassetteMiss says so, first -- run errors are cut
    to a few hundred characters."""
    from nasdaq_agent.sources.errors import REPLAY_ENVIRONMENT_DIFFERENCES, CassetteMiss
    assert str(CassetteMiss("no recording for GET x")) == "no recording for GET x"
    token = REPLAY_ENVIRONMENT_DIFFERENCES.set(("numpy 2.1.3 recorded, 2.5.3 here", "sandbox backend docker recorded, subprocess here"))
    try:
        assert str(CassetteMiss("no recording for GET x")) == (
            "replay environment differs from the recording (numpy 2.1.3 recorded, 2.5.3 here; "
            "sandbox backend docker recorded, subprocess here): no recording for GET x")
    finally:
        REPLAY_ENVIRONMENT_DIFFERENCES.reset(token)
    assert str(CassetteMiss("no recording for GET x")) == "no recording for GET x"


def test_compose_invalid_narrative_message_carries_no_pydantic_url(ready, deps):
    """Important 2: the error text reaches the model, and pydantic's documentation URL names its version."""
    from nasdaq_agent.agent.tools.compose import make_compose_report
    _verified(ready, deps)
    out = make_compose_report(ready, deps).invoke({**_good_args(), "trend_paragraph": "x" * 900})
    assert out.startswith("ERROR: invalid narrative") and "trend_paragraph" in out
    assert "errors.pydantic.dev" not in out and "http" not in out


# --- Fix round 2: recording must never move or delete anything that is not a cassette ------------------

RUN_ID = "20260924T220000Z-abc123"


def _layout(root: Path) -> dict[str, bytes | None]:
    """Every path under root with its bytes (None for a directory): any change at all shows up."""
    return {p.relative_to(root).as_posix(): (None if p.is_dir() else p.read_bytes())
            for p in sorted(root.rglob("*"))}


def _package_like(root: Path) -> Path:
    """The reviewer's probe A and the graph probe: a package directory that happens to hold a "sources" subpackage."""
    (root / "sources" / "sub").mkdir(parents=True)
    (root / "__init__.py").write_text("")
    (root / "sources" / "__init__.py").write_text("")
    (root / "sources" / "adapters.py").write_text("UNCOMMITTED WORK")
    (root / "sources" / "sub" / "deep.py").write_text("DEEP CODE")
    return root


def _refused_and_unchanged(final: Path, expected_offender: str) -> None:
    from nasdaq_agent.replay.recording import CassetteDirectoryRefused, start_recording
    before, siblings = _layout(final), sorted(p.name for p in final.parent.iterdir())
    with pytest.raises(CassetteDirectoryRefused) as refused:
        start_recording(final, RUN_ID)
    assert expected_offender in str(refused.value)
    assert _layout(final) == before and sorted(p.name for p in final.parent.iterdir()) == siblings


def test_guard_refuses_a_package_directory_holding_a_sources_subpackage(tmp_path):
    _refused_and_unchanged(_package_like(tmp_path / "mypkg"), "__init__.py")


def test_guard_refuses_a_directory_whose_manifest_is_not_a_cassette_manifest(tmp_path):
    """Probe B: a web app's public/ directory with a PWA manifest."""
    public = tmp_path / "public"
    public.mkdir()
    (public / "manifest.json").write_text(json.dumps({"name": "app"}))
    _refused_and_unchanged(public, "manifest.json")
    (public / "index.html").write_text("<html>")
    _refused_and_unchanged(public, "index.html")


def test_guard_refuses_the_repositorys_own_package_layout(tmp_path):
    """Probe H: nasdaq_agent/ passed the name-based guard because it holds nasdaq_agent/sources/."""
    from nasdaq_agent.replay.recording import CassetteDirectoryRefused, check_cassette_directory
    import nasdaq_agent
    copy = tmp_path / "nasdaq_agent"
    for relative in ("__init__.py", "cli.py", "sources/__init__.py", "sources/http.py", "replay/recording.py"):
        (copy / relative).parent.mkdir(parents=True, exist_ok=True)
        (copy / relative).write_text("code")
    _refused_and_unchanged(copy, "__init__.py")
    with pytest.raises(CassetteDirectoryRefused):  # and the real package, checked read-only
        check_cassette_directory(Path(nasdaq_agent.__file__).parent)


def test_guard_refuses_a_home_like_directory_and_symlinks_inside_a_cassette(tmp_path):
    """Probes C and G. The symlink targets hold entry-like files, so only the symlink itself can be the reason."""
    home = tmp_path / "home"
    (home / "sources" / "proj").mkdir(parents=True)
    (home / "sources" / "proj" / "main.c").write_text("int main(){}")
    _refused_and_unchanged(home, "sources/proj")
    from nasdaq_agent.replay import llm_cache as lc
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "k1.json").write_text("precious, and named like an entry")
    linked_dir, linked_file, linked_manifest = (tmp_path / name for name in ("dir-link", "file-link", "manifest-link"))
    for cassette in (linked_dir, linked_file, linked_manifest):
        lc.write_manifest(cassette, NOW_ISO, exit_code=0)
    (linked_dir / "http").symlink_to(outside, target_is_directory=True)
    (linked_file / "http").mkdir()
    (linked_file / "http" / "k1.json").symlink_to(outside / "k1.json")
    (tmp_path / "real-manifest.json").write_text((linked_manifest / "manifest.json").read_text())
    (linked_manifest / "manifest.json").unlink()
    (linked_manifest / "manifest.json").symlink_to(tmp_path / "real-manifest.json")
    _refused_and_unchanged(linked_dir, "http is a symlink")
    _refused_and_unchanged(linked_file, "http/k1.json is a symlink")
    _refused_and_unchanged(linked_manifest, "manifest.json is a symlink")
    assert (outside / "k1.json").read_text() == "precious, and named like an entry"


def test_guard_refuses_entry_directories_holding_anything_but_entry_files(tmp_path):
    from nasdaq_agent.replay import llm_cache as lc
    from nasdaq_agent.replay.http_cassette import HttpCassette
    cassette = tmp_path / "cassette"
    lc.write_manifest(cassette, NOW_ISO, exit_code=0)
    HttpCassette(cassette, "record").store("k1", 200, "ok")
    (cassette / "llm" / "nested").mkdir(parents=True)
    _refused_and_unchanged(cassette, "llm/nested")
    (cassette / "llm" / "nested").rmdir()
    (cassette / "llm" / "notes.txt").write_text("mine")
    _refused_and_unchanged(cassette, "llm/notes.txt")


def test_guard_accepts_empty_missing_and_leftover_cassette_directories(tmp_path):
    """Empty and missing directories are accepted, and so is a leftover cassette with entry files but no manifest --
    which is then replaced."""
    from nasdaq_agent.replay import llm_cache as lc
    from nasdaq_agent.replay.http_cassette import HttpCassette
    from nasdaq_agent.replay.recording import seal_recording, start_recording
    empty = tmp_path / "empty"
    empty.mkdir()
    for accepted in (tmp_path / "missing" / "cassette", empty):
        assert start_recording(accepted, RUN_ID).is_dir()
    leftover = tmp_path / "leftover"
    HttpCassette(leftover, "record").store("k-old", 200, "left over")
    staging = start_recording(leftover, RUN_ID)
    HttpCassette(staging, "record").store("k-new", 200, "new")
    seal_recording(staging, leftover, NOW_ISO, [], [], None, 0)
    assert HttpCassette(leftover, "replay").lookup("k-new") == (200, "new")
    assert HttpCassette(leftover, "replay").lookup("k-old") is None and lc.check_manifest(leftover)["now"] == NOW_ISO


def test_symlinked_cassette_path_resolves_and_only_the_targets_entries_change(tmp_path):
    """Probe E: the path is resolved before anything else, staging included; the link itself is untouched."""
    from nasdaq_agent.replay.http_cassette import HttpCassette
    from nasdaq_agent.replay.recording import seal_recording, staging_dir, start_recording
    real = tmp_path / "real" / "cassette"
    _previous_cassette(real)
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    staging = start_recording(link, RUN_ID)
    assert staging == staging_dir(link, RUN_ID) and staging.parent == real.resolve().parent
    HttpCassette(staging, "record").store("k-new", 200, "new")
    seal_recording(staging, link, NOW_ISO, [], [], None, 0)
    assert link.is_symlink() and link.resolve() == real.resolve()
    assert HttpCassette(real, "replay").lookup("k-new") == (200, "new")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["link", "real"]
    assert sorted(p.name for p in real.parent.iterdir()) == ["cassette"]


def test_dotdot_cassette_path_resolves_before_the_staging_path(tmp_path):
    """Probe F: a path through the cassette's own sources/ and back up put the staging directory inside sources/."""
    from nasdaq_agent.replay.http_cassette import HttpCassette
    from nasdaq_agent.replay.recording import seal_recording, start_recording
    cassette = tmp_path / "cassette"
    _previous_cassette(cassette)
    (cassette / "sources").mkdir()
    staging = start_recording(cassette / "sources" / "..", RUN_ID)
    assert staging.parent == cassette.resolve().parent
    HttpCassette(staging, "record").store("k-new", 200, "new")
    seal_recording(staging, cassette / "sources" / "..", NOW_ISO, [], [], None, 0)
    assert sorted(_layout(cassette)) == ["http", "http/k-new.json", "manifest.json"]


def test_promote_reverifies_each_entry_even_when_the_guard_is_bypassed(tmp_path, monkeypatch):
    """Item 2: the destructive step checks again, so even a bypassed guard never moves or deletes other content."""
    from nasdaq_agent.replay import recording
    from nasdaq_agent.replay.http_cassette import HttpCassette
    package = _package_like(tmp_path / "mypkg")
    before = _layout(package)
    monkeypatch.setattr(recording, "check_cassette_directory", lambda final: Path(final).resolve())
    staging = recording.start_recording(package, RUN_ID)
    HttpCassette(staging, "record").store("k-new", 200, "new")
    moves, real_replace = [], recording.os.replace

    def recorded_replace(source, target):
        moves.append(Path(source))
        return real_replace(source, target)

    monkeypatch.setattr(recording.os, "replace", recorded_replace)
    with pytest.raises(recording.CassetteDirectoryRefused):
        recording.seal_recording(staging, package, NOW_ISO, [], [], None, 0)
    monkeypatch.setattr(recording.os, "replace", real_replace)
    assert package.resolve() / "sources" not in moves  # refused before it was ever moved aside
    assert _layout(package) == before
    recording.discard_recording(staging)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["mypkg"]


def test_deleting_cassette_entries_never_removes_anything_else(tmp_path):
    """The deletion primitive removes verified entry files and then only empty directories."""
    from nasdaq_agent.replay import recording
    from nasdaq_agent.replay.http_cassette import HttpCassette
    directory = tmp_path / "staging"
    HttpCassette(directory, "record").store("k1", 200, "ok")
    (directory / "http" / "notes.txt").write_text("not an entry")
    (directory / "manifest.json").write_text(json.dumps({"name": "app"}))
    with pytest.raises(OSError):
        recording._delete_cassette_entries(directory)
    assert (directory / "http" / "notes.txt").read_text() == "not an entry"
    assert json.loads((directory / "manifest.json").read_text()) == {"name": "app"}
    assert not (directory / "http" / "k1.json").exists()


def test_failure_to_remove_the_backup_after_the_swap_is_a_warning_and_the_recording_is_sealed(tmp_path, monkeypatch,
                                                                                               caplog):
    """Item 3: once the new cassette is live, cleanup trouble is a warning, never "recording not sealed"."""
    from nasdaq_agent.replay import llm_cache as lc
    from nasdaq_agent.replay import recording
    from nasdaq_agent.replay.http_cassette import HttpCassette
    final = tmp_path / "cassette"
    _previous_cassette(final)
    real_delete = recording._delete_cassette_entries

    def backup_stays(directory):
        if directory.name.endswith(recording.BACKUP_SUFFIX):
            raise PermissionError("injected cleanup failure")
        real_delete(directory)

    monkeypatch.setattr(recording, "_delete_cassette_entries", backup_stays)
    staging = recording.start_recording(final, RUN_ID)
    HttpCassette(staging, "record").store("k-new", 200, "new")
    with caplog.at_level("WARNING", logger="nasdaq_agent.replay"):
        recording.seal_recording(staging, final, NOW_ISO, [], [], None, 0)
    assert HttpCassette(final, "replay").lookup("k-new") == (200, "new") and lc.check_manifest(final)["now"] == NOW_ISO
    assert any("could not be removed" in r.message and "injected cleanup failure" in r.message for r in caplog.records)


@pytest.mark.parametrize("failing_move", range(1, 8))
def test_a_failed_swap_rolls_back_to_the_previous_cassette(tmp_path, monkeypatch, failing_move):
    """The reviewer's rollback probe: whichever move fails, the previous cassette comes back byte for byte."""
    from nasdaq_agent.replay import recording
    from nasdaq_agent.replay.http_cassette import HttpCassette
    final = tmp_path / "cassette"
    _previous_cassette(final)
    (final / "llm").mkdir()
    (final / "llm" / "e1.json").write_text("{}")
    before = _layout(final)
    staging = recording.start_recording(final, RUN_ID)
    HttpCassette(staging, "record").store("k-new", 200, "new")
    (staging / "llm").mkdir()
    (staging / "llm" / "n1.json").write_text("{}")
    real_replace, calls = recording.os.replace, {"n": 0}

    def flaky(source, target):
        calls["n"] += 1
        if calls["n"] == failing_move:
            raise OSError("injected failure")
        return real_replace(source, target)

    monkeypatch.setattr(recording.os, "replace", flaky)
    try:
        recording.seal_recording(staging, final, NOW_ISO, [], [], None, 0)
        sealed = True
    except OSError:
        sealed = False
    monkeypatch.setattr(recording.os, "replace", real_replace)
    recording.discard_recording(staging)
    if not sealed:
        assert _layout(final) == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["cassette"]


def test_recording_lock_is_exclusive_and_released(tmp_path):
    """Item 5: one recording at a time per cassette, through an O_EXCL lock file beside the resolved directory."""
    from nasdaq_agent.replay.recording import RecordingInProgress, acquire_recording_lock, release_recording_lock
    lock = acquire_recording_lock(tmp_path / "cassette")
    assert lock.parent == tmp_path.resolve() and lock.exists()
    with pytest.raises(RecordingInProgress) as refused:
        acquire_recording_lock(tmp_path / "sub" / ".." / "cassette")  # the same directory, spelled differently
    assert str(lock) in str(refused.value)
    release_recording_lock(lock)
    assert not lock.exists()
    release_recording_lock(acquire_recording_lock(tmp_path / "cassette"))


def test_recorded_exit_code_reads_only_a_sealable_code(tmp_path):
    """Item 4: a manifest is written only after exit 0 or 2, so a hand-edited 1 (or anything else) reads as missing."""
    from nasdaq_agent.replay import llm_cache as lc
    lc.write_manifest(tmp_path, NOW_ISO, exit_code=2)
    assert lc.recorded_exit_code(tmp_path) == 2
    manifest = json.loads((tmp_path / lc.MANIFEST).read_text())
    for edited, expected in ((0, 0), (1, None), (3, None), (True, None), ("2", None), (None, None)):
        (tmp_path / lc.MANIFEST).write_text(json.dumps({**manifest, "exit_code": edited}))
        assert lc.recorded_exit_code(tmp_path) == expected, edited


def test_the_backup_is_verified_again_before_it_is_deleted(tmp_path, monkeypatch):
    """Item 2's last layer: a file that appears in a moved-aside entry during the swap stops the deletion -- the swap
    is rolled back, the previous cassette comes back, and the file is kept."""
    from nasdaq_agent.replay import llm_cache as lc
    from nasdaq_agent.replay import recording
    from nasdaq_agent.replay.http_cassette import HttpCassette
    final = tmp_path / "cassette"
    _previous_cassette(final)
    staging = recording.start_recording(final, RUN_ID)
    HttpCassette(staging, "record").store("k-new", 200, "new")
    real_replace = recording.os.replace

    def replace_then_intrude(source, target):
        real_replace(source, target)
        if Path(target).name == "http" and Path(target).parent.name.endswith(recording.BACKUP_SUFFIX):
            (Path(target) / "intruder.txt").write_text("appeared mid-swap")

    monkeypatch.setattr(recording.os, "replace", replace_then_intrude)
    with pytest.raises(recording.CassetteDirectoryRefused):
        recording.seal_recording(staging, final, NOW_ISO, [], [], None, 0)
    monkeypatch.setattr(recording.os, "replace", real_replace)
    assert (final / "http" / "intruder.txt").read_text() == "appeared mid-swap"
    assert HttpCassette(final, "replay").lookup("k-old") == (200, "previous recording")
    assert lc.check_manifest(final)["now"] == "2026-01-02T22:00:00+00:00"


def test_llm_cache_key_ignores_the_request_timeout_and_retries(tmp_path):
    # A timeout or retry count changes how long we wait, never what the model says, so changing either after recording
    # must not break replay. Everything that does shape the answer, such as the model name, stays in the key.
    from pydantic import SecretStr
    from langchain_openai import ChatOpenAI
    from nasdaq_agent.replay.llm_cache import JsonFileLLMCache

    def identity(model="m", timeout=90, retries=2):
        return ChatOpenAI(model=model, api_key=SecretStr("k"), base_url="http://localhost:1", temperature=0,
                          timeout=timeout, max_retries=retries)._get_llm_string()

    cache = JsonFileLLMCache(tmp_path, "record")
    assert cache._path("[]", identity(timeout=90, retries=2)) == cache._path("[]", identity(timeout=30, retries=0))
    assert cache._path("[]", identity(model="m")) != cache._path("[]", identity(model="other"))
