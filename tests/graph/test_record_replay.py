"""Record and replay through run_once: record a scripted run into a cassette, then replay it with a model that raises
if it is ever invoked. Also covered: replay never sleeps between model calls and never sends real mail; record reads
the clock once and seals a cassette only after exit 0 or 2 with no secret in it, never touching anything that is not
a cassette; and replay with a missing or stale cassette. Offline: the deps come from tests/graph/conftest.py's fakes,
the model is scripted."""
import json
from pathlib import Path

import pytest
from langchain_core.globals import get_llm_cache, set_llm_cache
from langchain_core.messages import AIMessage

from tests import fakes
from tests.fake_model import ScriptedChatModel, tool_call
from tests.graph.conftest import NOW, make_deps_factory
from tests.graph.test_orchestrator_scripts import happy_script
from tests.unit.test_grounding import good_narrative

WELL_FORMED_RUN_ID = "20260924T220000Z-abc123"


class StableScriptedModel(ScriptedChatModel):
    """The llm_string is part of every cache key, so it must not depend on the instance (its script, its object
    identity). Pinned here rather than relying on GenericFakeChatModel's current, empty defaults."""

    @property
    def _identifying_params(self) -> dict:
        return {"model": "scripted-for-replay"}

    @property
    def _llm_type(self) -> str:
        return "scripted-for-replay"

    def bind_tools(self, tools, **kwargs):
        """Also accepts a schema class, as with_structured_output passes one (the judge's JudgeVerdict)."""
        names = [getattr(tool, "name", getattr(tool, "__name__", str(tool))) for tool in tools]
        object.__setattr__(self, "bound_tool_names", self.bound_tool_names + [names])
        return self


class MustNotBeCalledModel(StableScriptedModel):
    """The replay model: same llm_string as the recording model, but any call means a response was not served from
    the cassette -- some run-specific value (run id, path, timestamp, message id) reached the cache key."""

    def _generate(self, *args, **kwargs):
        raise AssertionError("replay invoked the model: a prompt missed the cassette")

    def _stream(self, *args, **kwargs):
        raise AssertionError("replay invoked the model: a prompt missed the cassette")


def _modes(settings, tmp_path):
    from nasdaq_agent.config import Mode
    record = settings.model_copy(update={"mode": Mode.record, "cassette_dir": tmp_path / "cassette"})
    return record, record.model_copy(update={"mode": Mode.replay})


def _load_ctx(outcome):
    from nasdaq_agent.agent.context import RunContext
    return RunContext.load(Path(outcome.artifacts_path))


def _tool_log(outcome):
    lines = (Path(outcome.artifacts_path) / "tool_log.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def _emails(outcome):
    return sorted((Path(outcome.artifacts_path) / "outbox").glob("*.eml"))


def _tree(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): p.read_text() for p in sorted(root.rglob("*")) if p.is_file()}


def _previous_cassette(root: Path) -> dict[str, str]:
    """An earlier, valid recording at root; returned as {relative path: content} for comparison."""
    from nasdaq_agent.replay import llm_cache as lc
    from nasdaq_agent.replay.http_cassette import HttpCassette
    lc.write_manifest(root, "2026-01-02T22:00:00+00:00", exit_code=0)
    HttpCassette(root, "record").store("k-old", 200, "previous recording")
    return _tree(root)


def _siblings(cassette: Path) -> list[str]:
    """Everything beside the cassette directory: a staging or backup directory left behind would show here."""
    return sorted(p.name for p in cassette.parent.iterdir() if p.name != "runs")


def _judge_verdict_script():
    return [tool_call("JudgeVerdict", {"faithful": True, "issues": []}, "judge-1")]


@pytest.fixture(autouse=True)
def _no_global_llm_cache():
    """set_llm_cache is process-global; every test starts and ends without one."""
    set_llm_cache(None)
    yield
    set_llm_cache(None)


def test_record_then_replay_round_trip(settings, tmp_path):
    from nasdaq_agent.agent.graph import run_once
    record_settings, replay_settings = _modes(settings, tmp_path)

    recorded = run_once(record_settings, now=NOW, model=StableScriptedModel(messages=iter(happy_script())),
                        deps_factory=make_deps_factory())
    assert recorded.exit_code == 0
    manifest = json.loads((tmp_path / "cassette" / "manifest.json").read_text())
    assert manifest["now"] == NOW.isoformat()

    replayed = run_once(replay_settings, model=MustNotBeCalledModel(messages=iter([])), deps_factory=make_deps_factory())

    assert replayed.exit_code == 0, _load_ctx(replayed).errors
    assert replayed.run_id != recorded.run_id and replayed.artifacts_path != recorded.artifacts_path
    recorded_ctx, replayed_ctx = _load_ctx(recorded), _load_ctx(replayed)
    assert replayed_ctx.errors == [] and replayed_ctx.progress.sent
    assert [e["tool"] for e in _tool_log(replayed)] == [e["tool"] for e in _tool_log(recorded)]
    assert [e["outcome"] for e in _tool_log(replayed)] == [e["outcome"] for e in _tool_log(recorded)]
    assert (Path(replayed.artifacts_path) / "closing_summary.txt").read_text() == "Sent the report."
    assert replayed_ctx.report.subject == recorded_ctx.report.subject
    assert len(_emails(replayed)) == 1
    assert get_llm_cache() is None  # the run's cache is gone once the run ends


def test_run_restores_the_previous_global_llm_cache(settings, tmp_path):
    from langchain_core.caches import InMemoryCache
    from nasdaq_agent.agent.graph import run_once
    record_settings, replay_settings = _modes(settings, tmp_path)
    mine = InMemoryCache()
    set_llm_cache(mine)
    run_once(record_settings, now=NOW, model=StableScriptedModel(messages=iter(happy_script())),
             deps_factory=make_deps_factory())
    assert get_llm_cache() is mine
    run_once(replay_settings, model=MustNotBeCalledModel(messages=iter([])), deps_factory=make_deps_factory())
    assert get_llm_cache() is mine


def test_replay_forces_zero_pacing(settings, tmp_path):
    """Whatever AGENT_LLM_CALL_DELAY_SECONDS says, replay does not sleep between model calls."""
    from nasdaq_agent.agent.graph import run_once
    record_settings, replay_settings = _modes(settings, tmp_path)
    run_once(record_settings, now=NOW, model=StableScriptedModel(messages=iter(happy_script())),
             deps_factory=make_deps_factory())
    seen = {}
    inner = make_deps_factory()

    def spying_factory(settings, ctx, run_dir, clock, cassette=None):
        seen["delay"] = settings.llm_call_delay_seconds
        return inner(settings, ctx, run_dir, clock)

    slow = replay_settings.model_copy(update={"llm_call_delay_seconds": 30.0})
    outcome = run_once(slow, model=MustNotBeCalledModel(messages=iter([])), deps_factory=spying_factory)
    assert outcome.exit_code == 0 and seen["delay"] == 0


@pytest.mark.parametrize("problem", ["missing", "stale"])
def test_replay_without_a_valid_cassette_ends_in_a_failure_notice_in_the_outbox(settings, tmp_path, monkeypatch, problem):
    """A missing or stale cassette is a setup failure: a failure notice written to the outbox and exit 1, not a
    traceback -- and never real mail, whatever SMTP settings the environment holds."""
    import smtplib
    from pydantic import SecretStr
    from nasdaq_agent.agent.graph import run_once
    from nasdaq_agent.config import EmailTransport
    from nasdaq_agent.replay import llm_cache as lc
    _, replay_settings = _modes(settings, tmp_path)
    replay_settings = replay_settings.model_copy(update={
        "email_transport": EmailTransport.smtp, "smtp_host": "smtp.example.com", "smtp_username": "u",
        "smtp_password": SecretStr("pw"), "smtp_from": "bot@example.com"})
    monkeypatch.setattr(smtplib, "SMTP", lambda *a, **k: pytest.fail("replay must never open an SMTP connection"))
    if problem == "stale":
        lc.write_manifest(tmp_path / "cassette", NOW.isoformat())
        monkeypatch.setattr(lc, "prompt_fingerprint", lambda: "templates-changed")

    outcome = run_once(replay_settings, model=MustNotBeCalledModel(messages=iter([])), deps_factory=make_deps_factory())

    assert outcome.exit_code == 1
    (notice,) = _emails(outcome)
    body = notice.read_bytes().decode()
    assert "setup failed: CassetteMiss" in body and "record" in body
    assert json.loads((Path(outcome.artifacts_path) / "summary.json").read_text())["exit_code"] == 1


def test_record_uses_one_clock_captured_at_the_start(settings, tmp_path, monkeypatch):
    """With no --now, record captures now once, at the start; every tool sees that one instant and the manifest
    stores it. The clock here ticks a second per read, so a second read would show."""
    from datetime import datetime, timedelta
    from nasdaq_agent.agent import graph
    ticks = (NOW + timedelta(seconds=n) for n in range(1000))

    class TickingClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return next(ticks)

    monkeypatch.setattr(graph, "datetime", TickingClock)
    record_settings, _ = _modes(settings, tmp_path)
    outcome = graph.run_once(record_settings, model=StableScriptedModel(messages=iter(happy_script())),
                             deps_factory=make_deps_factory())
    assert outcome.exit_code == 0
    manifest_now = json.loads((tmp_path / "cassette" / "manifest.json").read_text())["now"]
    assert manifest_now == NOW.isoformat()
    assert {e["at"] for e in _tool_log(outcome)} == {manifest_now}


def test_failed_recording_leaves_the_previous_cassette_untouched(settings, tmp_path):
    """The manifest is written only after exit 0 or 2, and a recording happens in a staging directory, so a failed
    re-recording leaves the earlier cassette exactly as it was and nothing beside it."""
    from nasdaq_agent.agent.graph import run_once
    record_settings, _ = _modes(settings, tmp_path)
    before = _previous_cassette(tmp_path / "cassette")
    script = happy_script()[:3] + [AIMessage(content="I am done.")]  # ends without sending: exit 1
    outcome = run_once(record_settings, now=NOW, model=StableScriptedModel(messages=iter(script)),
                       deps_factory=make_deps_factory())
    assert outcome.exit_code == 1
    assert _tree(tmp_path / "cassette") == before and _siblings(tmp_path / "cassette") == ["cassette"]


def test_recording_holding_a_secret_fails_loudly_without_a_manifest(settings, tmp_path):
    """Cassettes are committed, so a configured secret anywhere in one blocks the manifest and the run exits 1,
    naming the file but never the secret. The previous cassette is left exactly as it was, with nothing beside it."""
    from pydantic import SecretStr
    from nasdaq_agent.agent.graph import run_once
    secret = "sk-test-cassette-leak-123"
    record_settings, _ = _modes(settings, tmp_path)
    record_settings = record_settings.model_copy(update={"google_api_key": SecretStr(secret)})
    before = _previous_cassette(tmp_path / "cassette")
    script = happy_script()[:-1] + [AIMessage(content=f"Sent the report. (debug: {secret})")]
    outcome = run_once(record_settings, now=NOW, model=StableScriptedModel(messages=iter(script)),
                       deps_factory=make_deps_factory())
    assert outcome.exit_code == 1
    log_lines = [json.loads(line) for line in (Path(outcome.artifacts_path) / "log.jsonl").read_text().splitlines()]
    assert any(line["level"] == "ERROR" and "recording not sealed" in line["msg"] for line in log_lines)
    assert secret not in (Path(outcome.artifacts_path) / "log.jsonl").read_text()
    assert _tree(tmp_path / "cassette") == before and _siblings(tmp_path / "cassette") == ["cassette"]
    ctx = _load_ctx(outcome)
    assert any("recording not sealed" in e and "llm/" in e for e in ctx.errors)
    summary = json.loads((Path(outcome.artifacts_path) / "summary.json").read_text())
    assert summary["exit_code"] == 1 and summary["sent"] is True
    assert secret not in json.dumps(ctx.errors) and secret not in json.dumps(summary)


# --- Staged recording, the real judge, exit codes, tracing, environment differences -------------------

def test_recording_replaces_the_previous_cassette_only_after_a_clean_seal(settings, tmp_path):
    """The recording is staged beside the cassette and swapped in after a clean seal; staging, backup and lock are
    all gone afterwards. The manifest records the environment and the exit code."""
    from nasdaq_agent.agent.graph import run_once
    from nasdaq_agent.replay import llm_cache as lc
    from nasdaq_agent.replay.http_cassette import HttpCassette
    record_settings, _ = _modes(settings, tmp_path)
    _previous_cassette(tmp_path / "cassette")
    outcome = run_once(record_settings, now=NOW, model=StableScriptedModel(messages=iter(happy_script())),
                       deps_factory=make_deps_factory())
    assert outcome.exit_code == 0
    manifest = lc.check_manifest(tmp_path / "cassette")
    assert manifest["now"] == NOW.isoformat() and manifest["exit_code"] == 0
    assert manifest["environment"]["sandbox_backend"] == "fake"  # the FakeRunner the deps factory supplies
    assert HttpCassette(tmp_path / "cassette", "replay").lookup("k-old") is None
    assert any((tmp_path / "cassette" / "llm").glob("*.json"))
    assert _siblings(tmp_path / "cassette") == ["cassette"]


def test_recording_refuses_a_directory_that_is_not_a_cassette(settings, tmp_path):
    """AGENT_CASSETTE_DIR pointing at a non-empty directory with no cassette in it is refused before anything is
    written or moved."""
    from nasdaq_agent.agent.graph import run_once
    from nasdaq_agent.config import Mode
    project = tmp_path / "project"
    project.mkdir()
    (project / "README.md").write_text("somebody's files")
    record_settings = settings.model_copy(update={"mode": Mode.record, "cassette_dir": project})
    outcome = run_once(record_settings, now=NOW, model=StableScriptedModel(messages=iter(happy_script())),
                       deps_factory=make_deps_factory())
    assert outcome.exit_code == 1
    assert "setup failed: CassetteDirectoryRefused" in _emails(outcome)[0].read_bytes().decode()
    assert _tree(project) == {"README.md": "somebody's files"} and _siblings(project) == ["project"]


def test_setup_failure_while_recording_removes_the_staging_directory(settings, tmp_path):
    from nasdaq_agent.agent.graph import run_once
    from nasdaq_agent.universe import UniverseError
    record_settings, _ = _modes(settings, tmp_path)
    before = _previous_cassette(tmp_path / "cassette")

    def unreachable(settings, ctx, run_dir, clock, cassette=None):
        raise UniverseError("nasdaqtrader unreachable")

    outcome = run_once(record_settings, now=NOW, model=StableScriptedModel(messages=iter([])), deps_factory=unreachable)
    assert outcome.exit_code == 1
    assert _tree(tmp_path / "cassette") == before and _siblings(tmp_path / "cassette") == ["cassette"]


def test_record_then_replay_round_trip_through_the_real_judge(settings, tmp_path):
    """The scripted model can produce the judge's structured output (a JudgeVerdict tool call), so the real judge
    path -- make_judge, with_structured_output, the LLM cache -- is recorded and replayed too."""
    from nasdaq_agent.agent.graph import run_once
    from nasdaq_agent.report.judge import make_judge
    record_settings, replay_settings = _modes(settings, tmp_path)
    recorded = run_once(record_settings, now=NOW, model=StableScriptedModel(messages=iter(happy_script())),
                        deps_factory=make_deps_factory(judge=make_judge(StableScriptedModel(messages=iter(_judge_verdict_script())))))
    assert recorded.exit_code == 0 and _load_ctx(recorded).report.judge_faithful is True
    replayed = run_once(replay_settings, model=MustNotBeCalledModel(messages=iter([])),
                        deps_factory=make_deps_factory(judge=make_judge(MustNotBeCalledModel(messages=iter([])))))
    replayed_ctx = _load_ctx(replayed)
    assert replayed.exit_code == 0 and replayed_ctx.errors == [] and replayed_ctx.report.judge_faithful is True


def _record_with_the_real_judge(settings, tmp_path):
    from nasdaq_agent.agent.graph import run_once
    from nasdaq_agent.report.judge import make_judge
    record_settings, replay_settings = _modes(settings, tmp_path)
    recorded = run_once(record_settings, now=NOW, model=StableScriptedModel(messages=iter(happy_script())),
                        deps_factory=make_deps_factory(judge=make_judge(StableScriptedModel(messages=iter(_judge_verdict_script())))))
    assert recorded.exit_code == 0
    return replay_settings


def test_replay_with_a_missing_judge_verdict_fails_loudly(settings, tmp_path):
    """In replay a judge CassetteMiss is not "judge unavailable": the run fails, with a failure notice and exit 1,
    and the run's errors name the judge."""
    from nasdaq_agent.agent.graph import run_once
    from nasdaq_agent.report.judge import make_judge
    replay_settings = _record_with_the_real_judge(settings, tmp_path)
    (judge_entry,) = [p for p in (tmp_path / "cassette" / "llm").glob("*.json") if "JudgeVerdict" in p.read_text()]
    judge_entry.unlink()
    replayed = run_once(replay_settings, model=MustNotBeCalledModel(messages=iter([])),
                        deps_factory=make_deps_factory(judge=make_judge(MustNotBeCalledModel(messages=iter([])))))
    ctx = _load_ctx(replayed)
    assert replayed.exit_code == 1 and not ctx.progress.sent and len(_emails(replayed)) == 1
    assert any("judge" in e and "CassetteMiss" in e for e in ctx.errors)
    assert not any("judge unavailable" in n for n in ctx.notes)


def test_recording_with_a_judge_failure_is_not_sealed(settings, tmp_path):
    """A judge failure while recording ("judge unavailable") leaves its verdict out of the cassette, so the
    recording is not sealed and the previous cassette stays."""
    from nasdaq_agent.agent.graph import run_once
    record_settings, _ = _modes(settings, tmp_path)
    before = _previous_cassette(tmp_path / "cassette")

    def judge_down(verified, headlines, narrative, extra_facts=None, session_facts=None):
        raise RuntimeError("provider unavailable")

    outcome = run_once(record_settings, now=NOW, model=StableScriptedModel(messages=iter(happy_script())),
                       deps_factory=make_deps_factory(judge=judge_down))
    ctx = _load_ctx(outcome)
    assert outcome.exit_code == 1 and ctx.progress.sent  # the report went out; the recording is what failed
    assert any("recording not sealed" in e and "judge" in e for e in ctx.errors)
    assert _tree(tmp_path / "cassette") == before and _siblings(tmp_path / "cassette") == ["cassette"]


def test_manifest_records_a_degraded_exit_code_and_replay_reproduces_it(settings, tmp_path):
    """The manifest stores the recorded exit code; a degraded recording (2) replays as 2."""
    from nasdaq_agent.agent.graph import run_once
    from nasdaq_agent.replay import llm_cache as lc
    record_settings, replay_settings = _modes(settings, tmp_path)
    script = happy_script()
    script[7] = tool_call("compose_report", {**good_narrative().model_dump(), "news_paragraph": "", "citations": [], "headline_notes": []}, "c8")
    script.pop(6)  # no sentiment without news
    news_down = fakes.FakeNewsSource(error="503")
    recorded = run_once(record_settings, now=NOW, model=StableScriptedModel(messages=iter(script)),
                        deps_factory=make_deps_factory(news_source=news_down))
    assert recorded.exit_code == 2 and lc.recorded_exit_code(tmp_path / "cassette") == 2
    replayed = run_once(replay_settings, model=MustNotBeCalledModel(messages=iter([])),
                        deps_factory=make_deps_factory(news_source=fakes.FakeNewsSource(error="503")))
    assert replayed.exit_code == 2


def test_replay_turns_langsmith_tracing_off(settings, tmp_path, monkeypatch):
    """Replay makes no network calls, so LangSmith tracing is off for the run even when LANGSMITH_TRACING is set, and
    back on afterwards."""
    from langchain_core.tracers.context import _tracing_v2_is_enabled
    from langsmith.utils import get_env_var, tracing_is_enabled
    from nasdaq_agent.agent.graph import run_once
    record_settings, replay_settings = _modes(settings, tmp_path)
    run_once(record_settings, now=NOW, model=StableScriptedModel(messages=iter(happy_script())),
             deps_factory=make_deps_factory())
    tracing_variables = {"LANGSMITH_TRACING": "true", "LANGSMITH_API_KEY": "lsv2-not-a-real-key"}
    for name, value in tracing_variables.items():
        monkeypatch.setenv(name, value)
    get_env_var.cache_clear()  # langsmith caches environment lookups
    try:
        assert tracing_is_enabled()
        seen = {}
        inner = make_deps_factory()

        def spying_factory(settings, ctx, run_dir, clock, cassette=None):
            seen["langsmith"], seen["langchain"] = tracing_is_enabled(), _tracing_v2_is_enabled()
            return inner(settings, ctx, run_dir, clock)

        outcome = run_once(replay_settings, model=MustNotBeCalledModel(messages=iter([])), deps_factory=spying_factory)
        assert outcome.exit_code == 0 and seen == {"langsmith": False, "langchain": False}
        assert tracing_is_enabled()  # back on once the replay's scope has ended
    finally:
        for name in tracing_variables:
            monkeypatch.delenv(name)
        get_env_var.cache_clear()  # later tests must not inherit a cached "tracing on"


def test_replay_notes_environment_differences_and_cassette_misses_name_them(settings, tmp_path):
    """A replay in a different environment logs a warning and adds a run note naming each difference,
    and every CassetteMiss raised during that run carries them."""
    from nasdaq_agent.agent.graph import run_once
    record_settings, replay_settings = _modes(settings, tmp_path)
    run_once(record_settings, now=NOW, model=StableScriptedModel(messages=iter(happy_script())),
             deps_factory=make_deps_factory())
    manifest_path = tmp_path / "cassette" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["environment"]["packages"]["langchain-core"] = "0.0.1"
    manifest["environment"]["sandbox_backend"] = "docker"
    manifest_path.write_text(json.dumps(manifest))

    replayed = run_once(replay_settings, model=MustNotBeCalledModel(messages=iter([])), deps_factory=make_deps_factory())
    notes = " | ".join(_load_ctx(replayed).notes)
    assert replayed.exit_code == 0
    assert "langchain-core 0.0.1 recorded" in notes and "sandbox backend docker recorded, fake here" in notes

    first_turn = sorted((tmp_path / "cassette" / "llm").glob("*.json"))
    for entry in first_turn:
        entry.unlink()
    missed = run_once(replay_settings, model=MustNotBeCalledModel(messages=iter([])), deps_factory=make_deps_factory())
    errors = " | ".join(_load_ctx(missed).errors)
    assert missed.exit_code == 1 and "CassetteMiss" in errors and "langchain-core 0.0.1 recorded" in errors


# --- Recording never moves or deletes anything that is not a cassette -------------------------------------

def _layout(root: Path) -> dict[str, bytes | None]:
    return {p.relative_to(root).as_posix(): (None if p.is_dir() else p.read_bytes()) for p in sorted(root.rglob("*"))}


def test_recording_into_a_package_directory_changes_nothing(settings, tmp_path):
    """AGENT_CASSETTE_DIR at a package holding a sources/ subpackage once deleted sources/adapters.py through
    run_once. It is now refused before anything is written, moved or deleted."""
    from nasdaq_agent.agent.graph import run_once
    from nasdaq_agent.config import Mode
    package = tmp_path / "project" / "mypkg"
    (package / "sources").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "sources" / "__init__.py").write_text("")
    (package / "sources" / "adapters.py").write_text("UNCOMMITTED WORK")
    before = _layout(package)
    record_settings = settings.model_copy(update={"mode": Mode.record, "cassette_dir": package})
    outcome = run_once(record_settings, now=NOW, model=StableScriptedModel(messages=iter(happy_script())),
                       deps_factory=make_deps_factory())
    assert outcome.exit_code == 1
    assert "setup failed: CassetteDirectoryRefused" in _emails(outcome)[0].read_bytes().decode()
    assert _layout(package) == before and _siblings(package) == ["mypkg"]


def test_a_concurrent_second_recording_is_refused(settings, tmp_path):
    """While one recording holds the lock beside the cassette, a second refuses and leaves both the cassette
    and the first recording's lock alone."""
    from nasdaq_agent.agent.graph import run_once
    from nasdaq_agent.replay.recording import acquire_recording_lock, release_recording_lock
    record_settings, _ = _modes(settings, tmp_path)
    before = _previous_cassette(tmp_path / "cassette")
    first = acquire_recording_lock(tmp_path / "cassette")
    try:
        outcome = run_once(record_settings, now=NOW, model=StableScriptedModel(messages=iter(happy_script())),
                           deps_factory=make_deps_factory())
        assert outcome.exit_code == 1
        notice = _emails(outcome)[0].read_bytes().decode()
        assert "setup failed: RecordingInProgress" in notice
        assert first.exists() and _tree(tmp_path / "cassette") == before
    finally:
        release_recording_lock(first)
    assert _siblings(tmp_path / "cassette") == ["cassette"]


def test_cleanup_failure_after_the_swap_is_a_warning_and_the_run_exits_clean(settings, tmp_path, monkeypatch):
    """Once the swap is done the new cassette is live, so a backup that cannot be removed is only a warning in the
    run's log, not "recording not sealed" and exit 1."""
    from nasdaq_agent.agent.graph import run_once
    from nasdaq_agent.replay import llm_cache as lc
    from nasdaq_agent.replay import recording
    record_settings, _ = _modes(settings, tmp_path)
    _previous_cassette(tmp_path / "cassette")
    real_delete = recording._delete_cassette_entries

    def backup_stays(directory):
        if directory.name.endswith(recording.BACKUP_SUFFIX):
            raise PermissionError("injected cleanup failure")
        real_delete(directory)

    monkeypatch.setattr(recording, "_delete_cassette_entries", backup_stays)
    outcome = run_once(record_settings, now=NOW, model=StableScriptedModel(messages=iter(happy_script())),
                       deps_factory=make_deps_factory())
    assert outcome.exit_code == 0 and lc.check_manifest(tmp_path / "cassette")["now"] == NOW.isoformat()
    lines = [json.loads(line) for line in (Path(outcome.artifacts_path) / "log.jsonl").read_text().splitlines()]
    assert any(line["level"] == "WARNING" and "could not be removed" in line["msg"] for line in lines)
    assert not any("recording not sealed" in line["msg"] for line in lines)
