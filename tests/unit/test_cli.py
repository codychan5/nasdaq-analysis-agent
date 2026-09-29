# tests/unit/test_cli.py
"""The nasdaq-agent command line: what each command runs and prints, with resume taking a well-formed id through
--run-id; one clean stderr line and exit 1 for any error before a run exists, never a traceback or a settings value; a
user-supplied run id that never becomes a path outside the artefacts directory; a run tied to the mode it started in;
replay --check; usage errors exiting 1 through main(); and AGENT_SESSION_DATE, the trading day a run reports on."""
import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

WELL_FORMED_RUN_ID = "20260924T220000Z-abc123"
CREDENTIAL_VARIABLES = ("GOOGLE_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "MASSIVE_API_KEY", "ALPHAVANTAGE_API_KEY",
                        "OPENROUTER_API_KEY", "LANGSMITH_API_KEY", "LANGSMITH_TRACING", "LANGSMITH_PROJECT")


@pytest.fixture(autouse=True)
def _hermetic_cwd(tmp_path, monkeypatch):
    """Settings reads ./.env by default; run every command from an empty directory so a developer's .env never leaks
    into these tests. Exported credentials and AGENT_* settings are cleared for the same reason; each test sets what
    it needs."""
    monkeypatch.chdir(tmp_path)
    for name in [*CREDENTIAL_VARIABLES, *(n for n in os.environ if n.startswith("AGENT_"))]:
        monkeypatch.delenv(name, raising=False)


def test_run_prints_outcome_and_exit_code(monkeypatch):
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    from nasdaq_agent import cli
    from nasdaq_agent.agent.graph import RunOutcome
    seen = {}
    def fake_run_once(settings, now=None, **kw):
        seen["mode"], seen["now"] = settings.mode.value, now
        return RunOutcome(run_id="r1", exit_code=2, artifacts_path="/tmp/r1")
    monkeypatch.setattr(cli, "run_once", fake_run_once)
    result = CliRunner().invoke(cli.app, ["run", "--now", "2026-09-24T22:00:00+00:00"])
    assert result.exit_code == 2 and json.loads(result.stdout.strip().splitlines()[-1])["run_id"] == "r1"
    assert seen["mode"] == "live" and seen["now"].isoformat() == "2026-09-24T22:00:00+00:00"

def test_replay_forces_replay_mode(monkeypatch):
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    from nasdaq_agent import cli
    from nasdaq_agent.agent.graph import RunOutcome
    seen = {}
    monkeypatch.setattr(cli, "run_once", lambda settings, now=None, **kw: seen.setdefault("mode", settings.mode.value) and RunOutcome("r2", 0, "/tmp/r2"))
    result = CliRunner().invoke(cli.app, ["replay"])
    assert result.exit_code == 0 and seen["mode"] == "replay"

def test_resume_calls_resume_run(monkeypatch):
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    from nasdaq_agent import cli
    from nasdaq_agent.agent.graph import RunOutcome
    monkeypatch.setattr(cli, "resume_run", lambda settings, run_id, **kw: RunOutcome(run_id, 0, "/tmp/x"))
    result = CliRunner().invoke(cli.app, ["resume", "--run-id", WELL_FORMED_RUN_ID])
    assert result.exit_code == 0 and f'"run_id": "{WELL_FORMED_RUN_ID}"' in result.stdout


# --- Output contract ---------------------------------------------------------------------------------

def test_record_forces_record_mode_and_passes_now(monkeypatch):
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    from nasdaq_agent import cli
    from nasdaq_agent.agent.graph import RunOutcome
    seen = {}

    def fake_run_once(settings, now=None, **kw):
        seen["mode"], seen["now"] = settings.mode.value, now
        return RunOutcome("r3", 0, "/tmp/r3")

    monkeypatch.setattr(cli, "run_once", fake_run_once)
    result = CliRunner().invoke(cli.app, ["record", "--now", "2026-09-24T22:00:00+00:00"])
    assert result.exit_code == 0 and seen["mode"] == "record" and seen["now"].isoformat() == "2026-09-24T22:00:00+00:00"
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {"run_id": "r3", "exit_code": 0, "artifacts_path": "/tmp/r3"}


def test_env_file_option_is_read(monkeypatch, tmp_path):
    monkeypatch.delenv("AGENT_EMAIL_TO", raising=False)
    env = tmp_path / "alt.env"
    env.write_text("AGENT_EMAIL_TO=alt@example.com\nAGENT_BENCHMARK_SYMBOL=QQQ\n")
    from nasdaq_agent import cli
    from nasdaq_agent.agent.graph import RunOutcome
    seen = {}

    def fake_run_once(settings, now=None, **kw):
        seen["to"], seen["bench"] = settings.email_to, settings.benchmark_symbol
        return RunOutcome("r4", 0, "/tmp/r4")

    monkeypatch.setattr(cli, "run_once", fake_run_once)
    result = CliRunner().invoke(cli.app, ["run", "--env-file", str(env)])
    assert result.exit_code == 0 and seen == {"to": "alt@example.com", "bench": "QQQ"}


# --- Clean errors before a run exists --------------------------------------------------------------------

def _single_clean_error_line(result) -> str:
    assert result.exit_code == 1
    assert result.stdout == ""
    lines = result.stderr.strip().splitlines()
    assert len(lines) == 1 and "Traceback" not in result.stderr
    return lines[0]


def test_invalid_settings_print_field_names_only(monkeypatch):
    monkeypatch.setenv("AGENT_EMAIL_TO", "not-an-email")
    monkeypatch.setenv("AGENT_SMTP_PASSWORD", "hunter2-secret")
    monkeypatch.setenv("AGENT_MAX_CODE_RUNS", "lots")
    from nasdaq_agent import cli
    line = _single_clean_error_line(CliRunner().invoke(cli.app, ["run"]))
    assert "email_to" in line and "max_code_runs" in line
    assert "not-an-email" not in line and "hunter2-secret" not in line and "lots" not in line


def test_cross_field_settings_error_names_the_fields_not_the_values(monkeypatch):
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    monkeypatch.setenv("AGENT_SMTP_HOST", "smtp.internal.example.com")
    from nasdaq_agent import cli
    line = _single_clean_error_line(CliRunner().invoke(cli.app, ["run"]))
    assert "smtp_username" in line and "smtp_password" in line and "smtp_from" in line
    assert "smtp.internal.example.com" not in line


def test_missing_required_setting_is_named(monkeypatch):
    monkeypatch.delenv("AGENT_EMAIL_TO", raising=False)
    from nasdaq_agent import cli
    assert "email_to" in _single_clean_error_line(CliRunner().invoke(cli.app, ["replay"]))


def test_now_without_an_offset_is_a_clean_error(monkeypatch):
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    from nasdaq_agent import cli
    monkeypatch.setattr(cli, "run_once", lambda *a, **k: pytest.fail("no run may start with an ambiguous clock"))
    for bad in ("2026-09-24T22:00:00", "yesterday"):
        assert "--now" in _single_clean_error_line(CliRunner().invoke(cli.app, ["run", "--now", bad]))


def test_missing_env_file_is_a_clean_error(monkeypatch, tmp_path):
    from nasdaq_agent import cli
    line = _single_clean_error_line(CliRunner().invoke(cli.app, ["run", "--env-file", str(tmp_path / "nope.env")]))
    assert "--env-file" in line


def test_failure_before_the_run_exists_is_one_redacted_line(monkeypatch):
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    monkeypatch.setenv("GOOGLE_API_KEY", "g-secret-456")
    from nasdaq_agent import cli

    def run_dir_creation_fails(settings, now=None, **kw):
        raise PermissionError(f"cannot create run directory (key g-secret-456 in {settings.artifacts_dir})")

    monkeypatch.setattr(cli, "run_once", run_dir_creation_fails)
    line = _single_clean_error_line(CliRunner().invoke(cli.app, ["run"]))
    assert "PermissionError" in line and "g-secret-456" not in line and "***" in line


def test_real_run_directory_failure_is_a_clean_line(monkeypatch, tmp_path):
    """No fakes: the artefacts directory sits under a regular file, so creating the run directory fails in _prepare,
    before any context exists."""
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a directory")
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    monkeypatch.setenv("AGENT_ARTIFACTS_DIR", str(blocker / "runs"))
    from nasdaq_agent import cli
    line = _single_clean_error_line(CliRunner().invoke(cli.app, ["run"]))
    assert line.startswith("nasdaq-agent: error:")


# --- A run id is validated before it becomes a path --------------------------------------------------------

def test_validate_run_id_accepts_only_new_run_id_format():
    from nasdaq_agent.artifacts import new_run_id, validate_run_id
    assert validate_run_id(new_run_id())
    assert validate_run_id(WELL_FORMED_RUN_ID) == WELL_FORMED_RUN_ID
    arabic_indic_digits = "٢٠٢٦٠٩٢٤T٢٢٠٠٠٠Z-abc123"  # \d would accept these, so the pattern uses [0-9]
    for bad in ("../../etc", "abc", "20260924T220000Z-ABC123", WELL_FORMED_RUN_ID + "\n", "/" + WELL_FORMED_RUN_ID,
                "20260924T220000Z-abc123/../x", "", arabic_indic_digits):
        with pytest.raises(ValueError):
            validate_run_id(bad)


def _artifacts_env(monkeypatch, tmp_path) -> Path:
    runs = tmp_path / "runs"
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    monkeypatch.setenv("AGENT_ARTIFACTS_DIR", str(runs))
    return runs


def test_resume_rejects_path_traversal_and_creates_nothing(monkeypatch, tmp_path):
    runs = _artifacts_env(monkeypatch, tmp_path)
    before = sorted(p.name for p in tmp_path.parent.iterdir())
    from nasdaq_agent import cli
    line = _single_clean_error_line(CliRunner().invoke(cli.app, ["resume", "--run-id", "../../etc"]))
    assert "run id" in line
    assert not runs.exists() and not (tmp_path.parent / "etc").exists()
    assert sorted(p.name for p in tmp_path.parent.iterdir()) == before


def test_resume_rejects_an_unknown_run_and_creates_nothing(monkeypatch, tmp_path):
    runs = _artifacts_env(monkeypatch, tmp_path)
    from nasdaq_agent import cli
    line = _single_clean_error_line(CliRunner().invoke(cli.app, ["resume", "--run-id", WELL_FORMED_RUN_ID]))
    assert WELL_FORMED_RUN_ID in line
    assert not runs.exists()


def test_resume_run_itself_validates_before_creating_anything(monkeypatch, tmp_path):
    """The guard lives in resume_run, not only the CLI, since RunDir(...) creates the directory."""
    runs = _artifacts_env(monkeypatch, tmp_path)
    from nasdaq_agent.agent.graph import resume_run
    from nasdaq_agent.config import Settings
    settings = Settings(_env_file=None)
    with pytest.raises(ValueError):
        resume_run(settings, "../../etc")
    with pytest.raises(FileNotFoundError):
        resume_run(settings, WELL_FORMED_RUN_ID)
    assert not runs.exists()


def test_resume_refuses_a_recording(monkeypatch, tmp_path):
    """Resuming in record mode would restart the cassette and record only the remainder of the run, so it is refused
    before anything is touched; run `record` again instead."""
    runs = _artifacts_env(monkeypatch, tmp_path)
    from nasdaq_agent.agent.context import RunContext
    from nasdaq_agent.agent.graph import resume_run
    from nasdaq_agent.config import Mode, Settings
    RunContext.new(WELL_FORMED_RUN_ID, "h", str(runs / WELL_FORMED_RUN_ID)).save()
    settings = Settings(_env_file=None).model_copy(update={"mode": Mode.record})
    with pytest.raises(ValueError, match="record"):
        resume_run(settings, WELL_FORMED_RUN_ID)
    assert sorted(p.name for p in (runs / WELL_FORMED_RUN_ID).iterdir()) == ["context.json"]


# --- A run is tied to the mode it started in ----------------------------------------------------------------

def _seed_run(runs: Path, mode, sent: bool = False) -> Path:
    from nasdaq_agent.agent.context import RunContext
    ctx = RunContext.new(WELL_FORMED_RUN_ID, "h", str(runs / WELL_FORMED_RUN_ID), mode=mode)
    ctx.progress.sent = sent
    ctx.save()
    return runs / WELL_FORMED_RUN_ID


def _snapshot(root: Path) -> dict[str, bytes]:
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def _settings_in(mode):
    from nasdaq_agent.config import Settings
    return Settings(_env_file=None).model_copy(update={"mode": mode})


def test_context_stores_the_mode_it_was_created_in(tmp_path):
    from nasdaq_agent.agent.context import RunContext
    from nasdaq_agent.config import Mode
    RunContext.new(WELL_FORMED_RUN_ID, "h", str(tmp_path), mode=Mode.replay).save()
    assert RunContext.load(tmp_path).mode == Mode.replay
    assert RunContext.new(WELL_FORMED_RUN_ID, "h", str(tmp_path)).mode == Mode.live


def test_resuming_a_replay_run_is_refused_in_any_mode_and_creates_nothing(monkeypatch, tmp_path):
    """A failed replay, resumed with the default (live) mode, once ran live and sent real mail built from cassette
    data. Replay runs cannot be resumed at all -- rerunning replay is cheap."""
    runs = _artifacts_env(monkeypatch, tmp_path)
    from nasdaq_agent.agent.graph import resume_run
    from nasdaq_agent.config import Mode
    run_dir = _seed_run(runs, Mode.replay)
    before = _snapshot(runs)
    for mode in (Mode.live, Mode.replay, Mode.record):
        with pytest.raises(ValueError) as refused:
            resume_run(_settings_in(mode), WELL_FORMED_RUN_ID)
        assert "replay" in str(refused.value) and mode.value in str(refused.value)
    assert _snapshot(runs) == before and sorted(p.name for p in run_dir.iterdir()) == ["context.json"]


def test_resuming_a_live_run_in_replay_mode_is_refused_and_creates_nothing(monkeypatch, tmp_path):
    runs = _artifacts_env(monkeypatch, tmp_path)
    from nasdaq_agent.agent.graph import resume_run
    from nasdaq_agent.config import Mode
    _seed_run(runs, Mode.live)
    before = _snapshot(runs)
    with pytest.raises(ValueError) as refused:
        resume_run(_settings_in(Mode.replay), WELL_FORMED_RUN_ID)
    assert "live" in str(refused.value) and "replay" in str(refused.value)
    assert _snapshot(runs) == before


def test_mode_mismatch_through_the_cli_is_one_clean_line(monkeypatch, tmp_path):
    runs = _artifacts_env(monkeypatch, tmp_path)
    from nasdaq_agent import cli
    from nasdaq_agent.config import Mode
    _seed_run(runs, Mode.live)
    before = _snapshot(runs)
    monkeypatch.setenv("AGENT_MODE", "replay")
    line = _single_clean_error_line(CliRunner().invoke(cli.app, ["resume", "--run-id", WELL_FORMED_RUN_ID]))
    assert "live" in line and "replay" in line and _snapshot(runs) == before


def test_an_old_context_without_a_mode_resumes_as_live(monkeypatch, tmp_path):
    runs = _artifacts_env(monkeypatch, tmp_path)
    from nasdaq_agent.agent.context import RunContext
    from nasdaq_agent.agent.graph import resume_run
    from nasdaq_agent.config import Mode
    run_dir = _seed_run(runs, Mode.live, sent=True)
    stored = json.loads((run_dir / "context.json").read_text())
    del stored["mode"]  # written before contexts stored their mode
    (run_dir / "context.json").write_text(json.dumps(stored))
    assert RunContext.load(run_dir).mode == Mode.live
    outcome = resume_run(_settings_in(Mode.live), WELL_FORMED_RUN_ID)  # already sent: resolves without a model
    assert outcome.exit_code == 0


def test_resume_with_changed_settings_warns_but_proceeds(monkeypatch, tmp_path):
    runs = _artifacts_env(monkeypatch, tmp_path)
    from nasdaq_agent.agent.graph import resume_run
    from nasdaq_agent.config import Mode
    run_dir = _seed_run(runs, Mode.live, sent=True)  # stored settings hash "h" matches no real settings
    assert resume_run(_settings_in(Mode.live), WELL_FORMED_RUN_ID).exit_code == 0
    lines = [json.loads(line) for line in (run_dir / "log.jsonl").read_text().splitlines()]
    assert any(line["level"] == "WARNING" and "settings changed" in line["msg"] for line in lines)


# --- replay --check -----------------------------------------------------------------------------------------

def _cassette_env(monkeypatch, tmp_path, recorded_exit_code):
    from nasdaq_agent.replay import llm_cache as lc
    cassette = tmp_path / "cassette"
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    monkeypatch.setenv("AGENT_CASSETTE_DIR", str(cassette))
    if recorded_exit_code is not None:
        lc.write_manifest(cassette, "2026-09-24T22:00:00+00:00", exit_code=recorded_exit_code)


def _replaying(monkeypatch, exit_code):
    from nasdaq_agent import cli
    from nasdaq_agent.agent.graph import RunOutcome
    monkeypatch.setattr(cli, "run_once", lambda settings, now=None, **kw: RunOutcome("r5", exit_code, "/tmp/r5"))
    return cli


def test_replay_check_exits_zero_when_the_exit_code_matches_the_recording(monkeypatch, tmp_path):
    _cassette_env(monkeypatch, tmp_path, recorded_exit_code=2)
    cli = _replaying(monkeypatch, 2)
    result = CliRunner().invoke(cli.app, ["replay", "--check"])
    line = json.loads(result.stdout.strip().splitlines()[-1])
    assert result.exit_code == 0 and line["exit_code"] == 2 and line["recorded_exit_code"] == 2 and line["check"] == "match"


def test_replay_check_exits_one_on_a_different_exit_code_or_no_recording(monkeypatch, tmp_path):
    _cassette_env(monkeypatch, tmp_path, recorded_exit_code=0)
    cli = _replaying(monkeypatch, 1)
    result = CliRunner().invoke(cli.app, ["replay", "--check"])
    line = json.loads(result.stdout.strip().splitlines()[-1])
    assert result.exit_code == 1 and line["recorded_exit_code"] == 0 and line["check"] == "mismatch"
    (tmp_path / "cassette" / "manifest.json").unlink()
    result = CliRunner().invoke(cli.app, ["replay", "--check"])
    assert result.exit_code == 1 and json.loads(result.stdout.strip().splitlines()[-1])["recorded_exit_code"] is None


def test_replay_without_check_keeps_the_runs_own_exit_code(monkeypatch, tmp_path):
    _cassette_env(monkeypatch, tmp_path, recorded_exit_code=0)
    cli = _replaying(monkeypatch, 2)
    result = CliRunner().invoke(cli.app, ["replay"])
    assert result.exit_code == 2 and "check" not in json.loads(result.stdout.strip().splitlines()[-1])


# --- Usage errors exit 1 through main() ----------------------------------------------------------------------

def test_console_script_runs_main():
    import tomllib
    pyproject = tomllib.loads((Path(__file__).resolve().parents[2] / "pyproject.toml").read_text())
    assert pyproject["project"]["scripts"]["nasdaq-agent"] == "nasdaq_agent.cli:main"


@pytest.mark.parametrize("argv", [["resume"], ["resume", WELL_FORMED_RUN_ID], ["bogus"], [],
                                  ["resume", "--run-id", WELL_FORMED_RUN_ID, "extra"], ["run", "--bogus-option"]])
def test_main_turns_usage_errors_into_one_clean_line_and_exit_1(argv, capsys):
    from nasdaq_agent import cli
    with pytest.raises(SystemExit) as exited:
        cli.main(argv)
    captured = capsys.readouterr()
    lines = captured.err.strip().splitlines()
    assert exited.value.code == 1 and len(lines) == 1 and lines[0].startswith("nasdaq-agent: error:")
    assert "Traceback" not in captured.err + captured.out


def test_main_passes_the_runs_exit_code_through(monkeypatch, capsys):
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    cli = _replaying(monkeypatch, 2)
    with pytest.raises(SystemExit) as exited:
        cli.main(["replay"])
    assert exited.value.code == 2 and json.loads(capsys.readouterr().out.strip().splitlines()[-1])["exit_code"] == 2


def test_main_help_exits_zero(capsys):
    from nasdaq_agent import cli
    with pytest.raises(SystemExit) as exited:
        cli.main(["--help"])
    assert exited.value.code == 0 and "replay" in capsys.readouterr().out


# --- AGENT_SESSION_DATE: the trading day a run reports on -----------------------------------------------

PINNED_JULY_8 = "2026-07-08T16:30:00-04:00"


def _capture_run_once(monkeypatch, cli):
    from nasdaq_agent.agent.graph import RunOutcome
    seen = {}
    def fake_run_once(settings, now=None, **kw):
        seen["now"], seen["mode"] = now, settings.mode.value
        return RunOutcome(run_id="r1", exit_code=0, artifacts_path="/tmp/r1")
    monkeypatch.setattr(cli, "run_once", fake_run_once)
    return seen


@pytest.mark.parametrize("command", ["run", "record"])
def test_a_session_date_pins_the_runs_clock(monkeypatch, command):
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    monkeypatch.setenv("AGENT_SESSION_DATE", "2026-07-08")
    from nasdaq_agent import cli
    seen = _capture_run_once(monkeypatch, cli)
    result = CliRunner().invoke(cli.app, [command])
    assert result.exit_code == 0 and seen["now"].isoformat() == PINNED_JULY_8


def test_a_session_date_and_now_together_are_refused(monkeypatch):
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    monkeypatch.setenv("AGENT_SESSION_DATE", "2026-07-08")
    from nasdaq_agent import cli
    monkeypatch.setattr(cli, "run_once", lambda *a, **k: pytest.fail("no run may start with two clocks"))
    line = _single_clean_error_line(CliRunner().invoke(cli.app, ["run", "--now", "2026-09-24T22:00:00+00:00"]))
    assert "--now" in line and "AGENT_SESSION_DATE" in line


def test_a_session_date_that_is_not_a_trading_day_is_a_clean_error(monkeypatch):
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    monkeypatch.setenv("AGENT_SESSION_DATE", "2026-07-04")
    from nasdaq_agent import cli
    monkeypatch.setattr(cli, "run_once", lambda *a, **k: pytest.fail("no run may start for a day without a session"))
    line = _single_clean_error_line(CliRunner().invoke(cli.app, ["run"]))
    assert "AGENT_SESSION_DATE" in line and "not a NASDAQ trading day" in line


def test_the_scheduler_refuses_a_session_date(monkeypatch):
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    monkeypatch.setenv("AGENT_SESSION_DATE", "2026-07-08")
    from nasdaq_agent import cli
    monkeypatch.setattr(cli, "run_on_schedule", lambda *a, **k: pytest.fail("a pinned day must never run on a schedule"))
    line = _single_clean_error_line(CliRunner().invoke(cli.app, ["schedule"]))
    assert "AGENT_SESSION_DATE" in line


def test_replay_says_it_uses_the_recordings_day_instead_of_a_session_date(monkeypatch):
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    monkeypatch.setenv("AGENT_SESSION_DATE", "2026-07-08")
    from nasdaq_agent import cli
    seen = _capture_run_once(monkeypatch, cli)
    result = CliRunner().invoke(cli.app, ["replay"])
    assert result.exit_code == 0 and seen["mode"] == "replay" and seen["now"] is None
    assert "AGENT_SESSION_DATE" in result.stderr


def test_a_resumed_run_keeps_the_session_dates_clock(monkeypatch):
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    monkeypatch.setenv("AGENT_SESSION_DATE", "2026-07-08")
    from nasdaq_agent import cli
    from nasdaq_agent.agent.graph import RunOutcome
    seen = {}
    def fake_resume_run(settings, run_id, now=None, **kw):
        seen["now"] = now
        return RunOutcome(run_id, 0, "/tmp/x")
    monkeypatch.setattr(cli, "resume_run", fake_resume_run)
    result = CliRunner().invoke(cli.app, ["resume", "--run-id", WELL_FORMED_RUN_ID])
    assert result.exit_code == 0 and seen["now"].isoformat() == PINNED_JULY_8
