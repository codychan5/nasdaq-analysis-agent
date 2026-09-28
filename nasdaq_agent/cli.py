"""The nasdaq-agent command line (Task 23): run, resume, record and replay, and schedule, which runs the agent on the
AGENT_SCHEDULE_CRON schedule until stopped and prints one JSON line per trigger.

Each command prints one JSON line, {"run_id", "exit_code", "artifacts_path"}, and exits with the run's exit code: 0
clean, 2 degraded, 1 failed. A failure after the run's context exists comes back from run_once or resume_run as that
ordinary outcome. A failure before it exists (invalid settings, an invalid or unknown run id, a run directory that
cannot be created) prints one line to stderr and exits 1 -- never a traceback, which could carry settings
(correction d). So does a usage error, through main(), the console entry point (fix round 1, K8).
"""
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Callable, NoReturn, Sequence

import typer
import typer.main
from pydantic import ValidationError
from typer.exceptions import TyperException

from .agent.finalize import EXIT_FAILED
from .agent.graph import RunOutcome, resume_run, run_once
from .artifacts import redact
from .config import Mode, Settings, load_settings
from .replay.llm_cache import recorded_exit_code
from .scheduler import run_on_schedule

PROG_NAME = "nasdaq-agent"
ERROR_PREFIX = f"{PROG_NAME}: error:"
NOW_EXAMPLE = "2026-09-24T22:00:00+00:00"
EXIT_CHECK_MATCH = 0

# Every command catches its own errors; pretty_exceptions_show_locals=False is defence in depth, since a rich
# traceback with locals would print the settings object's fields.
app = typer.Typer(help="NASDAQ top-gainer analysis agent", add_completion=False, pretty_exceptions_show_locals=False)

NowOption = Annotated[str | None, typer.Option(help=f"Override the clock: ISO 8601 with a timezone offset, e.g. {NOW_EXAMPLE}")]
EnvFileOption = Annotated[Path | None, typer.Option(help="Read settings from this file instead of ./.env")]
RunIdOption = Annotated[str, typer.Option("--run-id", help="The run to resume, as printed by run: YYYYMMDDTHHMMSSZ-xxxxxx")]
CheckOption = Annotated[bool, typer.Option("--check", help="Compare the replayed exit code with the recorded one: "
                                                           "exit 0 when they match, 1 when they differ")]


class CliError(Exception):
    """A problem with the command line or the configuration, reported as one clean line."""


def _invalid_setting_names(error: ValidationError) -> list[str]:
    """Correction d: the names of the invalid settings, never their values -- a secret pasted into the wrong
    variable must not be printed back."""
    names: set[str] = set()
    for detail in error.errors(include_url=False, include_context=False, include_input=False):
        location = [str(part) for part in detail.get("loc", ())]
        if location:
            names.add(".".join(location))
            continue
        # A cross-field check (an incomplete SMTP configuration) has no location: print the settings fields its
        # message names, matched as whole words, and never the message itself.
        message = str(detail.get("msg", ""))
        names.update(name for name in Settings.model_fields if re.search(rf"\b{re.escape(name)}\b", message))
    return sorted(names) or ["(cross-field check)"]


def _load_settings(env_file: Path | None, mode: Mode | None) -> Settings:
    if env_file is not None and not env_file.is_file():
        raise CliError(f"--env-file {env_file} is not a readable file")
    try:
        settings = load_settings(env_file)
    except ValidationError as e:
        names = ", ".join(_invalid_setting_names(e))
        raise CliError(f"invalid settings (check .env and the AGENT_* variables): {names}") from None
    return settings.model_copy(update={"mode": mode}) if mode is not None else settings


def _parse_now(now: str | None) -> datetime | None:
    if now is None:
        return None
    try:
        parsed = datetime.fromisoformat(now)
    except ValueError:
        parsed = None
    if parsed is None or parsed.tzinfo is None:
        raise CliError(f"--now must be ISO 8601 with a timezone offset, for example {NOW_EXAMPLE}")
    return parsed


def _error_line(message: str) -> None:
    typer.echo(f"{ERROR_PREFIX} {' '.join(message.split())}", err=True)


def _fail(message: str) -> NoReturn:
    _error_line(message)
    raise typer.Exit(code=EXIT_FAILED)


def _start(start: Callable[[Settings, datetime | None], RunOutcome], env_file: Path | None, mode: Mode | None,
           now: str | None = None) -> tuple[Settings, RunOutcome]:
    settings: Settings | None = None
    try:
        when = _parse_now(now)
        settings = _load_settings(env_file, mode)
        return settings, start(settings, when)
    except CliError as e:
        _fail(str(e))
    except Exception as e:
        # Anything raised before a run exists: resume_run's run id checks, the run directory not being creatable.
        _fail(redact(f"{type(e).__name__}: {e}", settings.secret_values() if settings is not None else []))


def _emit(outcome: RunOutcome, extra: dict[str, Any] | None = None, exit_code: int | None = None) -> NoReturn:
    line = {"run_id": outcome.run_id, "exit_code": outcome.exit_code, "artifacts_path": outcome.artifacts_path}
    typer.echo(json.dumps({**line, **(extra or {})}))
    raise typer.Exit(code=outcome.exit_code if exit_code is None else exit_code)


def _run_command(start: Callable[[Settings, datetime | None], RunOutcome], env_file: Path | None,
                 mode: Mode | None, now: str | None = None) -> NoReturn:
    _emit(_start(start, env_file, mode, now)[1])


@app.command()
def run(now: NowOption = None, env_file: EnvFileOption = None) -> None:
    """Run the agent once, in the mode AGENT_MODE names (live by default)."""
    _run_command(lambda settings, when: run_once(settings, now=when), env_file, None, now)


@app.command()
def resume(run_id: RunIdOption, env_file: EnvFileOption = None) -> None:
    """Resume an interrupted run from its checkpoint."""
    _run_command(lambda settings, _: resume_run(settings, run_id), env_file, None)


@app.command()
def record(now: NowOption = None, env_file: EnvFileOption = None) -> None:
    """Run live once and record every HTTP and model response into the cassette directory."""
    _run_command(lambda settings, when: run_once(settings, now=when), env_file, Mode.record, now)


@app.command()
def replay(check: CheckOption = False, env_file: EnvFileOption = None) -> None:
    """Run the recorded cassette through the real graph, with no keys and no network."""
    settings, outcome = _start(lambda settings, _: run_once(settings), env_file, Mode.replay)
    if not check:
        _emit(outcome)
    # Fix round 1, K6: CI's replay step. A degraded recording (exit 2) replays as 2, and that is a pass here.
    recorded = recorded_exit_code(settings.cassette_dir)
    matched = recorded == outcome.exit_code
    _emit(outcome, {"recorded_exit_code": recorded, "check": "match" if matched else "mismatch"},
          exit_code=EXIT_CHECK_MATCH if matched else EXIT_FAILED)


@app.command()
def schedule(env_file: EnvFileOption = None) -> None:
    """Run the agent live whenever AGENT_SCHEDULE_CRON fires, read in AGENT_SCHEDULE_TIMEZONE (16:30 New York time,
    Monday to Friday, by default), until stopped. A trigger whose session was already reported is skipped."""
    try:
        settings = _load_settings(env_file, None)
    except CliError as e:
        _fail(str(e))
    if settings.mode is not Mode.live:
        _fail(f"schedule runs the agent live; AGENT_MODE is {settings.mode.value}")
    try:
        run_on_schedule(settings, lambda s: run_once(s), report=lambda line: typer.echo(json.dumps(line)))
    except KeyboardInterrupt:
        typer.echo(json.dumps({"schedule": "stopped"}))


def main(argv: Sequence[str] | None = None) -> NoReturn:
    """The console entry point ([project.scripts] in pyproject.toml). Runs the app without click's standalone
    handling, so a usage error -- a missing --run-id, an unknown command, no command at all -- is one clean line and
    exit 1 (fix round 1, K8): click's own exit 2 would read as a degraded run."""
    command = typer.main.get_command(app)
    try:
        code = command.main(args=list(argv) if argv is not None else None, prog_name=PROG_NAME, standalone_mode=False)
    except TyperException as e:  # typer's vendored click exceptions: usage errors and their kin
        _error_line(e.format_message() if hasattr(e, "format_message") else str(e))
        sys.exit(EXIT_FAILED)
    except typer.Abort:
        _error_line("aborted")
        sys.exit(EXIT_FAILED)
    sys.exit(code if isinstance(code, int) else 0)
