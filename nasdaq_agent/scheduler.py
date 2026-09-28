"""The scheduled trigger behind `nasdaq-agent schedule`: run the agent whenever AGENT_SCHEDULE_CRON fires, read in
AGENT_SCHEDULE_TIMEZONE (16:30 New York time, Monday to Friday, by default), until the process is stopped.

A trigger whose session was already reported is skipped. On a market holiday the last completed session is the one
reported at the previous trigger, and running again would only send the same report twice. The last reported session
is kept in schedule_state.json in the artifacts directory, beside the Alpha Vantage quota file, so a restart does not
forget it. Only a run that sent its report (exit 0, or 2 with problems noted) is recorded, so a failed run's session is tried
again at the next trigger that still finds it the last completed session.

It prints one JSON line with the next run time before each wait, and one with each trigger's outcome.
"""
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from .artifacts import redact
from .calendar import CalendarError, resolve_last_completed_session
from .config import Settings
from .cron import cron_trigger, next_fire_time

if TYPE_CHECKING:
    from .agent.graph import RunOutcome

log = logging.getLogger("nasdaq_agent.scheduler")

STATE_FILENAME = "schedule_state.json"
STATE_KEY = "last_reported_session"
# Exit codes of a run that sent its report: 0 clean, 2 with problems noted under its status line.
REPORTED_EXIT_CODES = frozenset({0, 2})
# The longest single sleep. Waking every minute to re-read the clock means a machine that was suspended, or a clock
# that was changed, is noticed within a minute rather than after a day-long sleep.
POLL_SECONDS = 60.0
SKIPPED_REPORTED = "this session was already reported"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ScheduleState:
    """The last session a scheduled run reported, persisted between restarts."""

    def __init__(self, path: Path):
        self.path = path

    def last_reported_session(self) -> str | None:
        try:
            data = json.loads(self.path.read_text())
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as e:
            # Unreadable state costs at most one repeated report, never a missed one.
            log.warning("schedule state %s is unreadable (%s); treating it as empty", self.path, type(e).__name__)
            return None
        value = data.get(STATE_KEY) if isinstance(data, dict) else None
        return value if isinstance(value, str) else None

    def record(self, session: str) -> None:
        """Written through a same-directory temp file and a rename, so a reader never sees a partial write."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps({STATE_KEY: session}))
        os.replace(tmp_path, self.path)


def _wait_until(moment: datetime, clock: Callable[[], datetime], sleep: Callable[[float], None]) -> None:
    while (remaining := (moment - clock()).total_seconds()) > 0:
        sleep(min(remaining, POLL_SECONDS))


def _trigger(settings: Settings, run_agent: Callable[[Settings], "RunOutcome"], state: ScheduleState,
             scheduled_for: datetime, clock: Callable[[], datetime]) -> dict[str, Any]:
    line: dict[str, Any] = {"scheduled_for": scheduled_for.isoformat()}
    try:
        session = resolve_last_completed_session(clock()).date.isoformat()
    except CalendarError as e:
        log.error("scheduled run skipped: the trading calendar could not resolve a session (%s)", e)
        return {**line, "error": type(e).__name__}
    line["session"] = session
    if state.last_reported_session() == session:
        log.info("scheduled run skipped: session %s was already reported", session)
        return {**line, "skipped": SKIPPED_REPORTED}
    try:
        outcome = run_agent(settings)
    except Exception as e:
        # A crash must not end the schedule. The message is redacted: an exception can quote a key or password.
        log.error("scheduled run crashed: %s", redact(f"{type(e).__name__}: {e}", settings.secret_values()))
        return {**line, "error": type(e).__name__}
    if outcome.exit_code in REPORTED_EXIT_CODES:
        state.record(session)
    return {**line, "run_id": outcome.run_id, "exit_code": outcome.exit_code}


def run_on_schedule(settings: Settings, run_agent: Callable[[Settings], "RunOutcome"], *,
                    clock: Callable[[], datetime] = _utc_now, sleep: Callable[[float], None] = time.sleep,
                    max_triggers: int | None = None, report: Callable[[dict[str, Any]], None] = print) -> None:
    """Run the agent at every fire time of the settings' schedule; max_triggers bounds the loop for tests. Each
    trigger's outcome goes to `report` as one dict."""
    trigger = cron_trigger(settings.schedule_cron, settings.schedule_timezone)
    state = ScheduleState(Path(settings.artifacts_dir) / STATE_FILENAME)
    after = clock()
    triggers = 0
    while max_triggers is None or triggers < max_triggers:
        fire_at = next_fire_time(trigger, after)
        # Printed, not only logged: a long-running process that says nothing until its first trigger looks broken.
        report({"next_run": fire_at.isoformat()})
        _wait_until(fire_at, clock, sleep)
        report(_trigger(settings, run_agent, state, fire_at, clock))
        triggers += 1
        # Strictly after this fire time, and never in the past: a run that outlasts later fire times skips them rather
        # than starting them back to back.
        after = max(clock(), fire_at + timedelta(seconds=1))
