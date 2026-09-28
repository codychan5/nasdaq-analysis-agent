"""The scheduled trigger: crontab semantics (nasdaq_agent.cron) and the scheduling loop (nasdaq_agent.scheduler)."""
import json
import logging
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

NY = ZoneInfo("America/New_York")


def ny(y, m, d, hh, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=NY)


def _fires(expression, start, count, tz="America/New_York"):
    """The next `count` fire times after `start`, as New York weekday and time."""
    from nasdaq_agent.cron import cron_trigger, next_fire_time
    trigger = cron_trigger(expression, tz)
    fires, after = [], start
    for _ in range(count):
        fire = next_fire_time(trigger, after)
        fires.append(fire.astimezone(NY).strftime("%a %m-%d %H:%M"))
        after = fire + timedelta(seconds=1)
    return fires


def _weekdays(expression):
    """The weekdays an expression fires on over one week."""
    return {fire.split()[0] for fire in _fires(expression, ny(2026, 9, 27, 0), 7)}


def test_the_default_schedule_is_1630_new_york_time_monday_to_friday():
    from nasdaq_agent.config import Settings
    default = Settings.model_fields["schedule_cron"].default
    assert _fires(default, ny(2026, 9, 25, 17, 0), 6) == ["Mon 09-28 16:30", "Tue 09-29 16:30", "Wed 09-30 16:30",
                                                          "Thu 10-01 16:30", "Fri 10-02 16:30", "Mon 10-05 16:30"]


@pytest.mark.parametrize("field, days", [
    # APScheduler 3.x numbers weekdays from Monday (0 = Monday); crontab numbers them from Sunday. Read raw, "1-5" would
    # run Tuesday to Saturday.
    ("1-5", {"Mon", "Tue", "Wed", "Thu", "Fri"}),
    ("0,6", {"Sun", "Sat"}),
    ("7", {"Sun"}),
    ("5-7", {"Fri", "Sat", "Sun"}),
    ("*/2", {"Sun", "Tue", "Thu", "Sat"}),
    ("1-5/2", {"Mon", "Wed", "Fri"}),
    ("mon-fri", {"Mon", "Tue", "Wed", "Thu", "Fri"}),
    ("SAT,sun", {"Sat", "Sun"}),
    ("*", {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"}),
])
def test_the_day_of_week_field_follows_crontab_numbering(field, days):
    assert _weekdays(f"0 12 * * {field}") == days


def test_an_expression_restricting_both_day_fields_is_refused():
    # crontab runs when either day field matches, APScheduler only when both do; refuse rather than run it differently.
    from nasdaq_agent.cron import cron_trigger
    with pytest.raises(ValueError, match="not both"):
        cron_trigger("0 9 1 * 1", "America/New_York")


@pytest.mark.parametrize("expression", ["30 16 * *", "30 16 * * 1-5 extra", "61 16 * * 1-5", "30 16 * * 8",
                                        "30 16 * * mon-", "30 16 * * 5-1", "30 16 * * L", "30 16 * * 1/0"])
def test_an_invalid_expression_is_refused(expression):
    from nasdaq_agent.cron import cron_trigger
    with pytest.raises(ValueError):
        cron_trigger(expression, "America/New_York")


def test_a_schedule_that_never_fires_is_refused():
    from nasdaq_agent.cron import cron_trigger, next_fire_time
    with pytest.raises(ValueError, match="never"):
        next_fire_time(cron_trigger("0 0 30 2 *", "UTC"), datetime(2026, 9, 28, tzinfo=timezone.utc))


def test_the_schedule_keeps_new_york_wall_time_across_the_daylight_saving_change():
    # Daylight saving ends on 2026-11-01: 16:30 New York is 20:30 UTC before and 21:30 UTC after.
    from nasdaq_agent.cron import cron_trigger, next_fire_time
    trigger = cron_trigger("30 16 * * 1-5", "America/New_York")
    friday = next_fire_time(trigger, ny(2026, 10, 30, 12, 0))
    monday = next_fire_time(trigger, friday + timedelta(seconds=1))
    assert friday.astimezone(timezone.utc).strftime("%m-%d %H:%M") == "10-30 20:30"
    assert monday.astimezone(timezone.utc).strftime("%m-%d %H:%M") == "11-02 21:30"


# The loop. A fake clock advances when the loop sleeps, so no test waits.

class FakeClock:
    def __init__(self, start: datetime):
        self.now = start.astimezone(timezone.utc)
        self.naps: list[float] = []

    def __call__(self) -> datetime:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.naps.append(seconds)
        self.now += timedelta(seconds=seconds)


def _settings(tmp_path, **over):
    from nasdaq_agent.config import Settings
    return Settings(_env_file=None, email_to="r@example.com", artifacts_dir=tmp_path, **over)


def _outcomes(lines):
    """The trigger lines, without the next-run lines printed before each wait."""
    return [line for line in lines if "scheduled_for" in line]


def _agent(clock, exit_codes, tmp_path):
    """A stand-in for run_once that records when it ran and returns the next exit code."""
    from nasdaq_agent.agent.graph import RunOutcome
    codes = iter(exit_codes)
    runs = []

    def run_agent(settings):
        runs.append(clock().astimezone(NY))
        return RunOutcome(run_id=f"run-{len(runs)}", exit_code=next(codes), artifacts_path=str(tmp_path))
    return run_agent, runs


def test_the_loop_waits_for_the_next_trigger_and_runs_the_agent(tmp_path):
    from nasdaq_agent.scheduler import run_on_schedule
    clock = FakeClock(ny(2026, 9, 25, 17, 0))  # Friday, after that session's report time
    run_agent, runs = _agent(clock, [0], tmp_path)
    lines = []
    run_on_schedule(_settings(tmp_path), run_agent, clock=clock, sleep=clock.sleep, max_triggers=1, report=lines.append)
    assert runs == [ny(2026, 9, 28, 16, 30)]  # the weekend passes without a run
    assert lines == [{"next_run": "2026-09-28T16:30:00-04:00"},
                     {"scheduled_for": "2026-09-28T16:30:00-04:00", "session": "2026-09-28", "run_id": "run-1",
                      "exit_code": 0}]


def test_the_loop_sleeps_in_short_naps_so_a_suspended_machine_or_clock_change_is_noticed(tmp_path):
    from nasdaq_agent.scheduler import POLL_SECONDS, run_on_schedule
    clock = FakeClock(ny(2026, 9, 25, 17, 0))
    run_agent, _ = _agent(clock, [0], tmp_path)
    run_on_schedule(_settings(tmp_path), run_agent, clock=clock, sleep=clock.sleep, max_triggers=1, report=lambda _: None)
    assert clock.naps and max(clock.naps) <= POLL_SECONDS


def test_a_trigger_whose_session_was_already_reported_is_skipped(tmp_path):
    # 2026-11-26 is Thanksgiving: at 16:30 that day the last completed session is still the 25th, reported the evening
    # before. Running again would only send the same report twice.
    from nasdaq_agent.scheduler import run_on_schedule
    clock = FakeClock(ny(2026, 11, 25, 12, 0))
    run_agent, runs = _agent(clock, [0], tmp_path)
    lines = []
    run_on_schedule(_settings(tmp_path), run_agent, clock=clock, sleep=clock.sleep, max_triggers=2, report=lines.append)
    assert runs == [ny(2026, 11, 25, 16, 30)]
    assert _outcomes(lines)[1] == {"scheduled_for": "2026-11-26T16:30:00-05:00", "session": "2026-11-25",
                                   "skipped": "this session was already reported"}


def test_a_failed_run_is_not_recorded_so_a_later_trigger_retries_its_session(tmp_path):
    from nasdaq_agent.scheduler import run_on_schedule
    clock = FakeClock(ny(2026, 11, 25, 12, 0))
    run_agent, runs = _agent(clock, [1, 0], tmp_path)
    run_on_schedule(_settings(tmp_path), run_agent, clock=clock, sleep=clock.sleep, max_triggers=2, report=lambda _: None)
    assert runs == [ny(2026, 11, 25, 16, 30), ny(2026, 11, 26, 16, 30)]


def test_a_degraded_run_counts_as_reported(tmp_path):
    # Exit 2 means the report went out with problems noted.
    from nasdaq_agent.scheduler import run_on_schedule
    clock = FakeClock(ny(2026, 11, 25, 12, 0))
    run_agent, runs = _agent(clock, [2], tmp_path)
    run_on_schedule(_settings(tmp_path), run_agent, clock=clock, sleep=clock.sleep, max_triggers=2, report=lambda _: None)
    assert len(runs) == 1


def test_the_reported_session_survives_a_restart(tmp_path):
    from nasdaq_agent.scheduler import STATE_FILENAME, run_on_schedule
    clock = FakeClock(ny(2026, 11, 25, 12, 0))
    run_agent, runs = _agent(clock, [0, 0], tmp_path)
    run_on_schedule(_settings(tmp_path), run_agent, clock=clock, sleep=clock.sleep, max_triggers=1, report=lambda _: None)
    assert json.loads((tmp_path / STATE_FILENAME).read_text()) == {"last_reported_session": "2026-11-25"}
    lines = []
    run_on_schedule(_settings(tmp_path), run_agent, clock=clock, sleep=clock.sleep, max_triggers=1, report=lines.append)
    assert len(runs) == 1 and "skipped" in _outcomes(lines)[0]


def test_a_crashing_run_is_logged_without_secrets_and_the_schedule_continues(tmp_path, caplog):
    from pydantic import SecretStr
    from nasdaq_agent.scheduler import run_on_schedule
    clock = FakeClock(ny(2026, 9, 28, 12, 0))
    calls = []

    def run_agent(settings):
        calls.append(clock())
        raise RuntimeError("provider rejected key sk-live-secret")
    lines = []
    settings = _settings(tmp_path, smtp_host="smtp.example.com", smtp_port=587, smtp_username="u",
                         smtp_password=SecretStr("sk-live-secret"), smtp_from="a@example.com")
    with caplog.at_level(logging.ERROR, logger="nasdaq_agent.scheduler"):
        run_on_schedule(settings, run_agent, clock=clock, sleep=clock.sleep, max_triggers=2, report=lines.append)
    first = _outcomes(lines)[0]
    assert len(calls) == 2 and first["error"] == "RuntimeError" and "run_id" not in first
    assert "sk-live-secret" not in caplog.text and "RuntimeError" in caplog.text


def test_an_unreadable_state_file_is_treated_as_empty_with_a_warning(tmp_path, caplog):
    from nasdaq_agent.scheduler import STATE_FILENAME, run_on_schedule
    (tmp_path / STATE_FILENAME).write_text("{not json")
    clock = FakeClock(ny(2026, 9, 28, 12, 0))
    run_agent, runs = _agent(clock, [0], tmp_path)
    with caplog.at_level(logging.WARNING, logger="nasdaq_agent.scheduler"):
        run_on_schedule(_settings(tmp_path), run_agent, clock=clock, sleep=clock.sleep, max_triggers=1, report=lambda _: None)
    assert len(runs) == 1 and "schedule state" in caplog.text


def test_the_schedule_command_runs_live_only(tmp_path, monkeypatch, capsys):
    from nasdaq_agent import cli
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AGENT_EMAIL_TO", "r@example.com")
    monkeypatch.setenv("AGENT_MODE", "replay")
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["schedule"])
    assert exit_info.value.code == 1 and "schedule runs the agent live" in capsys.readouterr().err
