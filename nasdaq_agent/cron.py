"""Standard five-field crontab expressions for the scheduled trigger, with fire times computed by APScheduler.

APScheduler's 3.x CronTrigger handles time zones and daylight-saving changes, but it differs from crontab in two ways
that would silently change when the agent runs, so this module does not hand it the expression as it stands:
  - It numbers weekdays from Monday (0 = Monday), where crontab numbers them from Sunday (0 or 7 = Sunday): read raw,
    "1-5" would run Tuesday to Saturday. The day-of-week field is therefore read here, with crontab numbering, and
    passed on as day names.
  - When both the day of the month and the day of the week are restricted, crontab runs when either matches and
    APScheduler only when both do. Such an expression is refused rather than run with the other meaning.
"""
import re
from datetime import datetime
from zoneinfo import ZoneInfo

from apscheduler.triggers.cron import CronTrigger

CRON_FIELDS = 5
# Crontab weekday numbers: the index is the number, and 7 is Sunday again.
CRONTAB_WEEKDAYS = ("sun", "mon", "tue", "wed", "thu", "fri", "sat")
CRONTAB_SUNDAY_AGAIN = 7
WEEKDAY_PART_PATTERN = re.compile(r"^(?P<base>\*|[0-9a-z]+(?:-[0-9a-z]+)?)(?:/(?P<step>\d+))?$")
ANY = "*"


def _weekday_number(token: str) -> int:
    if token.isdigit() and 0 <= int(token) <= CRONTAB_SUNDAY_AGAIN:
        return int(token)
    if token in CRONTAB_WEEKDAYS:
        return CRONTAB_WEEKDAYS.index(token)
    raise ValueError(f"day of week {token!r} is not 0-7 or sun-sat")


def crontab_weekdays(field: str) -> set[int]:
    """The weekdays a crontab day-of-week field names, as crontab numbers (0 = Sunday). Accepts numbers 0-7 and names,
    lists, ranges and steps: "1-5", "mon-fri", "0,6", "*/2", "1-5/2"."""
    days: set[int] = set()
    for part in field.lower().split(","):
        match = WEEKDAY_PART_PATTERN.match(part)
        if match is None:
            raise ValueError(f"day of week {part!r} is not a crontab value, range or step")
        base, step_text = match["base"], match["step"]
        step = int(step_text) if step_text else 1
        if step < 1:
            raise ValueError(f"day-of-week step in {part!r} must be at least 1")
        if base == ANY:
            first, last = 0, len(CRONTAB_WEEKDAYS) - 1
        elif "-" in base:
            first, last = (_weekday_number(token) for token in base.split("-"))
            if first > last:
                raise ValueError(f"day-of-week range {base!r} runs backwards")
        else:
            first = _weekday_number(base)
            # "5/2" runs from Friday to the end of the week, as "5-6/2" would.
            last = len(CRONTAB_WEEKDAYS) - 1 if step_text else first
        days.update(number % CRONTAB_SUNDAY_AGAIN for number in range(first, last + 1, step))
    return days


def cron_trigger(expression: str, timezone_name: str) -> CronTrigger:
    """A trigger that fires when the crontab expression does, read in the named time zone. Raises ValueError for an
    invalid expression, and ZoneInfoNotFoundError (a KeyError) for an unknown time zone."""
    fields = expression.split()
    if len(fields) != CRON_FIELDS:
        raise ValueError(f"a cron expression has {CRON_FIELDS} fields (minute hour day month weekday), not {len(fields)}")
    minute, hour, day, month, weekday = fields
    if day != ANY and weekday != ANY:
        raise ValueError("restrict the day of the month or the day of the week, not both: crontab runs when either "
                         "matches, and this scheduler would need both")
    day_of_week = ANY if weekday == ANY else ",".join(CRONTAB_WEEKDAYS[n] for n in sorted(crontab_weekdays(weekday)))
    return CronTrigger(minute=minute, hour=hour, day=day, month=month, day_of_week=day_of_week,
                       timezone=ZoneInfo(timezone_name))


def next_fire_time(trigger: CronTrigger, after: datetime) -> datetime:
    """The first fire time at or after `after`. Raises ValueError for a schedule that never fires, such as 30 February."""
    fire = trigger.get_next_fire_time(None, after)
    if fire is None:
        raise ValueError("the schedule never fires")
    return fire
