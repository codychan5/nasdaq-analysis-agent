"""Recorded failures. A live call that failed while recording is stored, and fails again on replay with the same text,
so the model sees exactly what it saw when the recording was made -- typically a SourceUnavailable after retries."""
import re
from pathlib import Path
from typing import Any, NoReturn

from ..sources.errors import (CassetteMiss, SourceError, SourceNoData, SourceNotAllowed, SourceQuotaExceeded,
                              SourceUnavailable)

FAILURE_KEY = "failure"
# Rebuilt as their own class, looked up by name in this fixed table: a cassette never chooses a class to instantiate.
# SourceNoData must keep its class, because find_top_gainer treats "no bars" and "no answer" differently.
REBUILT_FAILURES: dict[str, type[SourceError]] = {cls.__name__: cls for cls in (
    SourceError, SourceUnavailable, SourceNotAllowed, SourceQuotaExceeded, SourceNoData)}
FAILURE_TYPE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,99}")


class RecordedFailure(SourceError):
    """A recorded failure of a class outside REBUILT_FAILURES (a yfinance or parsing error, say). Its text is the
    "Type: message" line the tools record for such a failure, so what the model is shown does not change."""


def failure_entry(error: Exception) -> dict[str, Any]:
    return {FAILURE_KEY: {"type": type(error).__name__, "message": str(error), "source_error": isinstance(error, SourceError)}}


def is_failure_entry(data: Any) -> bool:
    return isinstance(data, dict) and FAILURE_KEY in data


def raise_recorded_failure(data: dict[str, Any], path: Path) -> NoReturn:
    failure = data.get(FAILURE_KEY)
    type_name = failure.get("type") if isinstance(failure, dict) else None
    message = failure.get("message") if isinstance(failure, dict) else None
    source_error = failure.get("source_error") if isinstance(failure, dict) else None
    if (not isinstance(type_name, str) or not FAILURE_TYPE_NAME.fullmatch(type_name) or not isinstance(message, str)
            or not isinstance(source_error, bool)):
        raise CassetteMiss(f"cassette entry {path} holds a malformed recorded failure; run `nasdaq-agent record` again")
    rebuilt = REBUILT_FAILURES.get(type_name)
    if rebuilt is not None:
        raise rebuilt(message)
    if source_error:
        raise SourceError(message)  # a SourceError subclass outside the table: the tools only ever show str(e)
    raise RecordedFailure(f"{type_name}: {message}")
