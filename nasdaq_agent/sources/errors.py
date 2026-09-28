import contextvars


class SourceError(Exception):
    """A source failed in a way that is not worth retrying."""

class SourceUnavailable(SourceError):
    """Transient failures exhausted the retry budget."""

class SourceNotAllowed(SourceError):
    """The source may not be used right now, for example a live screener while the market is open."""

class SourceQuotaExceeded(SourceError):
    """The persisted daily quota for this provider is used up."""

class SourceNoData(SourceError):
    """The source answered, and it has no data for the request. Every other SourceError means it gave no answer, so
    this one alone may count as evidence about the data, for example that a stock has no bars to check."""


# Tasks 22+23 fix round 1: while a replay runs in an environment that differs from the recording's (library versions,
# the sandbox backend), this holds one line per difference, and every CassetteMiss raised meanwhile leads with them --
# so the error blames the environment when that is the cause. The replay's run scope sets it (graph.py); it is empty
# otherwise. Leading, not trailing: run errors are cut to a few hundred characters.
REPLAY_ENVIRONMENT_DIFFERENCES: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar(
    "replay_environment_differences", default=())


class CassetteMiss(SourceError):
    """Replay mode found no recorded response for this request."""

    def __init__(self, message: str = ""):
        differences = REPLAY_ENVIRONMENT_DIFFERENCES.get()
        if differences:
            message = f"replay environment differs from the recording ({'; '.join(differences)}): {message}"
        super().__init__(message)
