import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .transports import SendResult

log = logging.getLogger("nasdaq_agent.email")

# The states a marker writes: begin() creates it as "sending"; complete() and fail() move it on.
WRITTEN_STATES = frozenset({"sending", "sent", "failed"})
# Final fix wave A5. "sending": an attempt began and never recorded its outcome (the process may have died on either
# side of the actual delivery). "unknown": the marker exists but is not a JSON object with a state this module writes,
# which only a crash inside begin() -- after the exclusive create, before the content is on disk -- or a stray edit can
# leave. Either way the transport may or may not have accepted the message, so nothing may send it again.
OUTCOME_UNKNOWN_STATES = frozenset({"sending", "unknown"})
# What open() uses, so the process umask applies as it does to the run's other artefacts. begin() and every update
# create their file with it, so replacing the marker never changes its permissions.
MARKER_FILE_MODE = 0o666


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fsync_directory(directory: Path) -> None:
    """Make a new or renamed entry in `directory` durable. fsync on a file covers its content, but the entry naming
    the file lives in the directory: a power loss right after begin() must not lose the marker, or a resume would find
    no marker and could send twice. Best-effort: some filesystems reject fsync on a directory, or opening one, so an
    OSError is logged at debug level and the write stands -- the file itself is already fsynced."""
    try:
        fd = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError as e:
        log.debug("could not fsync the send marker's directory %s: %s: %s", directory, type(e).__name__, e)


class SendMarker:
    """Written as `sending` before the transport call and `sent` or `failed` after. A resume that finds `sending`, or
    a marker it cannot read (`unknown`), cannot know whether the server accepted the message and must not send again.

    Final fix wave A5: every write is durable and atomic. begin() creates the file exclusively (O_EXCL: of any number
    of racing callers exactly one wins) and fsyncs it. An update writes a temp file in the same directory, fsyncs it
    and os.replace()s it over the marker, so a reader sees the previous marker or the new one, never a torn one. Both
    then fsync the directory, best-effort, so the new entry survives a power loss too.
    """

    def __init__(self, path: Path):
        self.path = path

    @property
    def state(self) -> str:
        """"none" (no marker), "sending", "sent", "failed", or "unknown" (a marker that cannot be read). Never raises."""
        return self._load()[0]

    def read(self) -> dict:
        """The marker's fields; {} when there is no marker or it cannot be read. Never raises."""
        return self._load()[1]

    def _load(self) -> tuple[str, dict]:
        try:
            text = self.path.read_text()
        except FileNotFoundError:
            return "none", {}
        except (OSError, ValueError):  # it exists but cannot be read, or is not text (UnicodeDecodeError)
            return "unknown", {}
        try:
            data = json.loads(text)
        except ValueError:  # json.JSONDecodeError: empty, or cut short mid-write
            return "unknown", {}
        state = data.get("state") if isinstance(data, dict) else None
        if not isinstance(state, str) or state not in WRITTEN_STATES:
            return "unknown", {}
        return state, data

    def begin(self) -> bool:
        """Create the marker as "sending". False when any marker already exists, whatever its state."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, MARKER_FILE_MODE)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w") as f:
            json.dump({"state": "sending", "started_at": _now()}, f)
            f.flush()
            os.fsync(f.fileno())
        _fsync_directory(self.path.parent)
        return True

    def _update(self, **fields) -> None:
        data = {**self.read(), **fields, "updated_at": _now()}
        temp_path = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        try:
            fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, MARKER_FILE_MODE)
            with os.fdopen(fd, "w") as f:
                json.dump(data, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temp_path, self.path)  # the commit point: before it the marker is unchanged
        except BaseException:
            self._remove_temp_file(temp_path)
            raise
        _fsync_directory(self.path.parent)

    @staticmethod
    def _remove_temp_file(temp_path: Path) -> None:
        """Best-effort cleanup after a failed update; the update's own exception is what propagates."""
        try:
            temp_path.unlink(missing_ok=True)
        except OSError as e:
            log.warning("could not remove the send marker's temp file %s: %s: %s", temp_path, type(e).__name__, e)

    def complete(self, result: SendResult) -> None:
        self._update(state="sent", transport=result.transport, message_id=result.message_id, location=result.location)

    def fail(self, error: str) -> None:
        self._update(state="failed", error=error)
