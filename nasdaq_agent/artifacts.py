import hashlib
import json
import logging
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

HASH_CHARS = 16
REDACTED = "***"
SENSITIVE_KEYS = {"password", "api_key", "apikey", "secret", "token", "authorization", "recipient", "to"}
RUN_ID_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"
RUN_ID_SUFFIX_CHARS = 6
# Exactly what new_run_id produces. Matched with fullmatch: "$" would also accept a trailing newline. [0-9], not \d,
# which also matches every other script's decimal digits (fix round 1, minor 1).
RUN_ID_PATTERN = re.compile(rf"[0-9]{{8}}T[0-9]{{6}}Z-[0-9a-f]{{{RUN_ID_SUFFIX_CHARS}}}")


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime(RUN_ID_TIMESTAMP_FORMAT) + "-" + uuid.uuid4().hex[:RUN_ID_SUFFIX_CHARS]


def validate_run_id(run_id: str) -> str:
    """Task 23 correction h: a run id typed on the command line becomes a directory name under the artefacts root, so
    anything but new_run_id's own format (no separators, no "..") is refused before it reaches the filesystem."""
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(f"invalid run id {run_id!r}: expected the form YYYYMMDDTHHMMSSZ-xxxxxx that new runs print")
    return run_id


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:HASH_CHARS]


def redact(text: str, secrets: Iterable[str]) -> str:
    """Replace each non-empty secret with REDACTED. Skips empty/falsy secrets: replacing "" would
    insert REDACTED between every character of `text`. Used both by SecretRedactingFilter (log
    records) and directly by finalize.py/graph.py (failure notices, summary.json, ctx.errors) --
    one implementation for every place a secret must never reach untouched."""
    for secret in secrets:
        if secret:
            text = text.replace(secret, REDACTED)
    return text


class RunDir:
    def __init__(self, root: Path, run_id: str):
        self.root, self.run_id = Path(root), run_id
        self.path = self.root / run_id
        self.path.mkdir(parents=True, exist_ok=True)

    @property
    def sandbox_dir(self) -> Path:
        return self.path / "sandbox"

    @property
    def outbox_dir(self) -> Path:
        return self.path / "outbox"

    @property
    def attempts_dir(self) -> Path:
        return self.path / "attempts"

    def write_json(self, name: str, obj: Any) -> Path:
        p = self.path / name
        p.write_text(json.dumps(obj, indent=2, default=str))
        return p

    def read_json(self, name: str) -> Any:
        return json.loads((self.path / name).read_text())

    def write_text(self, name: str, text: str) -> Path:
        p = self.path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        return p

    def append_jsonl(self, name: str, record: dict) -> None:
        with open(self.path / name, "a") as f:
            f.write(json.dumps(record, default=str) + "\n")


class SecretRedactingFilter(logging.Filter):
    def __init__(self, secrets: list[str]):
        super().__init__()
        self._secrets = [s for s in secrets if s]

    def _scrub(self, text: str) -> str:
        return redact(text, self._secrets)

    def filter(self, record: logging.LogRecord) -> bool:
        # Controller correction 1: format the record (substitute any %-style args) before
        # scrubbing, then clear args. Scrubbing the raw format string and clearing args
        # afterwards would leave literal "%s" placeholders in every formatted line instead
        # of the redacted, substituted message.
        message = record.getMessage()
        record.msg = self._scrub(message)
        record.args = ()
        # Review finding: log.exception(...) (used by the tool-crash handler) attaches a raw,
        # unscrubbed traceback via record.exc_info. Format it here with a throwaway
        # Formatter's formatException, scrub the resulting text, and store it on
        # record.exc_text; then clear record.exc_info so no handler/formatter downstream can
        # re-derive and re-format the raw, unscrubbed exception from it.
        if record.exc_info:
            record.exc_text = self._scrub(logging.Formatter().formatException(record.exc_info))
            record.exc_info = None
        for key in list(record.__dict__):
            if key.lower() in SENSITIVE_KEYS:
                record.__dict__[key] = REDACTED
        return True


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": datetime.now(timezone.utc).isoformat(), "level": record.levelname,
                   "logger": record.name, "msg": record.getMessage()}
        if record.exc_text:
            payload["exc"] = record.exc_text
        for key in ("tool", "run_id", "event"):
            if key in record.__dict__:
                payload[key] = record.__dict__[key]
        return json.dumps(payload, default=str)


def configure_logging(run_dir: RunDir, secrets: list[str]) -> logging.Logger:
    logger = logging.getLogger("nasdaq_agent")
    logger.setLevel(logging.INFO)
    # Controller correction 2: close and remove existing handlers before adding new ones,
    # so repeated calls in the same process (e.g. successive runs) do not leak open log
    # file descriptors.
    for h in list(logger.handlers):
        h.close()
        logger.removeHandler(h)
    redact = SecretRedactingFilter(secrets)
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir.path / "log.jsonl")):
        handler.setFormatter(_JsonFormatter())
        handler.addFilter(redact)
        logger.addHandler(handler)
    logger.propagate = False
    return logger
