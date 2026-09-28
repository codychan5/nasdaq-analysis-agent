"""File handling shared by the cassette stores. A cassette is committed to the repository and read back as input, so
entry names are checked before they become paths, writes are atomic, and unreadable content is a miss, never a crash."""
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from ..sources.errors import CassetteMiss

RECORD, REPLAY = "record", "replay"
CASSETTE_MODES = frozenset({RECORD, REPLAY})
ENTRY_SUFFIX = ".json"
# An entry name becomes a file name: no path separator and no dot, so no traversal and no hidden files.
SAFE_ENTRY_NAME = re.compile(r"[A-Za-z0-9_-]{1,128}")


def require_cassette_mode(mode: str) -> str:
    if mode not in CASSETTE_MODES:
        raise ValueError(f"cassette mode must be one of {sorted(CASSETTE_MODES)}, got {mode!r}")
    return mode


def entry_path(directory: Path, name: str) -> Path:
    if not isinstance(name, str) or not SAFE_ENTRY_NAME.fullmatch(name):
        raise ValueError(f"unsafe cassette entry name {name!r}")
    return directory / f"{name}{ENTRY_SUFFIX}"


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write through a temporary file in the same directory, then rename, so a crash never leaves a torn entry.
    Key order is preserved as given: a replayed message must serialize exactly as the recorded one did."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise


def read_json_entry(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as e:
        # Raised outside the handler's chain on purpose: the entry's content is not worth carrying in a traceback.
        error_name = type(e).__name__
    raise CassetteMiss(f"cassette entry {path} is unreadable ({error_name}); run `nasdaq-agent record` again")
