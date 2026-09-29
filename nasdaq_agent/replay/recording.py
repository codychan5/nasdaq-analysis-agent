"""The record command's cassette lifecycle: staging a recording beside the cassette under a lock, the content guard,
and sealing -- the secret scan, the manifest, then the swap that replaces the previous cassette.

A recording is written into a staging directory beside the cassette, under an exclusive lock. It is sealed -- manifest
written, then swapped in -- only after a run that finished with exit 0 or 2, with no judge failure, and with no
configured secret anywhere in it, since cassettes are committed. On any failure the staging directory is removed and
the previous cassette is left exactly as it was.

Nothing that is not a cassette is ever moved or deleted. The cassette path is resolved before anything else. An
existing directory is accepted only when its contents are recognisably a cassette (check_cassette_directory); each
entry is checked again immediately before it is moved aside, and again before the moved-aside copy is deleted; and
deletion itself removes only verified entry files and then empty directories. So even a bypassed guard cannot delete
anything else.
"""
import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote, quote_plus

from ..artifacts import RUN_ID_PATTERN
from .files import ENTRY_SUFFIX, SAFE_ENTRY_NAME
from .http_cassette import HTTP_DIR
from .llm_cache import LLM_DIR, MANIFEST, write_manifest
from .proxies import SOURCES_DIR

log = logging.getLogger("nasdaq_agent.replay")

RECORDING_DIRS = (HTTP_DIR, LLM_DIR, SOURCES_DIR)
# The cassette's own entries, in the order a new recording is swapped in: entries first, the manifest -- the thing
# replay checks -- last, so a cassette is never valid while half swapped.
CASSETTE_ENTRIES = (*RECORDING_DIRS, MANIFEST)
# What makes manifest.json a cassette's manifest rather than, say, a web app's.
MANIFEST_REQUIRED_KEYS = ("fingerprint", "now")
STAGING_INFIX = ".recording-"
BACKUP_SUFFIX = ".previous"
LOCK_SUFFIX = ".lock"
# The names this module creates beside a cassette -- staging, backup and lock -- accepted inside a cassette directory
# too. Accepting a name never makes it deletable: only CASSETTE_ENTRIES are ever moved or removed.
OWN_NAME = re.compile(rf"\.[^/]+(?:{re.escape(STAGING_INFIX)}{RUN_ID_PATTERN.pattern}(?:{re.escape(BACKUP_SUFFIX)})?"
                      rf"|{re.escape(LOCK_SUFFIX)})")
PERCENT_ESCAPE = re.compile(r"%[0-9A-F]{2}")


class RecordingNotSealable(RuntimeError):
    """The recording cannot become the cassette. The message says why and never contains a secret."""


class CassetteSecretLeak(RecordingNotSealable):
    """A configured secret value was found in the recording. The message names files by path only."""


class CassetteDirectoryRefused(ValueError):
    """AGENT_CASSETTE_DIR names something that is not, and is not about to become, a cassette directory."""


class RecordingInProgress(RuntimeError):
    """Another recording into the same cassette holds the lock."""


def _resolved(final: Path) -> Path:
    return Path(final).resolve()


def staging_dir(final: Path, run_id: str) -> Path:
    """This run's staging directory, beside the resolved cassette directory -- never inside it, whatever the path
    spells (a symlink, a "..")."""
    final = _resolved(final)
    return final.parent / f".{final.name}{STAGING_INFIX}{run_id}"


def lock_path(final: Path) -> Path:
    final = _resolved(final)
    return final.parent / f".{final.name}{LOCK_SUFFIX}"


def _backup_dir(staging: Path) -> Path:
    """Where the previous cassette's entries wait during the swap: beside the staging directory, named after it."""
    return staging.with_name(staging.name + BACKUP_SUFFIX)


def _entry_file_problem(path: Path, label: str) -> str | None:
    if path.is_symlink():
        return f"{label} is a symlink"
    if not path.is_file():
        return f"{label} is not a regular file"
    if path.suffix != ENTRY_SUFFIX or not SAFE_ENTRY_NAME.fullmatch(path.stem):
        return f"{label} is not a cassette entry file"
    return None


def _entries_dir_problem(directory: Path, label: str) -> str | None:
    if directory.is_symlink():
        return f"{label} is a symlink"
    if not directory.is_dir():
        return f"{label} is not a directory"
    for child in sorted(directory.iterdir()):
        problem = _entry_file_problem(child, f"{label}/{child.name}")
        if problem:
            return problem
    return None


def _manifest_problem(path: Path, label: str) -> str | None:
    if path.is_symlink():
        return f"{label} is a symlink"
    if not path.is_file():
        return f"{label} is not a regular file"
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return f"{label} is not JSON"
    if not isinstance(data, dict) or any(key not in data for key in MANIFEST_REQUIRED_KEYS):
        return f"{label} is not a cassette manifest (it has no {' and '.join(MANIFEST_REQUIRED_KEYS)})"
    return None


def _cassette_entry_problem(entry: Path) -> str | None:
    """None when entry is one of the cassette's entries and holds only what a cassette holds."""
    if entry.name == MANIFEST:
        return _manifest_problem(entry, entry.name)
    if entry.name in RECORDING_DIRS:
        return _entries_dir_problem(entry, entry.name)
    return f"{entry.name} is not part of a cassette"


def _contents_problem(directory: Path, *, allow_own_names: bool) -> str | None:
    """The first entry, in name order, that makes directory something other than a cassette; None when it is one."""
    for entry in sorted(directory.iterdir()):
        if allow_own_names and OWN_NAME.fullmatch(entry.name):
            continue
        problem = _cassette_entry_problem(entry)
        if problem:
            return problem
    return None


def check_cassette_directory(final: Path) -> Path:
    """The content guard: returns the resolved cassette path, or raises CassetteDirectoryRefused naming the first
    offending entry. A missing or empty directory is accepted, and so is a leftover cassette -- entry files but no
    manifest. Anything else is refused before any change: a name-only check would let a package directory holding a
    sources/ subpackage, or a web app's manifest.json, be swapped out and deleted."""
    final = _resolved(final)
    if not final.exists():
        return final
    if not final.is_dir():
        raise CassetteDirectoryRefused(f"refusing to record into {final}: it is not a directory")
    problem = _contents_problem(final, allow_own_names=True)
    if problem:
        raise CassetteDirectoryRefused(f"refusing to record into {final}: {problem}. Nothing was changed; point "
                                       "AGENT_CASSETTE_DIR at an empty directory or an existing cassette")
    return final


def start_recording(final: Path, run_id: str) -> Path:
    """Checks the cassette directory and creates this run's empty staging directory beside it."""
    final = check_cassette_directory(final)
    staging = staging_dir(final, run_id)
    staging.mkdir(parents=True)  # never exist_ok: a staging directory is always this run's own
    return staging


def acquire_recording_lock(final: Path) -> Path:
    """One recording at a time per cassette: an exclusive lock file, created with O_EXCL, beside the resolved cassette
    directory. The caller holds it for the whole recording and releases it in a finally."""
    lock = lock_path(final)
    lock.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        holder = lock.read_text(errors="replace").strip() if lock.is_file() else "unknown"
        raise RecordingInProgress(f"another recording into {_resolved(final)} is in progress (lock {lock}, held by "
                                  f"{holder}); if no recording is running -- a killed process leaves its lock behind "
                                  f"-- delete {lock} and record again") from None
    with os.fdopen(descriptor, "w") as handle:
        json.dump({"pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat()}, handle)
    return lock


def release_recording_lock(lock: Path) -> None:
    Path(lock).unlink(missing_ok=True)


def _delete_cassette_entries(directory: Path) -> None:
    """Removes a staging or backup directory by its cassette entries only: each recording directory's verified entry
    files, then that directory once empty; a verified cassette manifest; then directory itself once empty. Nothing
    that fails verification is removed, so a directory holding anything else raises OSError and keeps it."""
    for name in RECORDING_DIRS:
        subdir = directory / name
        if subdir.is_symlink() or not subdir.is_dir():
            continue
        for child in sorted(subdir.iterdir()):
            if _entry_file_problem(child, child.name) is None:
                child.unlink()
        subdir.rmdir()
    manifest = directory / MANIFEST
    if manifest.exists() and _manifest_problem(manifest, MANIFEST) is None:
        manifest.unlink()
    directory.rmdir()


def discard_recording(staging: Path) -> None:
    """Removes this run's staging directory if it is still there. Never raises: whatever cannot be removed -- which by
    construction is not a cassette entry -- is left in place with a warning."""
    staging = Path(staging)
    if not staging.exists():
        return
    try:
        _delete_cassette_entries(staging)
    except OSError as e:
        log.warning("the staging directory %s could not be removed: %s", staging, e)


def _encoded_forms(secret: str) -> set[str]:
    """How a secret could appear inside a cassette file: raw or with its slashes escaped as \\/; JSON-escaped once (a
    string in our own JSON) or twice (a recorded JSON body stored inside our JSON), with or without non-ASCII
    escaping, each with or without escaped slashes; or percent-encoded -- quote() with "/" left raw as well as fully
    encoded, quote_plus() with "+" for spaces -- with upper- or lower-case escapes."""
    forms = {secret, secret.replace("/", "\\/")}
    for ensure_ascii in (True, False):
        for base in (secret, secret.replace("/", "\\/")):
            once = json.dumps(base, ensure_ascii=ensure_ascii)[1:-1]
            for level_one in (once, once.replace("/", "\\/")):
                forms.update({level_one, json.dumps(level_one, ensure_ascii=ensure_ascii)[1:-1]})
    for encoded in (quote(secret), quote(secret, safe=""), quote_plus(secret, safe="")):
        forms.update({encoded, PERCENT_ESCAPE.sub(lambda escape: escape.group(0).lower(), encoded)})
    return {form for form in forms if form}


def find_secret_leaks(root: Path, secrets: Iterable[str]) -> list[str]:
    """Paths, relative to root, of every file holding any configured secret value."""
    root = Path(root)
    needles = set().union(*(_encoded_forms(secret) for secret in secrets if secret))
    if not needles or not root.exists():
        return []
    leaks = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        text = path.read_bytes().decode("utf-8", errors="replace")
        if any(needle in text for needle in needles):
            leaks.append(path.relative_to(root).as_posix())
    return leaks


def _move_back(source: Path, target: Path) -> None:
    try:
        os.replace(source, target)
    except OSError as e:
        log.error("rolling back the cassette swap: could not move %s back to %s: %s", source, target, e)


def _roll_back(staging: Path, final: Path, backup: Path, promoted: list[str], retired: list[str]) -> None:
    """Puts the previous cassette back: the new entries return to staging, the new manifest first, then the previous
    entries return from the backup, their manifest last. Best effort: a step that fails is logged and the rest run."""
    for name in reversed(promoted):
        _move_back(final / name, staging / name)
    for name in reversed(retired):
        _move_back(backup / name, final / name)
    try:
        backup.rmdir()
    except OSError as e:
        log.error("rolling back the cassette swap: the backup %s is not empty and was kept: %s", backup, e)


def _promote(staging: Path, final: Path) -> None:
    """Swaps the staged recording in for the previous cassette's entries.

    Before anything moves, the whole directory passes the content guard again; each previous entry is re-verified
    immediately before it is moved aside, and the moved-aside backup once more immediately before it is deleted. A
    failed check or a failed move rolls everything back and raises. Once the new cassette is live, trouble removing the
    backup or the staging directory is only a warning."""
    final = check_cassette_directory(final)
    final.mkdir(parents=True, exist_ok=True)
    backup = _backup_dir(staging)
    backup.mkdir()
    retired: list[str] = []
    promoted: list[str] = []
    try:
        for name in (MANIFEST, *RECORDING_DIRS):  # the previous manifest leaves first
            entry = final / name
            if not (entry.exists() or entry.is_symlink()):
                continue
            problem = _cassette_entry_problem(entry)
            if problem:
                raise CassetteDirectoryRefused(f"refusing to replace the cassette at {final}: {problem}; "
                                               "nothing was changed")
            os.replace(entry, backup / name)
            retired.append(name)
        for name in CASSETTE_ENTRIES:  # the new manifest arrives last
            if (staging / name).exists():
                os.replace(staging / name, final / name)
                promoted.append(name)
        problem = _contents_problem(backup, allow_own_names=False)
        if problem:
            raise CassetteDirectoryRefused(f"refusing to delete the previous cassette's entries from {backup}: "
                                           f"{problem}; the previous cassette was put back")
    except BaseException:
        _roll_back(staging, final, backup, promoted, retired)
        raise
    for directory, what in ((backup, "the previous cassette's backup"), (staging, "the staging directory")):
        try:
            _delete_cassette_entries(directory)
        except Exception as e:  # noqa: BLE001  the swap is done; cleanup must not turn a sealed recording into a failure
            log.warning("the new cassette at %s is in place, but %s %s could not be removed: %s: %s",
                        final, what, directory, type(e).__name__, e)


def seal_recording(staging: Path, final: Path, now_iso: str, secrets: Iterable[str], keyed_sources: Iterable[str] = (),
                   environment: dict[str, Any] | None = None, exit_code: int | None = None) -> None:
    """Scans the staged recording for secrets, writes its manifest and swaps it in. Raises RecordingNotSealable (or
    CassetteSecretLeak, or CassetteDirectoryRefused) before anything reaches the cassette; the caller removes the
    staging directory."""
    staging, final = Path(staging), _resolved(final)
    leaks = find_secret_leaks(staging, secrets)
    if leaks:
        raise CassetteSecretLeak(f"a configured secret value appears in {', '.join(leaks)}; the recording was discarded "
                                 f"and the cassette at {final} left as it was -- record again once the source of the "
                                 "leak is fixed")
    write_manifest(staging, now_iso, keyed_sources, environment, exit_code)
    _promote(staging, final)
