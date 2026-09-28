"""HTTP responses fetched through our own client (sources/http.py), stored under their request hash -- and, since fix
round 1 (K2), requests that failed after their retries, which fail the same way on replay."""
from pathlib import Path

from ..sources.errors import CassetteMiss
from .failures import failure_entry, is_failure_entry, raise_recorded_failure
from .files import entry_path, read_json_entry, require_cassette_mode, write_json_atomic

HTTP_DIR = "http"


class HttpCassette:
    """Implements the Cassette protocol HttpClient consumes: lookup(key), store(key, status, text) and
    store_failure(key, error)."""

    def __init__(self, root: Path, mode: str):
        self.root, self.mode = Path(root) / HTTP_DIR, require_cassette_mode(mode)
        self.root.mkdir(parents=True, exist_ok=True)

    def lookup(self, key: str) -> tuple[int, str] | None:
        """The recorded (status, text), None when nothing was recorded, or the recorded failure raised again."""
        path = entry_path(self.root, key)
        if not path.exists():
            return None
        data = read_json_entry(path)
        if is_failure_entry(data):
            raise_recorded_failure(data, path)
        status = data.get("status") if isinstance(data, dict) else None
        text = data.get("text") if isinstance(data, dict) else None
        if not isinstance(status, int) or isinstance(status, bool) or not isinstance(text, str):
            raise CassetteMiss(f"cassette entry {path} is malformed; run `nasdaq-agent record` again")
        return status, text

    def store(self, key: str, status: int, text: str) -> None:
        write_json_atomic(entry_path(self.root, key), {"status": status, "text": text})

    def store_failure(self, key: str, error: Exception) -> None:
        write_json_atomic(entry_path(self.root, key), failure_entry(error))
