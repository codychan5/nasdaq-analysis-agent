"""Model responses, recorded and replayed through LangChain's global LLM cache, plus the cassette manifest."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from langchain_core.caches import BaseCache
from langchain_core.globals import set_llm_cache
from langchain_core.messages import messages_from_dict, messages_to_dict
from langchain_core.outputs import ChatGeneration, Generation

from ..metrics import DEFINITIONS_TEXT
from ..sources.errors import CassetteMiss
from .environment import current_environment
from .files import REPLAY, RECORD, read_json_entry, require_cassette_mode, write_json_atomic

MANIFEST = "manifest.json"
LLM_DIR = "llm"
FINGERPRINT_CHARS = 16
# A record run is sealed only after exit 0 (clean) or 2 (degraded), so those are the only exit codes a genuine
# manifest can hold (fix round 2, item 4).
SEALABLE_EXIT_CODES = frozenset({0, 2})
# The only message type a chat model's generation holds. Anything else in a cassette is refused, not rebuilt.
AI_MESSAGE_TYPE = "ai"
# Message fields left out of the cache key. The record-then-replay round trip (tests/graph/test_record_replay.py)
# showed one leak: on a cache hit langchain-core returns the message with "total_cost": 0 added to usage_metadata, so
# from the second turn on the replayed prompt differs from the recorded one in that field alone. Message ids are
# generated per call; langchain-core 1.6.5 already nulls them in the cache prompt, and the key drops them too in
# case a later version stops. Neither field changes what the model would say.
VOLATILE_MESSAGE_FIELDS = ("usage_metadata", "id")
# Client settings left out of the model's identity in the cache key. They change how long a call may wait and how
# often it is retried, never what the model says, so changing them after recording must not break replay.
TRANSPORT_FIELDS = ("request_timeout", "timeout", "default_request_timeout", "max_retries")
LLM_STRING_SEPARATOR = "---"


def prompt_fingerprint() -> str:
    from ..agent.orchestrator import PROMPT_PATH
    from ..report.judge import JUDGE_PROMPT
    material = PROMPT_PATH.read_text() + JUDGE_PROMPT + DEFINITIONS_TEXT
    return hashlib.sha256(material.encode()).hexdigest()[:FINGERPRINT_CHARS]


def _require_aware_iso(now_iso: Any) -> str:
    try:
        parsed = datetime.fromisoformat(now_iso) if isinstance(now_iso, str) else None
    except ValueError:
        parsed = None
    if parsed is None or parsed.tzinfo is None:
        raise ValueError(f"the recording clock must be ISO 8601 with a timezone offset, got {now_iso!r}")
    return now_iso


def write_manifest(root: Path, now_iso: str, keyed_sources: Iterable[str] = (), environment: dict[str, Any] | None = None,
                   exit_code: int | None = None) -> None:
    """keyed_sources names the key-gated data sources the recording used, so replay can rebuild the same chain with
    no keys (graph._replay_settings). Fix round 1: environment is the recording environment (replay compares it,
    Important 2; this process's own when not given) and exit_code the recorded run's (replay --check, K6)."""
    if exit_code is not None and (not isinstance(exit_code, int) or isinstance(exit_code, bool)):
        raise ValueError(f"the recorded exit code must be an integer, got {exit_code!r}")
    data = {"fingerprint": prompt_fingerprint(), "now": _require_aware_iso(now_iso), "keyed_sources": sorted(keyed_sources),
            "environment": environment if environment is not None else current_environment(), "exit_code": exit_code}
    write_json_atomic(Path(root) / MANIFEST, data)


def check_manifest(root: Path) -> dict[str, Any]:
    path = Path(root) / MANIFEST
    if not path.exists():
        raise CassetteMiss(f"no cassette manifest at {path}; run `nasdaq-agent record` first")
    data = read_json_entry(path)
    if not isinstance(data, dict):
        raise CassetteMiss(f"cassette manifest {path} is malformed; run `nasdaq-agent record` again")
    if data.get("fingerprint") != prompt_fingerprint():
        raise CassetteMiss(f"prompt templates changed since the cassette at {root} was recorded; "
                           "run `nasdaq-agent record` again")
    try:
        _require_aware_iso(data.get("now"))
    except ValueError:
        raise CassetteMiss(f"cassette manifest {path} has no valid recording clock; run `nasdaq-agent record` again") from None
    keyed = data.get("keyed_sources", [])
    if not isinstance(keyed, list) or not all(isinstance(name, str) for name in keyed):
        raise CassetteMiss(f"cassette manifest {path} lists its sources malformed; run `nasdaq-agent record` again")
    environment = data.get("environment")
    if environment is not None and not (isinstance(environment, dict) and isinstance(environment.get("packages", {}), dict)):
        raise CassetteMiss(f"cassette manifest {path} records its environment malformed; run `nasdaq-agent record` again")
    return data


def recorded_exit_code(root: Path) -> int | None:
    """The exit code the recorded run ended with (K6), or None when there is no readable manifest or it records none.
    Only 0 or 2 count: a manifest is written only after those, so anything else -- a hand-edited 1, say -- reads as
    missing (fix round 2, item 4). Deliberately not check_manifest: replay --check reports the recorded code even when
    the run itself failed on a stale or broken cassette."""
    path = Path(root) / MANIFEST
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    code = data.get("exit_code") if isinstance(data, dict) else None
    sealable = isinstance(code, int) and not isinstance(code, bool) and code in SEALABLE_EXIT_CODES
    return code if sealable else None


def _to_plain(generations: Sequence[Generation]) -> dict[str, Any]:
    """Correction a: plain data only -- messages_to_dict for the message, plus the generation's text and info."""
    entries = []
    for generation in generations:
        entry: dict[str, Any] = {"text": generation.text, "generation_info": generation.generation_info}
        if isinstance(generation, ChatGeneration):
            entry["message"] = messages_to_dict([generation.message])[0]
        entries.append(entry)
    return {"generations": entries}


def _from_plain(payload: Any, path: Path) -> list[Generation]:
    """Rebuilds generations with messages_from_dict, which maps a closed set of message type names to their classes.
    Correction a: never langchain_core.load on cassette content -- a committed file is input, and generic
    deserialization of it is the serialization-injection class behind CVE-2025-68664."""
    malformed = CassetteMiss(f"cassette entry {path} is malformed; run `nasdaq-agent record` again")
    entries = payload.get("generations") if isinstance(payload, dict) else None
    if not isinstance(entries, list):
        raise malformed
    generations: list[Generation] = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise malformed
        info, message = entry.get("generation_info"), entry.get("message")
        if info is not None and not isinstance(info, dict):
            raise malformed
        if message is None:
            if not isinstance(entry.get("text"), str):
                raise malformed
            generations.append(Generation(text=entry["text"], generation_info=info))
            continue
        if not isinstance(message, dict) or message.get("type") != AI_MESSAGE_TYPE:
            raise malformed
        try:
            (rebuilt,) = messages_from_dict([message])
        except (KeyError, TypeError, ValueError):
            raise malformed from None
        generations.append(ChatGeneration(message=rebuilt, generation_info=info))
    return generations


def _key_material(prompt: str) -> str:
    """The prompt (LangChain's serialized message list) without VOLATILE_MESSAGE_FIELDS, keys sorted."""
    try:
        messages = json.loads(prompt)
    except ValueError:
        return prompt
    if not isinstance(messages, list):
        return prompt
    for message in messages:
        fields = message.get("kwargs") if isinstance(message, dict) else None
        if isinstance(fields, dict):
            for name in VOLATILE_MESSAGE_FIELDS:
                fields.pop(name, None)
    return json.dumps(messages, sort_keys=True, separators=(",", ":"))


def _model_key_material(llm_string: str) -> str:
    """LangChain's model identity without TRANSPORT_FIELDS. For a serializable chat model it is the constructor as
    JSON, then LLM_STRING_SEPARATOR and the call parameters; anything else is returned unchanged."""
    constructor, separator, call_params = llm_string.partition(LLM_STRING_SEPARATOR)
    try:
        serialized = json.loads(constructor)
    except ValueError:
        return llm_string
    fields = serialized.get("kwargs") if isinstance(serialized, dict) else None
    if not isinstance(fields, dict):
        return llm_string
    for name in TRANSPORT_FIELDS:
        fields.pop(name, None)
    return json.dumps(serialized, sort_keys=True, separators=(",", ":")) + separator + call_params


class JsonFileLLMCache(BaseCache):
    """Records every model response in record mode; serves them in replay mode; a miss in replay is an error."""

    def __init__(self, root: Path, mode: str):
        self.root, self.mode = Path(root) / LLM_DIR, require_cassette_mode(mode)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, prompt: str, llm_string: str) -> Path:
        material = _key_material(prompt) + "\n" + _model_key_material(llm_string)
        return self.root / (hashlib.sha256(material.encode()).hexdigest() + ".json")

    def lookup(self, prompt: str, llm_string: str) -> Sequence[Generation] | None:
        path = self._path(prompt, llm_string)
        if path.exists():
            return _from_plain(read_json_entry(path), path)
        if self.mode == REPLAY:
            raise CassetteMiss(f"no recorded model response {path.name} in the cassette at {self.root.parent}; "
                               "run `nasdaq-agent record` again")
        return None

    def update(self, prompt: str, llm_string: str, return_val: Sequence[Generation]) -> None:
        if self.mode == RECORD:
            write_json_atomic(self._path(prompt, llm_string), _to_plain(return_val))

    def clear(self, **kwargs: Any) -> None:
        for path in self.root.glob("*.json"):
            path.unlink()


def install_llm_cache(cache: BaseCache | None) -> None:
    set_llm_cache(cache)
