import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from ...artifacts import RunDir
from ...config import Settings
from ...email.transports import Transport
from ...report.judge import JudgeFn
from ...sandbox.runner import CodeRunner
from ...sources.base import GainerSource, HistorySource, NewsSource
from ...universe import SessionUniverse
from ..context import RunContext

log = logging.getLogger("nasdaq_agent.tools")
# Decimal places kept in floats a tool shows the model. numpy builds can differ in the last digit across platforms,
# which would change the next prompt and miss the replay cache.
MODEL_VISIBLE_DECIMALS = 6


@dataclass
class Deps:
    settings: Settings
    universe: SessionUniverse
    gainer_sources: list[GainerSource]
    history_sources: list[HistorySource]
    news_sources: list[NewsSource]
    runner: CodeRunner | None
    transport: Transport | None
    judge: JudgeFn | None
    clock: Callable[[], datetime]
    run_dir: RunDir


def err(message: str) -> str:
    return f"ERROR: {message}"


def precondition(ctx: RunContext, flag: str, tool_name: str, needed: str) -> str | None:
    """Return an error string when the flag is not set. Tools call this first, whatever the model was shown."""
    if not getattr(ctx.progress, flag):
        return err(f"precondition for {tool_name} not met: {needed}")
    return None


def record_call(ctx: RunContext, deps: Deps, tool_name: str, args: dict[str, Any], outcome: str) -> None:
    ctx.tool_calls += 1
    ctx.last_tool = tool_name
    deps.run_dir.append_jsonl("tool_log.jsonl", {"n": ctx.tool_calls, "tool": tool_name, "args": args,
                                                  "outcome": outcome[:500], "at": deps.clock().isoformat()})
    ctx.save()
    log.info("tool %s: %s", tool_name, outcome[:200], extra={"tool": tool_name, "run_id": ctx.run_id})


def ok(payload: dict[str, Any]) -> str:
    return json.dumps(payload, default=str)


def round_floats(value: Any, places: int = MODEL_VISIBLE_DECIMALS) -> Any:
    """A copy of value with every finite float rounded, for a model-visible result; internal values are never passed
    through this. Negative zero becomes zero, so a value near zero cannot print as -0.0 on one machine and 0.0 on
    another."""
    if isinstance(value, float):
        if not math.isfinite(value):
            return value
        rounded = round(value, places)
        return 0.0 if rounded == 0 else rounded
    if isinstance(value, dict):
        return {key: round_floats(item, places) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [round_floats(item, places) for item in value]
    return value
