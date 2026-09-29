import json

from pydantic import BaseModel, ConfigDict, ValidationError

from ..metrics import AnalysisResult

SENTINEL = "__RESULT__"
# This cap slices characters, not bytes, so the name says characters.
STDOUT_CAP_CHARS = 65536
TAIL_CHARS = 2000
SANDBOX_INPUT_FILES = ("ticker.csv", "benchmark.csv", "meta.json")


class ContractError(ValueError):
    """The sandbox output did not satisfy the result contract."""


def _validation_summary(error: ValidationError) -> str:
    """Field and problem for each invalid part of the result. Built from errors(include_url=False)
    because this text reaches the model, and pydantic's documentation URL is version-stamped, so a
    dependency upgrade would otherwise change the next prompt and miss the replay cache."""
    return "; ".join(f"{'.'.join(str(part) for part in detail['loc']) or 'result'}: {detail['msg']}"
                     for detail in error.errors(include_url=False))


class RunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    exit_code: int
    backend: str
    wall_time_s: float
    timed_out: bool = False
    stdout_tail: str = ""
    stderr_tail: str = ""
    result: AnalysisResult | None = None
    error: str | None = None


def tail(text: str, chars: int = TAIL_CHARS) -> str:
    return text[-chars:]


def parse_sentinel_output(stdout: str) -> AnalysisResult:
    """Take the last sentinel line of stdout and validate it against the result model."""
    lines = [ln for ln in stdout.splitlines() if ln.startswith(SENTINEL)]
    if not lines:
        raise ContractError(f"no line starting with {SENTINEL} in stdout; assign a dict named result")
    payload = lines[-1][len(SENTINEL):]
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as e:
        raise ContractError(f"result line is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise ContractError("result must be a JSON object")
    try:
        return AnalysisResult(**data)
    except ValidationError as e:
        raise ContractError(f"result does not match the contract: {_validation_summary(e)}") from e
    except TypeError as e:
        raise ContractError(f"result does not match the contract: {e}") from e
