"""The recording environment (Tasks 22+23 fix round 1, Important 2). Cache keys contain provider kwargs and tool JSON
schemas as the installed langchain-core and pydantic generate them, and sandbox results depend on numpy and pandas, so
the manifest records these versions and the sandbox backend, and replay names every difference it finds."""
import platform
from importlib.metadata import PackageNotFoundError, version
from typing import Any

ENVIRONMENT_PACKAGES = ("langchain-core", "langchain", "langgraph", "langchain-google-genai", "langchain-anthropic", "langchain-openai",
                        "pydantic", "numpy", "pandas")
NOT_INSTALLED = "not installed"
UNRECORDED = "the cassette does not record its environment"


def _installed(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return NOT_INSTALLED


def current_environment(sandbox_backend: str | None = None) -> dict[str, Any]:
    """This process's environment; sandbox_backend is the backend that ran (or will run) the model's code, if known."""
    return {"python": platform.python_version(), "packages": {name: _installed(name) for name in ENVIRONMENT_PACKAGES},
            "sandbox_backend": sandbox_backend}


def sandbox_backend_differences(recorded: Any, backend: str | None) -> list[str]:
    """The backend difference, if any. Nothing to compare until this run knows its backend, or when the cassette does
    not record an environment at all (environment_differences already says so)."""
    if backend is None or not isinstance(recorded, dict) or recorded.get("sandbox_backend") == backend:
        return []
    return [f"sandbox backend {recorded.get('sandbox_backend')} recorded, {backend} here"]


def environment_differences(recorded: Any, current: dict[str, Any]) -> list[str]:
    """One human-readable line per difference. The sandbox backend is compared only once `current` names one."""
    if not isinstance(recorded, dict):
        return [UNRECORDED]
    differences = []
    if recorded.get("python") != current["python"]:
        differences.append(f"python {recorded.get('python')} recorded, {current['python']} here")
    recorded_packages = recorded.get("packages") if isinstance(recorded.get("packages"), dict) else {}
    for name, installed in current["packages"].items():
        if recorded_packages.get(name) != installed:
            differences.append(f"{name} {recorded_packages.get(name, NOT_INSTALLED)} recorded, {installed} here")
    return differences + sandbox_backend_differences(recorded, current.get("sandbox_backend"))
