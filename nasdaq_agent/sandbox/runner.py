import json
import logging
import os
import platform
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .bootstrap import write_sandbox_files
from .contract import STDOUT_CAP_CHARS, ContractError, RunResult, parse_sentinel_output, tail
from .gate import check_code

log = logging.getLogger(__name__)

GATE_EXIT_CODE = 2
TIMEOUT_EXIT_CODE = 124
SETUP_EXIT_CODE = 125
BYTES_PER_MB = 1024 * 1024
SANDBOX_SUBDIR = "sandbox"

# The child's entire environment. Deliberately not inherited from the parent (no PATH, HOME, etc.)
# so the sandbox cannot pick up ambient configuration. The three THREADS variables force pandas'/
# numpy's BLAS backend (OpenBLAS, MKL, or the OpenMP-based reductions) to run single-threaded:
# RLIMIT_NPROC is intentionally not used to cap concurrency here (see _apply_rlimits), so this is
# the actual control on how many OS threads the child's math libraries are allowed to spin up.
SANDBOX_ENV = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


class CodeRunner(Protocol):
    name: str
    def run(self, code: str, sandbox_dir: Path) -> RunResult: ...


@dataclass(frozen=True)
class SandboxLimits:
    wall_clock_s: float = 20.0
    cpu_s: int = 10
    memory_mb: int = 512
    file_size_mb: int = 10
    max_processes: int = 64
    max_open_files: int = 32


def prepare_sandbox_dir(run_dir: Path, ticker_csv: str, benchmark_csv: str, meta: dict) -> Path:
    sandbox_dir = run_dir / SANDBOX_SUBDIR
    sandbox_dir.mkdir(parents=True, exist_ok=True)
    (sandbox_dir / "ticker.csv").write_text(ticker_csv)
    (sandbox_dir / "benchmark.csv").write_text(benchmark_csv)
    (sandbox_dir / "meta.json").write_text(json.dumps(meta))
    return sandbox_dir


def _apply_rlimits(limits: SandboxLimits) -> None:
    """Runs in the child, after fork and before exec (via preexec_fn). Memory caps are Linux-only;
    macOS ignores RLIMIT_AS for this purpose. RLIMIT_NPROC is deliberately NOT set here: it is a
    per-user limit rather than a per-process one, and on Linux it counts threads, not processes,
    which starves numpy's OpenBLAS thread pool and can make even a correct import fail under load.
    `limits.max_processes` is kept on SandboxLimits for the Docker backend, which enforces it via
    a cgroup/container flag instead of this rlimit.
    """
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (limits.cpu_s, limits.cpu_s))
    resource.setrlimit(resource.RLIMIT_FSIZE, (limits.file_size_mb * BYTES_PER_MB,) * 2)
    resource.setrlimit(resource.RLIMIT_NOFILE, (limits.max_open_files,) * 2)
    if platform.system() == "Linux":
        resource.setrlimit(resource.RLIMIT_AS, (limits.memory_mb * BYTES_PER_MB,) * 2)


class SubprocessRunner:
    name = "subprocess"

    def __init__(self, limits: SandboxLimits = SandboxLimits(), python_exe: str = sys.executable):
        self.limits, self.python_exe = limits, python_exe

    def run(self, code: str, sandbox_dir: Path) -> RunResult:
        # Make the sandbox path absolute, as DockerRunner's volume mount already is. The child runs with the
        # sandbox directory as its working directory, so a relative path to the bootstrap would be resolved against
        # the sandbox again and never found (the default AGENT_ARTIFACTS_DIR, ./runs, is relative).
        sandbox_dir = Path(sandbox_dir).resolve()
        gate = check_code(code)
        if not gate.ok:
            return RunResult(exit_code=GATE_EXIT_CODE, backend=self.name, wall_time_s=0.0,
                             error="rejected before execution: " + "; ".join(gate.reasons))
        started = time.monotonic()
        setup_error: Exception | None = None
        proc: subprocess.Popen | None = None
        try:
            bootstrap_path, code_path = write_sandbox_files(sandbox_dir, code)
            cmd = [self.python_exe, "-I", str(bootstrap_path), str(code_path)]
            proc = subprocess.Popen(cmd, cwd=sandbox_dir, env=SANDBOX_ENV, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    start_new_session=True, preexec_fn=lambda: _apply_rlimits(self.limits))
        except (OSError, ValueError, subprocess.SubprocessError) as e:
            setup_error = e
        if setup_error is not None:
            return RunResult(exit_code=SETUP_EXIT_CODE, backend=self.name, wall_time_s=0.0,
                             error=f"sandbox setup failed: {type(setup_error).__name__}: {setup_error}")
        try:
            out, err = proc.communicate(timeout=self.limits.wall_clock_s)
            timed_out, exit_code = False, proc.returncode
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            out, err = proc.communicate()
            timed_out, exit_code = True, TIMEOUT_EXIT_CODE
        wall = time.monotonic() - started
        stdout = out.decode(errors="replace")[:STDOUT_CAP_CHARS]
        stderr = err.decode(errors="replace")
        if timed_out:
            return RunResult(exit_code=exit_code, backend=self.name, wall_time_s=wall, timed_out=True,
                             stdout_tail=tail(stdout), stderr_tail=tail(stderr),
                             error=f"killed after {self.limits.wall_clock_s}s wall clock")
        if exit_code != 0:
            return RunResult(exit_code=exit_code, backend=self.name, wall_time_s=wall,
                             stdout_tail=tail(stdout), stderr_tail=tail(stderr), error="code exited with an error")
        try:
            result = parse_sentinel_output(stdout)
        except ContractError as e:
            return RunResult(exit_code=1, backend=self.name, wall_time_s=wall, stdout_tail=tail(stdout),
                             stderr_tail=tail(stderr), error=str(e))
        return RunResult(exit_code=0, backend=self.name, wall_time_s=wall, stdout_tail=tail(stdout),
                         stderr_tail=tail(stderr), result=result)


import uuid
from typing import Callable

from ..config import SandboxBackend

# 0.2.0: pandas and numpy pinned to constraints.txt. A new tag, so an image built from the
# old Dockerfile is never mistaken for one with the pinned versions.
SANDBOX_IMAGE = "nasdaq-agent-sandbox:0.2.0"
DOCKER_INFO_TIMEOUT_S = 5
DOCKER_KILL_TIMEOUT_S = 10
SANDBOX_IMAGE_BUILD_CMD = f"docker build -f docker/Dockerfile.sandbox -t {SANDBOX_IMAGE} ."


def docker_available(docker_bin: str = "docker") -> bool:
    try:
        return subprocess.run([docker_bin, "info"], capture_output=True, timeout=DOCKER_INFO_TIMEOUT_S).returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def sandbox_image_available(image: str = SANDBOX_IMAGE, docker_bin: str = "docker") -> bool:
    """Best-effort probe for the built sandbox image, used only to decide whether the Docker
    backend and the Docker test suite can actually run. Any failure here -- missing docker
    binary, unreachable daemon, malformed image reference -- means "not available"; nothing
    downstream needs the failure detail, only the yes/no answer, so the broad except below does
    not silently hide an error anyone would need to act on.
    """
    try:
        return subprocess.run([docker_bin, "image", "inspect", image], capture_output=True,
                              timeout=DOCKER_INFO_TIMEOUT_S).returncode == 0
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


def docker_backend_available() -> bool:
    """The Docker backend is only really usable when BOTH the daemon is reachable AND the sandbox
    image has been built. A daemon-only check would let `auto` select Docker and then fail every
    run with "Unable to find image"."""
    return docker_available() and sandbox_image_available()


class DockerRunner:
    name = "docker"

    def __init__(self, image: str = SANDBOX_IMAGE, limits: SandboxLimits = SandboxLimits(), docker_bin: str = "docker"):
        self.image, self.limits, self.docker_bin = image, limits, docker_bin

    def _command(self, container: str, sandbox_dir: Path) -> list[str]:
        # Same thread-count variables as SANDBOX_ENV (the subprocess backend's env) so both
        # backends constrain pandas'/numpy's BLAS thread pools identically.
        env_flags = [flag for k, v in SANDBOX_ENV.items() for flag in ("-e", f"{k}={v}")]
        return [self.docker_bin, "run", "--rm", "--name", container, "--network", "none", "--read-only",
                "--tmpfs", "/tmp:size=64m", "--memory", f"{self.limits.memory_mb}m", "--cpus", "1",
                "--pids-limit", str(self.limits.max_processes), "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges", "--user", "65534",
                *env_flags,
                "-v", f"{sandbox_dir.resolve()}:/work:ro", "-w", "/work", self.image,
                "python", "-I", "/work/bootstrap.py", "/work/analysis_code.py"]

    def run(self, code: str, sandbox_dir: Path) -> RunResult:
        gate = check_code(code)
        if not gate.ok:
            return RunResult(exit_code=GATE_EXIT_CODE, backend=self.name, wall_time_s=0.0,
                             error="rejected before execution: " + "; ".join(gate.reasons))
        container = f"nasdaq-sbx-{uuid.uuid4().hex[:12]}"
        started = time.monotonic()
        # Mirrors SubprocessRunner's failure contract: write_sandbox_files and the subprocess
        # invocation are wrapped together so OSError/ValueError/SubprocessError (missing docker
        # binary, bad arguments, disk full while writing the sandbox files, ...) return a
        # SETUP_EXIT_CODE RunResult instead of an uncaught exception. TimeoutExpired is caught
        # first and separately below because it is a SubprocessError subclass but is not a setup
        # failure -- the process started fine and simply ran past the wall clock.
        setup_error: Exception | None = None
        timed_out = exit_code = out = err = None
        try:
            write_sandbox_files(sandbox_dir, code)
            proc = subprocess.run(self._command(container, sandbox_dir), capture_output=True,
                                  timeout=self.limits.wall_clock_s)
            timed_out, exit_code, out, err = False, proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as e:
            try:
                subprocess.run([self.docker_bin, "kill", container], capture_output=True,
                               timeout=DOCKER_KILL_TIMEOUT_S)
            except (OSError, subprocess.SubprocessError):
                # Best-effort cleanup: --rm may have already removed the container on exit, or
                # the daemon is wedged and the kill itself hit DOCKER_KILL_TIMEOUT_S (a
                # subprocess.TimeoutExpired, itself a SubprocessError, so it lands here too).
                # Either way we do not let a stuck kill block run() forever -- we still fall
                # through and return the timeout RunResult below.
                pass
            timed_out, exit_code, out, err = True, TIMEOUT_EXIT_CODE, e.stdout or b"", e.stderr or b""
        except (OSError, ValueError, subprocess.SubprocessError) as e:
            setup_error = e
        if setup_error is not None:
            return RunResult(exit_code=SETUP_EXIT_CODE, backend=self.name, wall_time_s=0.0,
                             error=f"sandbox setup failed: {type(setup_error).__name__}: {setup_error}")
        wall = time.monotonic() - started
        stdout = out.decode(errors="replace")[:STDOUT_CAP_CHARS]
        stderr = err.decode(errors="replace")
        if timed_out:
            return RunResult(exit_code=exit_code, backend=self.name, wall_time_s=wall, timed_out=True,
                             stdout_tail=tail(stdout), stderr_tail=tail(stderr),
                             error=f"killed after {self.limits.wall_clock_s}s wall clock")
        if exit_code != 0:
            return RunResult(exit_code=exit_code, backend=self.name, wall_time_s=wall,
                             stdout_tail=tail(stdout), stderr_tail=tail(stderr), error="code exited with an error")
        try:
            result = parse_sentinel_output(stdout)
        except ContractError as e:
            return RunResult(exit_code=1, backend=self.name, wall_time_s=wall, stdout_tail=tail(stdout),
                             stderr_tail=tail(stderr), error=str(e))
        return RunResult(exit_code=0, backend=self.name, wall_time_s=wall, stdout_tail=tail(stdout),
                         stderr_tail=tail(stderr), result=result)


def select_runner(backend: SandboxBackend, available: Callable[[], bool] = docker_backend_available) -> CodeRunner:
    """Docker, unless the operator explicitly chose the subprocess sandbox. Both auto and docker refuse to run when
    Docker cannot: the subprocess sandbox screens model-written code but does not isolate it, so it runs only by
    explicit choice, never as a silent fallback. The refusal happens at setup, before any model call is paid for.

    `available` defaults to `docker_backend_available`, which requires both a reachable daemon
    and a built sandbox image: a daemon-only check would let `auto` select Docker and then fail
    every run with "Unable to find image".
    """
    if backend == SandboxBackend.subprocess:
        return SubprocessRunner()
    if available():
        return DockerRunner()
    # Kept under the failure notice's 300-character cap (finalize.MAX_ERROR_CHARS), prefix included, so both ways
    # forward reach the operator whole.
    raise RuntimeError(
        "No Docker sandbox: the daemon is unreachable or the sandbox image is missing. "
        f"Build it with: {SANDBOX_IMAGE_BUILD_CMD} "
        "Or set AGENT_SANDBOX_BACKEND=subprocess for a sandbox that screens the code but does not isolate it."
    )
