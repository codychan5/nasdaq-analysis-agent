import os
import sys
import sysconfig
from pathlib import Path

from langchain_core.tools import tool

from ...artifacts import sha256_text
from ...metrics import METRIC_NAMES
from ...sandbox.gate import ALLOWED_IMPORTS
from ..context import AttemptInfo, RunContext
from .common import Deps, err, ok, precondition, record_call, round_floats

CONTRACT_REMINDER = (f"df, bench and meta are preloaded; use column 'adj_close'; only {sorted(ALLOWED_IMPORTS)} may be imported; "
                     f"no network; assign a dict named result with keys {list(METRIC_NAMES)}")
STDERR_TAIL_CHARS = 600
# Task 22 correction f: model-visible error text shows the Docker sandbox's layout (docker/Dockerfile.sandbox:
# python:3.12-slim, WORKDIR /work) whichever backend ran the code. The subprocess backend's tracebacks otherwise carry
# this machine's paths -- the sandbox directory, which names the run, and the interpreter's own directories, which
# name the OS user -- so the next prompt would differ across runs and machines, replay would miss the cache, and the
# local filesystem layout would be sent to the model provider.
SANDBOX_IMAGE_WORKDIR = "/work"
SANDBOX_IMAGE_PYTHON_LIB = f"/usr/local/lib/python{sys.version_info.major}.{sys.version_info.minor}"
SANDBOX_IMAGE_SITE_PACKAGES = f"{SANDBOX_IMAGE_PYTHON_LIB}/site-packages"


def _host_path_rewrites(sandbox_dir: Path) -> list[tuple[str, str]]:
    interpreter = sysconfig.get_paths()
    pairs = [(sandbox_dir, SANDBOX_IMAGE_WORKDIR),
             (interpreter["purelib"], SANDBOX_IMAGE_SITE_PACKAGES), (interpreter["platlib"], SANDBOX_IMAGE_SITE_PACKAGES),
             (interpreter["stdlib"], SANDBOX_IMAGE_PYTHON_LIB), (interpreter["platstdlib"], SANDBOX_IMAGE_PYTHON_LIB)]
    rewrites = {(variant, image_path) for host_path, image_path in pairs
                for variant in (str(host_path), os.path.realpath(host_path))}
    # Longest first: a virtualenv's site-packages sits inside its platstdlib directory and must be rewritten first.
    return sorted(rewrites, key=lambda rewrite: len(rewrite[0]), reverse=True)


def _as_in_sandbox_image(text: str, sandbox_dir: Path) -> str:
    for host_path, image_path in _host_path_rewrites(sandbox_dir):
        text = text.replace(host_path, image_path)
    return text


def make_run_python(ctx: RunContext, deps: Deps):
    @tool("run_python")
    def run_python(code: str) -> str:
        """Execute Python analysis code in the sandbox. Variables df (ticker bars), bench (benchmark bars) and meta
        (dict with definitions and required_keys) are preloaded as pandas objects. Only pandas, numpy, math,
        statistics, json and datetime may be imported. There is no network. The code must assign a dict named
        result containing every required key. Returns the parsed result or the traceback."""
        # Computed up front (not only once history_ready is confirmed) so a precondition
        # refusal below can still be recorded against the code that was actually submitted.
        code_hash = sha256_text(code)

        # Amendment: every return path below funnels through _record, so tool_log.jsonl gets
        # exactly one entry per call whatever the outcome -- refusals and terminal guards
        # included, not only the success/failure path that used to fall through to the bottom.
        def _record(outcome: str) -> str:
            record_call(ctx, deps, "run_python", {"code_hash": code_hash}, outcome)
            return outcome

        blocked = precondition(ctx, "history_ready", "run_python", "call get_price_history first")
        if blocked:
            return _record(blocked)
        # Review fix (Important): a finished stage must close. Once verify_analysis has
        # passed, running more analysis code can only waste cap budget on numbers the report
        # will never use (compose_report cites the verified attempt, not the latest one).
        if ctx.progress.verified:
            return _record(err("precondition for run_python not met: analysis is already verified; "
                               "call compose_report instead of running more analysis code"))
        if ctx.progress.analysis_terminal:
            return _record(err("analysis is terminal after repeated failures; call give_up with a reason"))
        n = len(ctx.analysis.attempts) + 1
        if n > deps.settings.max_code_runs:
            # Guard for a run of successes with no intervening verify_analysis call: this
            # branch's own failure path (below) only sets analysis_terminal on a FAILED
            # attempt reaching the cap, and verify_analysis's own cap check (see verify.py)
            # only fires when verify_analysis is actually called. If the model calls
            # run_python repeatedly without ever verifying in between, every one of those
            # attempts succeeds, and neither of those checks runs -- this is the only thing
            # that stops it once total attempts exceed the cap. Not unreachable in practice.
            ctx.progress.analysis_terminal = True
            return _record(err(f"code-run cap of {deps.settings.max_code_runs} reached; analysis is terminal; call give_up"))
        deps.run_dir.write_text(f"attempts/attempt_{n}.py", code)
        # Absolute, because the subprocess backend starts its child inside the sandbox directory: a relative path
        # (the default AGENT_ARTIFACTS_DIR is ./runs) would be resolved against itself and the bootstrap never found.
        sandbox_dir = deps.run_dir.sandbox_dir.resolve()
        res = deps.runner.run(code, sandbox_dir)
        deps.run_dir.write_text(f"attempts/attempt_{n}.out", res.stdout_tail)
        deps.run_dir.write_text(f"attempts/attempt_{n}.err", res.stderr_tail)
        succeeded = res.exit_code == 0 and res.result is not None
        ctx.analysis.attempts.append(AttemptInfo(number=n, code_hash=code_hash, backend=res.backend, ok=succeeded,
                                                 exit_code=res.exit_code, wall_time_s=res.wall_time_s, error=res.error))
        # Task 22 correction b: no backend in a model-visible result; it depends on the machine (Docker or not), and
        # stays in ctx.analysis.attempts and the report's footer. Fix round 1, K1: the model sees the result's floats
        # rounded; ctx.analysis.latest_result keeps full precision for verify_analysis.
        if succeeded:
            ctx.analysis.latest_result = res.result
            ctx.progress.successful_run = True
            outcome = ok({"attempt": n, "exit_code": 0, "code_hash": code_hash,
                          "result": round_floats(res.result.model_dump()), "next": "call verify_analysis"})
        else:
            remaining = deps.settings.max_code_runs - n
            error = _as_in_sandbox_image(str(res.error), sandbox_dir)
            # Rewritten before the tail is cut, so the tail's length does not depend on this machine's paths.
            stderr_tail = _as_in_sandbox_image(res.stderr_tail.strip(), sandbox_dir)[-STDERR_TAIL_CHARS:]
            # Controller correction a: mark analysis terminal on THIS attempt once its number
            # reaches the cap, instead of waiting for a would-be seventh call that the model
            # was never told to expect. The up-front check above is kept as a guard only.
            if n >= deps.settings.max_code_runs:
                ctx.progress.analysis_terminal = True
                outcome = err(f"attempt {n} failed: {error}. exit {res.exit_code}. "
                              f"stderr tail: {stderr_tail}. code-run cap of {deps.settings.max_code_runs} "
                              "reached; analysis is terminal; call give_up with a reason.")
            else:
                outcome = err(f"attempt {n} failed: {error}. exit {res.exit_code}. "
                              f"stderr tail: {stderr_tail}. remaining attempts: {remaining}. {CONTRACT_REMINDER}")
        return _record(outcome)
    return run_python
