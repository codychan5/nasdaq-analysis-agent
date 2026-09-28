import subprocess
from pathlib import Path

import pytest


def test_auto_prefers_docker_when_available():
    from nasdaq_agent.config import SandboxBackend
    from nasdaq_agent.sandbox.runner import select_runner
    runner, warning = select_runner(SandboxBackend.auto, available=lambda: True)
    assert runner.name == "docker" and warning is None


def test_auto_falls_back_with_warning():
    from nasdaq_agent.config import SandboxBackend
    from nasdaq_agent.sandbox.runner import select_runner
    runner, warning = select_runner(SandboxBackend.auto, available=lambda: False)
    assert runner.name == "subprocess" and "Docker" in warning


def test_explicit_docker_without_daemon_raises():
    from nasdaq_agent.config import SandboxBackend
    from nasdaq_agent.sandbox.runner import select_runner
    with pytest.raises(RuntimeError):
        select_runner(SandboxBackend.docker, available=lambda: False)


def test_auto_falls_back_when_daemon_up_but_image_missing(monkeypatch):
    """Coordinator addendum: a reachable daemon alone is not enough. select_runner's default
    availability check must also require the sandbox image to be built, or `auto` would select
    Docker and then fail every run with "Unable to find image"."""
    from nasdaq_agent.config import SandboxBackend
    from nasdaq_agent.sandbox import runner as runner_module
    from nasdaq_agent.sandbox.runner import select_runner

    monkeypatch.setattr(runner_module, "docker_available", lambda: True)
    monkeypatch.setattr(runner_module, "sandbox_image_available", lambda: False)
    runner, warning = select_runner(SandboxBackend.auto)
    assert runner.name == "subprocess" and warning is not None and "sandbox image" in warning


def test_docker_command_has_isolation_flags():
    """Controller correction 1: replaces the task plan's daemon-dependent test_docker_has_no_network
    (which asserted on the process outcome, not the flag) with a pure command-construction check
    that needs no Docker daemon."""
    from nasdaq_agent.sandbox.runner import DockerRunner

    cmd = DockerRunner()._command("c1", Path("/tmp/x"))

    def follows(flag: str, value: str) -> bool:
        return cmd[cmd.index(flag) + 1] == value

    assert follows("--network", "none")
    assert "--read-only" in cmd
    assert follows("--cap-drop", "ALL")
    assert follows("--security-opt", "no-new-privileges")
    assert follows("--memory", "512m")
    assert follows("--pids-limit", "64")
    assert cmd[cmd.index("-v") + 1].endswith(":/work:ro")


def test_docker_setup_failure_returns_setup_exit_code(tmp_run_dir, monkeypatch):
    """Controller correction 4: DockerRunner must match SubprocessRunner's failure contract when
    the docker CLI invocation itself cannot be run (e.g. the docker binary is missing)."""
    from nasdaq_agent.sandbox import runner as runner_module
    from nasdaq_agent.sandbox.runner import DockerRunner, SETUP_EXIT_CODE

    def _raise_oserror(*args, **kwargs):
        raise OSError("boom")

    monkeypatch.setattr(runner_module.subprocess, "run", _raise_oserror)
    r = DockerRunner().run("result = {}", tmp_run_dir)
    assert r.exit_code == SETUP_EXIT_CODE and "sandbox setup failed" in r.error


def test_docker_kill_on_timeout_has_its_own_timeout(tmp_path, monkeypatch):
    """Review fix (Important): the `docker kill` issued after a TimeoutExpired must itself carry
    DOCKER_KILL_TIMEOUT_S, so a wedged daemon can't block run() forever. If the kill call also
    times out, that TimeoutExpired (a SubprocessError) must be swallowed and run() must still
    return the original timeout RunResult rather than raising."""
    from nasdaq_agent.sandbox import runner as runner_module
    from nasdaq_agent.sandbox.runner import DockerRunner, DOCKER_KILL_TIMEOUT_S, TIMEOUT_EXIT_CODE

    calls = []

    def _fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout"), output=b"", stderr=b"")

    monkeypatch.setattr(runner_module.subprocess, "run", _fake_run)
    r = DockerRunner().run("result = {}", tmp_path)

    assert r.timed_out and r.exit_code == TIMEOUT_EXIT_CODE
    assert len(calls) == 2
    run_cmd, run_kwargs = calls[0]
    kill_cmd, kill_kwargs = calls[1]
    assert run_cmd[1] == "run" and kill_cmd[1] == "kill"
    assert kill_kwargs.get("timeout") == DOCKER_KILL_TIMEOUT_S
