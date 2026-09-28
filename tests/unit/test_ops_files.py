"""Tests for the ops/deployment files: compose stack, CI workflow, agent Dockerfile,
.dockerignore, and the cron example.

These assert on parsed structure (via `yaml.safe_load`) rather than raw substrings
wherever the file is YAML, so the checks survive reformatting and catch structural
regressions (e.g. a security option landing under the wrong service) that a substring
match would miss. Dockerfile/.dockerignore/cron.example are plain text, so those are
checked by substring.

Security posture under test (see task-25 controller rulings):
  - no Docker socket is mounted anywhere in the compose file (root-equivalent host access)
  - the sandbox-image service is gone; the agent's own container is the sandbox boundary,
    running the model-written analysis code via the subprocess backend
  - the agent container runs locked down: read-only rootfs, all capabilities dropped,
    no-new-privileges, non-root user
  - Mailpit -- a local test inbox, never a production relay -- is confined to loopback
"""
import os
import re
import shutil
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def _load_yaml(rel_path: str) -> dict:
    return yaml.safe_load((ROOT / rel_path).read_text())


def _compose() -> dict:
    return _load_yaml("docker/compose.yaml")


def _ci() -> dict:
    return _load_yaml(".github/workflows/ci.yml")


def _agent_service() -> dict:
    return _compose()["services"]["agent"]


def _env_as_dict(service: dict) -> dict:
    """Normalize a compose `environment` block (list ["K=V", ...] or mapping {K: V})
    into a plain dict of str -> str, so tests don't care which form was used."""
    env = service.get("environment") or []
    if isinstance(env, dict):
        return {str(k): str(v) for k, v in env.items()}
    result: dict[str, str] = {}
    for item in env:
        key, _, value = str(item).partition("=")
        result[key] = value
    return result


def _all_steps(ci: dict):
    for job in ci["jobs"].values():
        yield from job.get("steps", [])


# --- compose.yaml: no root-equivalent host access -------------------------------


def test_no_volume_mounts_the_docker_socket():
    compose = _compose()
    for service_name, service in compose["services"].items():
        for volume in service.get("volumes", []) or []:
            volume_str = volume if isinstance(volume, str) else str(volume)
            assert "docker.sock" not in volume_str, f"{service_name} must not mount the docker socket"


def test_sandbox_image_service_is_not_present():
    assert "sandbox-image" not in _compose()["services"]


# --- compose.yaml: agent service --------------------------------------------------


def test_agent_environment_uses_subprocess_backend_and_plaintext_mailpit():
    env = _env_as_dict(_agent_service())
    assert env["AGENT_SANDBOX_BACKEND"] == "subprocess"
    assert env["AGENT_SMTP_HOST"] == "mailpit"
    assert env["AGENT_SMTP_STARTTLS"] == "false"


def test_agent_service_is_locked_down():
    agent = _agent_service()
    assert agent["read_only"] is True
    assert "ALL" in agent["cap_drop"]
    assert "no-new-privileges:true" in agent["security_opt"]
    assert "/tmp" in agent["tmpfs"]


def test_agent_service_has_resource_backstops_for_the_subprocess_sandbox():
    # There is no nested sandbox container (no --network none / --pids-limit of its
    # own) with the subprocess backend, so these container-level limits are what bound
    # model-written code instead: a process-count ceiling, a memory ceiling generous
    # enough for the agent plus one 512 MB-RLIMIT_AS sandbox child, and a CPU ceiling.
    agent = _agent_service()
    assert agent["pids_limit"] == 128
    assert agent["mem_limit"] == "1536m"
    assert agent["cpus"] == 2.0


def test_runs_directory_is_a_named_volume_not_a_bind_mount():
    compose = _compose()
    assert "runs" in (compose.get("volumes") or {})
    agent_volumes = _agent_service().get("volumes", [])
    assert any(
        (v.startswith("runs:") if isinstance(v, str) else v.get("source") == "runs")
        for v in agent_volumes
    ), "expected the named volume 'runs' mounted into the agent service"


def test_scheduler_service_runs_the_schedule_command_locked_down_like_the_agent():
    services = _compose()["services"]
    scheduler, agent = services["scheduler"], services["agent"]
    assert scheduler["command"] == ["schedule"] and scheduler["restart"] == "unless-stopped"
    assert scheduler["profiles"] == ["schedule"]  # a plain `up` still starts only the one-shot agent
    for key in ("read_only", "cap_drop", "security_opt", "tmpfs", "pids_limit", "mem_limit", "cpus", "volumes"):
        assert scheduler[key] == agent[key], key


def test_scheduler_sends_through_the_relay_in_env_not_mailpit():
    # The daily report must reach a real inbox: no Mailpit overrides, so the AGENT_SMTP_* settings from .env apply.
    scheduler = _compose()["services"]["scheduler"]
    env = _env_as_dict(scheduler)
    assert env["AGENT_SANDBOX_BACKEND"] == "subprocess"
    assert not any(name.startswith("AGENT_SMTP_") for name in env)
    assert "mailpit" not in (scheduler.get("depends_on") or [])


# --- compose.yaml: mailpit --------------------------------------------------------


def test_mailpit_accepts_insecure_local_auth_and_binds_loopback_only():
    mailpit = _compose()["services"]["mailpit"]
    env = _env_as_dict(mailpit)
    assert env.get("MP_SMTP_AUTH_ACCEPT_ANY") == "1"
    assert env.get("MP_SMTP_AUTH_ALLOW_INSECURE") == "1"
    ports = [str(p) for p in mailpit.get("ports", [])]
    assert len(ports) >= 2
    assert all(p.startswith("127.0.0.1:") for p in ports), "mailpit ports must not bind all interfaces"


# --- .github/workflows/ci.yml ------------------------------------------------------


def test_ci_workflow_has_read_only_top_level_permissions():
    assert _ci()["permissions"]["contents"] == "read"


def test_ci_workflow_triggers_on_push_and_pull_request():
    ci = _ci()
    # PyYAML (core schema, YAML 1.1) parses the bare `on:` key as the boolean True,
    # not the string "on" -- so the trigger list is looked up under `ci[True]`.
    triggers = ci[True]
    assert "push" in triggers
    assert "pull_request" in triggers


def test_ci_runs_unit_and_graph_test_tiers():
    runs = [step["run"] for step in _all_steps(_ci()) if "run" in step]
    assert any("pytest tests/unit" in r for r in runs)
    assert any("pytest tests/graph" in r for r in runs)


def test_ci_replay_step_is_conditional_on_a_recorded_cassette():
    for step in _all_steps(_ci()):
        if "run" in step and "nasdaq-agent replay" in step["run"]:
            assert "fixtures/cassettes/manifest.json" in step.get("if", "")
            return
    raise AssertionError("expected a step running `nasdaq-agent replay`")


# --- docker/Dockerfile.agent -------------------------------------------------------


def test_dockerfile_agent_runs_as_non_root_and_installs_no_docker_cli():
    dockerfile = (ROOT / "docker" / "Dockerfile.agent").read_text()
    assert "USER agent" in dockerfile
    assert "docker-cli" not in dockerfile


def test_dockerfile_agent_creates_user_with_uid_10001():
    dockerfile = (ROOT / "docker" / "Dockerfile.agent").read_text()
    assert "--uid 10001" in dockerfile


def test_dockerfile_agent_sets_matplotlib_cache_and_bytecode_env_vars():
    dockerfile = (ROOT / "docker" / "Dockerfile.agent").read_text()
    assert "MPLCONFIGDIR=/tmp/matplotlib" in dockerfile
    assert "XDG_CACHE_HOME=/tmp/.cache" in dockerfile
    assert "PYTHONDONTWRITEBYTECODE=1" in dockerfile


# --- .dockerignore ------------------------------------------------------------------


def test_dockerignore_excludes_env_and_venv():
    dockerignore = (ROOT / ".dockerignore").read_text()
    assert ".env" in dockerignore
    assert ".venv" in dockerignore


# --- ops/cron.example ----------------------------------------------------------------


def test_cron_example_runs_weekdays_via_the_agent_cli():
    cron = (ROOT / "ops" / "cron.example").read_text()
    assert "1-5" in cron
    assert "nasdaq-agent run" in cron


# --- Tasks 22+23 fix round 1: one pinned environment for record, replay, CI and the images ---------------------


def _constraints() -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in (ROOT / "constraints.txt").read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            name, _, version = line.partition("==")
            pins[name.lower().replace("_", "-")] = version
    return pins


def test_systemd_unit_keeps_the_scheduler_running_as_an_unprivileged_user():
    unit = (ROOT / "ops" / "nasdaq-agent-schedule.service").read_text()
    lines = [line.strip() for line in unit.splitlines() if line.strip() and not line.startswith("#")]
    assert any(line.startswith("ExecStart=") and line.endswith("nasdaq-agent schedule") for line in lines)
    assert "Restart=on-failure" in lines and "NoNewPrivileges=true" in lines
    user = next(line for line in lines if line.startswith("User="))
    assert user != "User=root"
    assert "WantedBy=multi-user.target" in lines


def test_constraints_file_pins_the_environment_and_says_how_to_regenerate_it():
    lines = (ROOT / "constraints.txt").read_text().splitlines()
    assert lines[0].startswith("#") and "pip freeze --exclude-editable" in lines[0]
    assert all(re.fullmatch(r"[A-Za-z0-9_.\-]+==\S+", line) for line in lines[1:] if line.strip())
    pins = _constraints()
    for name in ("langchain-core", "langchain", "langgraph", "langchain-google-genai", "langchain-anthropic",
                 "langchain-openai", "openai", "pydantic", "numpy", "pandas", "typer", "yfinance"):
        assert name in pins, name
    assert "nasdaq-agent" not in pins  # the project itself is installed editable, never pinned


def test_install_paths_use_the_constraints():
    makefile = (ROOT / "Makefile").read_text()
    assert 'pip install -c constraints.txt -e ".[dev]"' in makefile
    runs = [step["run"] for step in _all_steps(_ci()) if "run" in step]
    assert any('pip install -c constraints.txt -e ".[dev]"' in run for run in runs)
    dockerfile = (ROOT / "docker" / "Dockerfile.agent").read_text()
    assert "COPY constraints.txt" in dockerfile and "pip install --no-cache-dir -c constraints.txt ." in dockerfile


def test_sandbox_image_pins_the_same_pandas_and_numpy_as_the_constraints():
    """The subprocess sandbox runs model code in the host environment and the Docker sandbox in its image: the same
    pins mean a replay's results do not depend on which backend ran."""
    sandbox = (ROOT / "docker" / "Dockerfile.sandbox").read_text()
    image_pins = dict(re.findall(r'"(pandas|numpy)==([^"]+)"', sandbox))
    pins = _constraints()
    assert image_pins == {"pandas": pins["pandas"], "numpy": pins["numpy"]}


def test_make_replay_supplies_a_default_recipient_to_that_command_only():
    makefile = (ROOT / "Makefile").read_text()
    assert "AGENT_EMAIL_TO ?= replay@example.com" in makefile
    make = shutil.which("make")
    if make is None:
        return
    env = {name: value for name, value in os.environ.items() if name != "AGENT_EMAIL_TO"}
    replay = subprocess.run([make, "-n", "replay"], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    assert "AGENT_EMAIL_TO=replay@example.com" in replay.stdout and "nasdaq-agent replay" in replay.stdout
    run = subprocess.run([make, "-n", "run"], cwd=ROOT, env=env, capture_output=True, text=True, check=True)
    assert "replay@example.com" not in run.stdout


def test_make_record_notes_the_pinned_environment_precondition():
    makefile = (ROOT / "Makefile").read_text()
    record_recipe = makefile.split("\nrecord:", 1)[1].split("\n\n", 1)[0]
    assert "constraints.txt" in record_recipe or "constraints.txt" in makefile.split("\nrecord:", 1)[0].rsplit("\n\n", 1)[-1]


def test_ci_replay_checks_the_recorded_exit_code():
    for step in _all_steps(_ci()):
        if "run" in step and "nasdaq-agent replay" in step["run"]:
            assert "nasdaq-agent replay --check" in step["run"]
            return
    raise AssertionError("expected a step running `nasdaq-agent replay`")
