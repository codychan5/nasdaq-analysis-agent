import pytest

from tests.unit.test_subprocess_runner import GOOD_CODE, sandbox_dir  # noqa: F401  reuse the fixture

pytestmark = pytest.mark.docker


@pytest.fixture(autouse=True)
def _need_docker():
    """Controller correction 2: skip with a clear, distinct reason for each of the two
    environmental preconditions -- an unreachable daemon and a missing sandbox image -- instead
    of a single generic skip that would hide which one is actually missing.
    """
    from nasdaq_agent.sandbox.runner import SANDBOX_IMAGE, docker_available, sandbox_image_available

    if not docker_available():
        pytest.skip("Docker daemon not reachable")
    if not sandbox_image_available():
        pytest.skip(
            f"Sandbox image {SANDBOX_IMAGE} not built; run: "
            f"docker build -f docker/Dockerfile.sandbox -t {SANDBOX_IMAGE} ."
        )


def test_docker_happy_path(sandbox_dir):
    from nasdaq_agent.sandbox.runner import DockerRunner
    r = DockerRunner().run(GOOD_CODE, sandbox_dir)
    assert r.exit_code == 0 and r.backend == "docker" and abs(r.result.cumulative_return_pct - 11.0) < 0.01


def test_docker_timeout(sandbox_dir):
    from nasdaq_agent.sandbox.runner import DockerRunner, SandboxLimits, TIMEOUT_EXIT_CODE
    r = DockerRunner(limits=SandboxLimits(wall_clock_s=2)).run("while True:\n    pass", sandbox_dir)
    assert r.timed_out and r.exit_code == TIMEOUT_EXIT_CODE
