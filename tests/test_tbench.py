"""The Terminal-Bench adapter, on a task directory written here (no benchmark content)."""

import io
import tarfile
from pathlib import Path

import pytest

from fakes import FakeBox, FakeFactory
from forkeval.policy import GatesPolicy
from forkeval.runner import Runner, built_image_name, pytest_passed
from forkeval.store import Store
from forkeval.tbench import Unsupported, load_tbench_task, load_tbench_tasks, parse_dockerfile

COMPOSE = """services:
  client:
    build: {dockerfile: Dockerfile}
    command: ["sh", "-c", "sleep infinity"]
    environment: [TEST_DIR=/tests]
"""


def make_task(root: Path, name: str = "demo", dockerfile: str | None = None, compose: str = COMPOSE) -> Path:
    d = root / name
    (d / "tests").mkdir(parents=True)
    (d / "deps" / "sub").mkdir(parents=True)
    (d / "deps" / "a.txt").write_text("A")
    (d / "deps" / "sub" / "b.txt").write_text("B")
    (d / "tool.sh").write_text("#!/bin/sh\necho hi\n")
    (d / "tool.sh").chmod(0o755)
    (d / "task.yaml").write_text("instruction: |-\n  Make it work.\n  Then stop.\nmax_test_timeout_sec: 60.0\n")
    (d / "docker-compose.yaml").write_text(compose)
    (d / "run-tests.sh").write_text("pytest $TEST_DIR/test_outputs.py -rA\n")
    (d / "tests" / "test_outputs.py").write_text("def test_it(): pass\n")
    (d / "solution.sh").write_text("echo solved > /app/out\n")
    (d / "Dockerfile").write_text(dockerfile or (
        "# a comment\n"
        "FROM example.org/base:1\n"
        "ENV TOOL_HOME=/opt/tool MODE=fast\n"
        "WORKDIR /app\n"
        "RUN apt-get update && \\\n"
        "    apt-get install -y \\\n"
        "    # a comment inside a continuation\n"
        "    git\n"
        "COPY deps ./deps/\n"
        "COPY tool.sh /usr/local/bin/tool\n"
        'CMD ["python", "server.py"]\n'
    ))
    return d


def names(step) -> list[str]:
    with tarfile.open(fileobj=io.BytesIO(step.tar)) as tar:
        return sorted(tar.getnames())


def test_a_dockerfile_becomes_a_base_image_and_steps(tmp_path):
    task = load_tbench_task(make_task(tmp_path))
    assert task.image == "example.org/base:1"
    assert task.workdir == "/app" and task.env == {"TOOL_HOME": "/opt/tool", "MODE": "fast"}
    assert task.instruction == "Make it work.\nThen stop."
    mkdir, apt, copy_dir, copy_file = task.build
    assert mkdir.run == "mkdir -p -- /app" and mkdir.cwd == ""
    assert apt.run.split() == "apt-get update && apt-get install -y git".split()
    assert apt.cwd == "/app" and dict(apt.env)["MODE"] == "fast"
    # a directory is copied by its contents; a file keeps its mode
    assert names(copy_dir) == ["a.txt", "sub", "sub/b.txt"] and "./deps/" in copy_dir.run
    assert names(copy_file) == ["tool.sh"] and "/usr/local/bin/tool" in copy_file.run
    with tarfile.open(fileobj=io.BytesIO(copy_file.tar)) as tar:
        member = tar.getmember("tool.sh")
        assert member.mode & 0o111 and member.uid == 0
    assert set(task.test_files) == {"/tests/run-tests.sh", "/tests/test_outputs.py"}
    assert task.test_parser == "pytest" and "TEST_DIR=/tests" in task.test
    assert "solved" in task.solution


@pytest.mark.parametrize("dockerfile, why", [
    ("FROM a\nFROM b\n", "more than one FROM"),
    ("FROM a\nUSER app\n", "USER"),
    ("FROM a\nENV PATH=/x:$PATH\n", "PATH"),
    ("FROM a\nRUN --mount=type=cache,target=/x make\n", "flags"),
    ("FROM a\nCOPY --chown=1:1 tool.sh /x\n", "flags"),
    ("FROM a\nCOPY missing.txt /x\n", "not in the task directory"),
    ("FROM a\nRUN cat <<EOF > /x\nhi\nEOF\n", "heredoc"),
    ("FROM a\nADD https://example.org/x.tar /x\n", "URL"),
])
def test_what_it_cannot_reproduce_is_refused(tmp_path, dockerfile, why):
    with pytest.raises(Unsupported, match=why):
        parse_dockerfile(dockerfile, make_task(tmp_path))


def test_tasks_that_need_more_than_one_container_are_skipped_and_counted(tmp_path):
    make_task(tmp_path, "one")
    make_task(tmp_path, "two", compose=COMPOSE + "  db:\n    image: postgres\n")
    tasks, skipped = load_tbench_tasks(tmp_path)
    assert [t.id for t in tasks] == ["one"] and "2 services" in skipped["two"]


def test_env_later_lines_see_earlier_ones(tmp_path):
    _, _, _, env = parse_dockerfile("FROM a\nENV BASE=/opt\nENV BIN=${BASE}/bin\n", make_task(tmp_path))
    assert env == {"BASE": "/opt", "BIN": "/opt/bin"}


def test_pytest_summary_is_what_decides():
    ok = "collected 2\n=== short test summary info ===\nPASSED t.py::a\nPASSED t.py::b\n=== 2 passed ===\n"
    assert pytest_passed(ok)
    assert not pytest_passed(ok.replace("PASSED t.py::b", "FAILED t.py::b - boom"))
    assert not pytest_passed(ok.replace("PASSED t.py::b", "ERROR t.py::b"))
    assert not pytest_passed("bash: pytest: command not found\n")
    assert not pytest_passed("=== short test summary info ===\n=== no tests ran ===\n")


@pytest.mark.asyncio
async def test_the_image_is_built_once_and_runs_start_from_it(tmp_path):
    FakeFactory.images.clear()
    FakeFactory.created.clear()
    task = load_tbench_task(make_task(tmp_path))
    runner = Runner(factory=FakeFactory(), policy=GatesPolicy([]), store=Store(tmp_path / "data"), experiment="x")
    image = await runner.image_for(task)
    assert image == ":".join(built_image_name(task)) and image in FakeFactory.images
    built = FakeFactory.images[image]["commands"]
    assert [c[1] for c in built] == ["", "/app", "/app", "/app"]         # each step in its working directory
    assert await runner.image_for(task) == image
    assert [k for _, k in FakeFactory.created] == ["build"]               # not built twice
    # a second runner finds the image instead of building it
    again = Runner(factory=FakeFactory(), policy=GatesPolicy([]), store=Store(tmp_path / "data"), experiment="y")
    assert await again.image_for(task) == image and len(FakeFactory.created) == 1
    assert not FakeBox.live


@pytest.mark.asyncio
async def test_a_failing_build_step_stops_the_build(tmp_path):
    FakeFactory.images.clear()
    task = load_tbench_task(make_task(tmp_path, dockerfile="FROM a\nRUN fail now\n"))
    runner = Runner(factory=FakeFactory(), policy=GatesPolicy([]), store=Store(tmp_path / "data"), experiment="x")
    with pytest.raises(RuntimeError, match="build step failed"):
        await runner.image_for(task)
    assert not FakeFactory.images and not FakeBox.live
