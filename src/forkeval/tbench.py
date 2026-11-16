"""Tasks in the Terminal-Bench layout, read from a local checkout of that benchmark.

A task there is a directory:

    task.yaml            the instruction and the limits
    Dockerfile           the environment
    docker-compose.yaml  how the benchmark's own harness starts it
    run-tests.sh         installs the test's dependencies and runs pytest
    tests/               the test files, shown to nobody until scoring
    solution.sh          a reference solution

Nothing of the benchmark is copied into this repository: its files ask not to
be republished, so the adapter reads them where the user has put them.

There is no Docker here. The Dockerfile is read as what it mostly is, a base
image and a list of steps, and the steps are applied inside a sandbox started
from the base image, which is then saved as an image (see `Runner.image_for`).
That covers single-stage Dockerfiles made of FROM, WORKDIR, ENV, RUN, COPY and
ADD. Anything else raises `Unsupported`, and the caller counts the task as
skipped instead of running something that is not the task.
"""

from __future__ import annotations

import glob
import io
import json
import re
import shlex
import tarfile
from pathlib import Path

from .tasks import BuildStep, Task

#: Where the benchmark's harness puts the tests, and what run-tests.sh reads.
TEST_DIR = "/tests"


class Unsupported(ValueError):
    """The task needs something this adapter does not do."""


def _logical_lines(text: str) -> list[str]:
    """Dockerfile instructions, with continuation lines joined and comments dropped."""
    out: list[str] = []
    cur = ""
    for raw in text.splitlines():
        if raw.lstrip().startswith("#"):
            continue
        line = raw.rstrip()
        if line.endswith("\\"):
            cur += line[:-1] + " "
            continue
        cur += line
        if cur.strip():
            out.append(cur.strip())
        cur = ""
    if cur.strip():
        out.append(cur.strip())
    return out


_VAR = re.compile(r"\$(?:\{(\w+)\}|(\w+))")


def _expand(value: str, env: dict[str, str]) -> str:
    """Substitute references to variables set by earlier ENV lines. A reference
    to anything else (the base image's PATH, say) cannot be resolved without
    running the image, so it is refused."""

    def sub(m: re.Match[str]) -> str:
        name = m.group(1) or m.group(2)
        if name not in env:
            raise Unsupported(f"ENV refers to ${name}, which the Dockerfile does not set")
        return env[name]

    return _VAR.sub(sub, value)


def _q(s: str) -> str:
    return shlex.quote(s)


def _tar(members: list[tuple[Path, str]]) -> bytes:
    """An archive of (file or directory on disk, name inside the archive), owned by root."""

    def root(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.uid = info.gid = 0
        info.uname = info.gname = "root"
        return info

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for path, name in members:
            tar.add(path, arcname=name, filter=root)
    return buf.getvalue()


def _copy_step(task_dir: Path, args: str, cwd: str, env: dict[str, str]) -> BuildStep:
    parts = shlex.split(args)
    if any(p.startswith("--") for p in parts):
        raise Unsupported(f"COPY/ADD with flags: {args}")
    if len(parts) < 2:
        raise Unsupported(f"COPY/ADD without a destination: {args}")
    *patterns, dest = parts
    sources: list[Path] = []
    for pat in patterns:
        if "://" in pat:
            raise Unsupported(f"ADD from a URL: {pat}")
        found = sorted(glob.glob(str(task_dir / pat)))
        if not found:
            raise Unsupported(f"COPY source not in the task directory: {pat}")
        sources += [Path(f) for f in found]
    root = task_dir.resolve()
    for src in sources:
        if root != src.resolve() and root not in src.resolve().parents:
            raise Unsupported(f"COPY source outside the task directory: {src}")
    env_t = tuple(sorted(env.items()))
    archive = BuildStep.ARCHIVE
    single_file = len(sources) == 1 and sources[0].is_file()
    if single_file and not dest.endswith("/") and dest not in (".", ".."):
        # One file to a path that does not say whether it is a directory:
        # Docker decides by whether the destination exists as one.
        name = sources[0].name
        tar = _tar([(sources[0], name)])
        tmp = "/tmp/.forkeval-copy.d"
        run = (
            f"if [ -d {_q(dest)} ]; then tar -xf {archive} -C {_q(dest)}; "
            f"else mkdir -p -- \"$(dirname -- {_q(dest)})\" {tmp} && tar -xf {archive} -C {tmp} "
            f"&& mv -f -- {tmp}/{_q(name)} {_q(dest)} && rmdir {tmp}; fi && rm -f {archive}"
        )
        return BuildStep(run=run, tar=tar, cwd=cwd, env=env_t)
    # Into a directory: a source directory contributes its contents, a file itself.
    members: list[tuple[Path, str]] = []
    for src in sources:
        if src.is_dir():
            members += [(child, child.name) for child in sorted(src.iterdir())]
        else:
            members.append((src, src.name))
    run = f"mkdir -p -- {_q(dest)} && tar -xf {archive} -C {_q(dest)} && rm -f {archive}"
    return BuildStep(run=run, tar=_tar(members), cwd=cwd, env=env_t)


def parse_dockerfile(text: str, task_dir: Path, *, ignore_cmd: bool = False) -> tuple[str, list[BuildStep], str, dict[str, str]]:
    """(base image, build steps, final working directory, final environment)."""
    if re.search(r"<<-?\s*['\"]?\w", text):
        raise Unsupported("heredoc in the Dockerfile")
    image = ""
    steps: list[BuildStep] = []
    cwd = ""
    env: dict[str, str] = {}
    for line in _logical_lines(text):
        word, _, rest = line.partition(" ")
        word, rest = word.upper(), rest.strip()
        if word == "FROM":
            if image:
                raise Unsupported("more than one FROM (multi-stage build)")
            parts = [p for p in rest.split() if not p.startswith("--")]
            if not parts or len(parts) > 1 and parts[1].upper() != "AS":
                raise Unsupported(f"FROM {rest}")
            image = parts[0]
        elif not image:
            raise Unsupported(f"{word} before FROM")
        elif word == "WORKDIR":
            path = _expand(rest.strip("\"'"), env)
            if not path.startswith("/"):
                if not cwd:
                    raise Unsupported("relative WORKDIR with no earlier absolute one")
                path = f"{cwd.rstrip('/')}/{path}"
            steps.append(BuildStep(run=f"mkdir -p -- {_q(path)}", env=tuple(sorted(env.items()))))
            cwd = path
        elif word == "ENV":
            pairs = shlex.split(rest)
            if pairs and "=" not in pairs[0]:
                # The old form: ENV NAME the rest of the line is the value
                name, _, value = rest.partition(" ")
                env[name] = _expand(value.strip(), env)
            else:
                for pair in pairs:
                    name, eq, value = pair.partition("=")
                    if not eq:
                        raise Unsupported(f"ENV {rest}")
                    env[name] = _expand(value, env)
        elif word == "RUN":
            if rest.startswith("--"):
                raise Unsupported(f"RUN with flags: {rest[:60]}")
            if rest.startswith("["):
                try:
                    rest = shlex.join(json.loads(rest))
                except (ValueError, TypeError):
                    pass  # not the exec form after all: a shell command starting with [
            steps.append(BuildStep(run=rest, cwd=cwd, env=tuple(sorted(env.items()))))
        elif word in ("COPY", "ADD"):
            steps.append(_copy_step(task_dir, rest, cwd, env))
        elif word == "EXPOSE" or word == "LABEL" or (word == "CMD" and ignore_cmd):
            continue
        else:
            raise Unsupported(f"{word} in the Dockerfile")
    if not image:
        raise Unsupported("no FROM")
    return image, steps, cwd, env


def load_tbench_task(task_dir: Path, *, flavor: str = "medium", max_steps: int = 30, step_timeout: int = 180) -> Task:
    import yaml  # an optional dependency: only this adapter needs it

    def read(name: str) -> str:
        path = task_dir / name
        if not path.is_file():
            raise Unsupported(f"no {name}")
        return path.read_text()

    meta = yaml.safe_load(read("task.yaml")) or {}
    instruction = meta.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        raise Unsupported("task.yaml has no instruction")

    compose = yaml.safe_load(read("docker-compose.yaml")) or {}
    services = compose.get("services") or {}
    if len(services) != 1:
        raise Unsupported(f"{len(services)} services in docker-compose.yaml; one sandbox holds one")
    (service,) = services.values()
    extra = set(service) - {"build", "image", "container_name", "command", "environment", "volumes"}
    if extra:
        raise Unsupported(f"docker-compose.yaml sets {sorted(extra)}")
    # The benchmark starts the container with a sleep, so the image's own CMD never runs.
    sleeps = "sleep" in json.dumps(service.get("command", ""))

    image, steps, workdir, env = parse_dockerfile(read("Dockerfile"), task_dir, ignore_cmd=sleeps)

    test_files: dict[str, str | bytes] = {f"{TEST_DIR}/run-tests.sh": (task_dir / "run-tests.sh").read_bytes()
                                           if (task_dir / "run-tests.sh").is_file() else b""}
    if not test_files[f"{TEST_DIR}/run-tests.sh"]:
        raise Unsupported("no run-tests.sh")
    tests = task_dir / "tests"
    for path in sorted(tests.rglob("*")) if tests.is_dir() else []:
        if path.is_file():
            test_files[f"{TEST_DIR}/{path.relative_to(tests).as_posix()}"] = path.read_bytes()

    solution = ""
    if (task_dir / "solution.sh").is_file():
        solution = (task_dir / "solution.sh").read_text()
    elif (task_dir / "solution.yaml").is_file():
        # The other reference form: a list of commands typed one after another.
        entries = yaml.safe_load((task_dir / "solution.yaml").read_text()) or []
        solution = "\n".join(str(e["command"]) for e in entries if isinstance(e, dict) and e.get("command"))
    return Task(
        id=task_dir.name,
        instruction=instruction,
        image=image,
        flavor=flavor,
        build=tuple(steps),
        workdir=workdir,
        env=env,
        test=f"TEST_DIR={TEST_DIR} bash {TEST_DIR}/run-tests.sh 2>&1",
        test_files=test_files,
        test_parser="pytest",
        test_timeout=int(float(meta.get("max_test_timeout_sec") or 180)) + 120,
        max_steps=max_steps,
        step_timeout=step_timeout,
        solution=solution,
    )


def load_tbench_tasks(directory: Path, only: set[str] | None = None, **kw) -> tuple[list[Task], dict[str, str]]:
    """Every supported task under `directory`, and why each of the others was skipped."""
    tasks: list[Task] = []
    skipped: dict[str, str] = {}
    for task_dir in sorted(p for p in directory.iterdir() if p.is_dir()):
        if only is not None and task_dir.name not in only:
            continue
        try:
            tasks.append(load_tbench_task(task_dir, **kw))
        except Unsupported as e:
            skipped[task_dir.name] = str(e)
    return tasks, skipped
