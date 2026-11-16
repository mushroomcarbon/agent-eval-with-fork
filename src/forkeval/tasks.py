"""A task: an environment, an instruction, and a test that decides pass or fail."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class BuildStep:
    """One step of preparing a task's environment, baked into an image once:
    a command, optionally with an archive placed at `ARCHIVE` first (how a set
    of files reaches the sandbox with its modes intact; the command unpacks it)."""

    ARCHIVE = "/tmp/.forkeval-copy.tar"

    run: str = ""
    tar: bytes = b""
    #: Working directory and environment the step runs with ("" is the image's own).
    cwd: str = ""
    env: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class Task:
    id: str
    instruction: str
    #: Shell command; exit status 0 means the task is solved.
    test: str
    image: str = "default"
    flavor: str = "small"
    #: Shell command run once in a fresh sandbox, before the agent's first step.
    setup: str = ""
    #: Files written into the sandbox before setup (path -> content).
    files: dict[str, str] = field(default_factory=dict)
    #: Files written only when scoring, so the agent never sees them.
    test_files: dict[str, str | bytes] = field(default_factory=dict)
    #: How the test's result is read: "exit" (status 0 passes) or "pytest"
    #: (the short summary must list at least one PASSED and no FAILED or ERROR).
    test_parser: str = "exit"
    #: Steps applied to `image` once and saved as an image of their own; every
    #: run of the task then starts from that image. Empty: runs start from `image`.
    build: tuple[BuildStep, ...] = ()
    #: Working directory and environment of every command: setup, the agent's, the test.
    workdir: str = ""
    env: dict[str, str] = field(default_factory=dict)
    #: A command known to solve the task. Never shown to the agent; used to check the task itself.
    solution: str = ""
    max_steps: int = 30
    step_timeout: int = 120
    test_timeout: int = 120
    #: Sandbox lifetime in seconds. Branch points inherit it, so it has to
    #: cover the trunk and every continuation taken from it.
    ttl: int = 7200


def load_task(path: Path) -> Task:
    with path.open("rb") as f:
        raw = tomllib.load(f)
    raw.setdefault("id", path.stem)
    known = set(Task.__dataclass_fields__)
    unknown = set(raw) - known
    if unknown:
        raise ValueError(f"{path}: unknown task field(s) {sorted(unknown)}")
    return Task(**raw)


def load_tasks(directory: Path) -> list[Task]:
    tasks = [load_task(p) for p in sorted(directory.glob("*.toml"))]
    if not tasks:
        raise ValueError(f"no *.toml tasks in {directory}")
    ids = [t.id for t in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate task ids in {directory}")
    return tasks
