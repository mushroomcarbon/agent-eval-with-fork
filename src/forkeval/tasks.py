"""A task: an environment, an instruction, and a test that decides pass or fail."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


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
    test_files: dict[str, str] = field(default_factory=dict)
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
