"""An in-memory Box: state is a dict, fork is a deep copy."""

from __future__ import annotations

import copy
import itertools

from forkeval.env import ExecOutput

_ids = itertools.count()


class FakeBox:
    """Understands the gates task's commands, which is all the runner tests need."""

    live: set[str] = set()

    def __init__(self, state: dict | None = None) -> None:
        self.id = f"fake-{next(_ids)}"
        self.state = state if state is not None else {"gates": {}, "files": {}}
        self.suspended = False
        FakeBox.live.add(self.id)

    async def run(self, command: str, *, timeout: int, cwd: str = "", env: dict | None = None) -> ExecOutput:
        assert not self.suspended, "command sent to a suspended box"
        self.state.setdefault("commands", []).append((command, cwd, dict(env or {})))
        if command.startswith("fail"):
            return ExecOutput(1, "", "boom")
        parts = command.split()
        if "set" in parts:
            i = parts.index("set")
            self.state["gates"][int(parts[i + 1])] = parts[i + 2]
            return ExecOutput(0, "ok\n", "")
        if "check" in parts:
            want = int(parts[parts.index("check") + 1])
            g = self.state["gates"]
            good = len(g) == want and all(g.get(i) == "ok" for i in range(want))
            return ExecOutput(0 if good else 1, "", "")
        return ExecOutput(0, "", "")

    async def write(self, path: str, content: str | bytes) -> None:
        self.state["files"][path] = content

    async def commit(self, name: str, tag: str) -> str:
        FakeFactory.images[f"{name}:{tag}"] = copy.deepcopy(self.state)
        return f"{name}:{tag}"

    async def fork(self, count: int) -> list["FakeBox"]:
        return [FakeBox(copy.deepcopy(self.state)) for _ in range(count)]

    async def suspend(self) -> None:
        self.suspended = True

    async def delete(self) -> None:
        FakeBox.live.discard(self.id)


class FakeFactory:
    #: image id -> the state it was committed with
    images: dict[str, dict] = {}
    created: list[tuple[str, str]] = []

    async def create(self, *, image: str, flavor: str, ttl: int, tags: dict[str, str]) -> FakeBox:
        FakeFactory.created.append((image, tags.get("kind", "")))
        if image in FakeFactory.images:
            return FakeBox(copy.deepcopy(FakeFactory.images[image]))
        return FakeBox()

    async def has_image(self, image_id: str) -> bool:
        return image_id in FakeFactory.images
