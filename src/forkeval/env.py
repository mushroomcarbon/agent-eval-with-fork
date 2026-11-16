"""The environment an agent acts on: one sandbox, and the operations the harness needs from it.

`Box` is the whole surface. `SandboxBox` implements it on the platform's public
SDK; tests implement it in memory. Nothing else in the harness imports the SDK.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ExecOutput:
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


class Box(Protocol):
    id: str

    async def run(self, command: str, *, timeout: int, cwd: str = "", env: dict[str, str] | None = None) -> ExecOutput: ...
    async def write(self, path: str, content: str | bytes) -> None: ...
    async def fork(self, count: int) -> list["Box"]: ...
    async def suspend(self) -> None: ...
    async def delete(self) -> None: ...
    async def commit(self, name: str, tag: str) -> str:
        """Save the filesystem as an image; returns the id a create takes."""
        ...


class BoxFactory(Protocol):
    async def create(self, *, image: str, flavor: str, ttl: int, tags: dict[str, str]) -> Box: ...
    async def has_image(self, image_id: str) -> bool: ...


# --- the platform ----------------------------------------------------------

#: How long a refused fork or create is retried before the refusal is an error.
#: A full node or a reached quota is the harness's own load, not a failure of
#: the run, so it waits for its other sandboxes to finish.
CAPACITY_WAIT_SECONDS = 600


async def _retry_capacity(op, what: str):
    from zillion_asb import (  # imported here so tests need no SDK
        NodeAtCapacityError,
        QuotaExceededError,
        SandboxNotRunningError,
        SandboxRestoreFailedError,
    )

    deadline = time.monotonic() + CAPACITY_WAIT_SECONDS
    delay = 2.0
    while True:
        try:
            return await op()
        except (NodeAtCapacityError, QuotaExceededError, SandboxRestoreFailedError, SandboxNotRunningError) as e:
            if time.monotonic() + delay > deadline:
                raise RuntimeError(f"{what}: still refused after {CAPACITY_WAIT_SECONDS}s: {e}") from e
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, 20.0)


class SandboxBox:
    """A sandbox on the platform, through the customer SDK."""

    def __init__(self, sandbox) -> None:
        self._sb = sandbox
        self.id: str = sandbox.id

    async def run(self, command: str, *, timeout: int, cwd: str = "", env: dict[str, str] | None = None) -> ExecOutput:
        from zillion_asb import ExecTimeoutError

        try:
            r = await self._sb.run(command, timeout=timeout, cwd=cwd or None, env=env or None)
        except ExecTimeoutError as e:
            return ExecOutput(None, getattr(e, "stdout", "") or "", getattr(e, "stderr", "") or "", timed_out=True)
        return ExecOutput(r.exit_code, r.stdout, r.stderr)

    async def write(self, path: str, content: str | bytes) -> None:
        await self._sb.files.write(path, content)

    async def commit(self, name: str, tag: str) -> str:
        image = await self._sb.commit(name, tag, wait=True, timeout=900)
        return image.image_id

    async def fork(self, count: int) -> list["SandboxBox"]:
        forks = await _retry_capacity(lambda: self._sb.fork(count), f"fork {self.id}")
        return [SandboxBox(f) for f in forks]

    async def suspend(self) -> None:
        await self._sb.suspend()

    async def delete(self) -> None:
        from zillion_asb import NotFoundError

        try:
            await self._sb.delete()
        except NotFoundError:
            pass


class SandboxFactory:
    async def create(self, *, image: str, flavor: str, ttl: int, tags: dict[str, str]) -> SandboxBox:
        from zillion_asb import AsyncSandbox

        sb = await _retry_capacity(
            lambda: AsyncSandbox.create(image, flavor, ttl=ttl, tags=tags), "create sandbox"
        )
        return SandboxBox(sb)

    async def has_image(self, image_id: str) -> bool:
        from zillion_asb import AsyncImage

        return any(i.image_id == image_id and i.status == "ready" for i in await AsyncImage.list())
