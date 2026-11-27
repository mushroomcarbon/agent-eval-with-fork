"""Where results go: one SQLite file, and one JSONL transcript per run."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from pathlib import Path

from .policy import Message, Usage

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id            TEXT PRIMARY KEY,
    experiment    TEXT NOT NULL,
    task_id       TEXT NOT NULL,
    policy        TEXT NOT NULL,
    -- trunk: a full run that branch points are taken from
    -- cont:  a continuation of a branch point
    -- fresh: a full run in a sandbox of its own, no fork involved
    kind          TEXT NOT NULL CHECK (kind IN ('trunk', 'cont', 'fresh')),
    branch_point  TEXT REFERENCES branch_points(id),
    sandbox_id    TEXT,
    started_at    REAL NOT NULL,
    ended_at      REAL,
    -- 1 pass, 0 fail, NULL not scored (still running, or it errored)
    outcome       INTEGER,
    steps         INTEGER NOT NULL DEFAULT 0,
    stop          TEXT,
    error         TEXT,
    input_tokens       INTEGER NOT NULL DEFAULT 0,
    output_tokens      INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens  INTEGER NOT NULL DEFAULT 0,
    cache_write_tokens INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS branch_points (
    id          TEXT PRIMARY KEY,
    trunk       TEXT NOT NULL REFERENCES runs(id),
    -- the state after this many tool results; 0 is before the first model call
    step        INTEGER NOT NULL,
    sandbox_id  TEXT,
    fork_ms     INTEGER,
    suspend_ms  INTEGER,
    -- for a policy with a known answer: the true P(success) of this state
    true_value  REAL,
    created_at  REAL NOT NULL
);
-- a judge model's answer to "which step decided this failed run?"
CREATE TABLE IF NOT EXISTS judgments (
    trunk       TEXT NOT NULL REFERENCES runs(id),
    judge       TEXT NOT NULL,
    step        INTEGER,
    reason      TEXT,
    steps_shown INTEGER NOT NULL,
    created_at  REAL NOT NULL,
    PRIMARY KEY (trunk, judge)
);
CREATE INDEX IF NOT EXISTS runs_experiment ON runs (experiment, task_id, kind);
CREATE INDEX IF NOT EXISTS runs_bp ON runs (branch_point);
"""


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


class Store:
    def __init__(self, directory: Path) -> None:
        self.dir = directory
        (directory / "transcripts").mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(directory / "forkeval.db", isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        if "cost_usd" not in {r["name"] for r in self.db.execute("PRAGMA table_info(runs)")}:
            self.db.execute("ALTER TABLE runs ADD COLUMN cost_usd REAL NOT NULL DEFAULT 0")
        if "skipped" not in {r["name"] for r in self.db.execute("PRAGMA table_info(branch_points)")}:
            # why no continuations were run from this state: runner.SETTLED, NO_ROOM, OVER_LIMIT, LOST, UNNEEDED
            self.db.execute("ALTER TABLE branch_points ADD COLUMN skipped INTEGER NOT NULL DEFAULT 0")
        if "introduced" not in {r["name"] for r in self.db.execute("PRAGMA table_info(judgments)")}:
            # prompt version 2: the step where the judge says the failing mistake was made (`step` is where it
            # became unrecoverable)
            self.db.execute("ALTER TABLE judgments ADD COLUMN introduced INTEGER")
        if "kept" not in {r["name"] for r in self.db.execute("PRAGMA table_info(branch_points)")}:
            # 1: the snapshot was left on the platform when a budget stopped the run, to be measured later
            self.db.execute("ALTER TABLE branch_points ADD COLUMN kept INTEGER NOT NULL DEFAULT 0")

    # -- runs ---------------------------------------------------------------

    def start_run(self, *, experiment: str, task_id: str, policy: str, kind: str,
                  branch_point: str | None = None, sandbox_id: str | None = None) -> str:
        run_id = new_id("run")
        self.db.execute(
            "INSERT INTO runs (id, experiment, task_id, policy, kind, branch_point, sandbox_id, started_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, experiment, task_id, policy, kind, branch_point, sandbox_id, time.time()),
        )
        return run_id

    def add_usage(self, run_id: str, u: Usage) -> None:
        self.db.execute(
            "UPDATE runs SET input_tokens = input_tokens + ?, output_tokens = output_tokens + ?, "
            "cache_read_tokens = cache_read_tokens + ?, cache_write_tokens = cache_write_tokens + ?, "
            "cost_usd = cost_usd + ? WHERE id = ?",
            (u.input_tokens, u.output_tokens, u.cache_read_tokens, u.cache_write_tokens, u.cost_usd, run_id),
        )

    def end_run(self, run_id: str, *, outcome: bool | None, steps: int, stop: str | None,
                error: str | None = None) -> None:
        self.db.execute(
            "UPDATE runs SET ended_at = ?, outcome = ?, steps = ?, stop = ?, error = ? WHERE id = ?",
            (time.time(), None if outcome is None else int(outcome), steps, stop, error, run_id),
        )

    def save_transcript(self, run_id: str, messages: list[Message]) -> None:
        """The messages of a run, or (under a branch point's id) of the state it holds."""
        path = self.dir / "transcripts" / f"{run_id}.jsonl"
        with path.open("w") as f:
            for m in messages:
                f.write(json.dumps(m, ensure_ascii=False) + "\n")

    def load_transcript(self, run_id: str) -> list[Message]:
        path = self.dir / "transcripts" / f"{run_id}.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines() if line]

    # -- branch points --------------------------------------------------------

    def add_branch_point(self, *, trunk: str, step: int, sandbox_id: str | None, fork_ms: int | None,
                         suspend_ms: int | None, true_value: float | None, skipped: int = 0) -> str:
        bp_id = new_id("bp")
        self.db.execute(
            "INSERT INTO branch_points (id, trunk, step, sandbox_id, fork_ms, suspend_ms, true_value, created_at, "
            "skipped) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (bp_id, trunk, step, sandbox_id, fork_ms, suspend_ms, true_value, time.time(), skipped),
        )
        return bp_id

    def skip_branch_point(self, bp_id: str, why: int = 1) -> None:
        self.db.execute("UPDATE branch_points SET skipped = ?, kept = 0 WHERE id = ?", (why, bp_id))

    def keep_point(self, bp_id: str, kept: bool) -> None:
        self.db.execute("UPDATE branch_points SET kept = ? WHERE id = ?", (int(kept), bp_id))

    def points(self, trunk: str) -> list[sqlite3.Row]:
        """The branch points of one trunk, in step order."""
        return self.db.execute(
            "SELECT id, step, sandbox_id, skipped, kept FROM branch_points WHERE trunk = ? ORDER BY step", (trunk,)
        ).fetchall()

    def kept(self, experiment: str) -> list[sqlite3.Row]:
        """Snapshots left on the platform by a stop: one row each, newest trunk first."""
        return self.db.execute(
            "SELECT t.task_id, bp.trunk, bp.id, bp.step, bp.sandbox_id FROM branch_points bp "
            "JOIN runs t ON t.id = bp.trunk WHERE t.experiment = ? AND bp.kept = 1 "
            "ORDER BY t.started_at DESC, bp.step", (experiment,)
        ).fetchall()

    # -- reading ----------------------------------------------------------------

    def run_outcome(self, run_id: str) -> bool | None:
        row = self.db.execute("SELECT outcome FROM runs WHERE id = ?", (run_id,)).fetchone()
        return None if row is None or row["outcome"] is None else bool(row["outcome"])

    def point_result(self, bp_id: str, policy: str | None = None) -> tuple[int, int]:
        """(passes, scored continuations) of one branch point; with `policy`, only that policy's continuations."""
        row = self.db.execute("SELECT COALESCE(SUM(outcome), 0) AS x, COUNT(outcome) AS n FROM runs "
                              "WHERE branch_point = ? AND kind = 'cont' AND (? IS NULL OR policy = ?)",
                              (bp_id, policy, policy)).fetchone()
        return int(row["x"]), int(row["n"])

    def start_result(self, experiment: str, task_id: str, policy: str) -> tuple[int, int]:
        """(passes, scored continuations) of one policy from the step-0 states of a task's trunks:
        every trunk of a task starts from the same state, so one set of these is enough per task."""
        row = self.db.execute(
            "SELECT COALESCE(SUM(c.outcome), 0) AS x, COUNT(c.outcome) AS n FROM runs c "
            "JOIN branch_points bp ON bp.id = c.branch_point JOIN runs t ON t.id = bp.trunk "
            "WHERE t.experiment = ? AND t.task_id = ? AND bp.step = 0 AND c.kind = 'cont' AND c.policy = ?",
            (experiment, task_id, policy)).fetchone()
        return int(row["x"]), int(row["n"])

    def by_policy(self, experiment: str) -> list[sqlite3.Row]:
        """Per branch point and continuation policy: passes and count."""
        return self.db.execute(
            """
            SELECT t.task_id, bp.trunk, bp.id AS branch_point, bp.step, c.policy,
                   COUNT(c.outcome) AS n, COALESCE(SUM(c.outcome), 0) AS passes
              FROM branch_points bp JOIN runs t ON t.id = bp.trunk
              JOIN runs c ON c.branch_point = bp.id AND c.kind = 'cont'
             WHERE t.experiment = ?
             GROUP BY bp.id, c.policy
             ORDER BY t.task_id, t.started_at, bp.step
            """, (experiment,)).fetchall()

    def trunks(self, experiment: str) -> list[sqlite3.Row]:
        """Finished trunks, oldest first."""
        return self.db.execute(
            "SELECT id, experiment, task_id, policy, outcome, steps, stop FROM runs "
            "WHERE experiment = ? AND kind = 'trunk' AND outcome IS NOT NULL ORDER BY started_at",
            (experiment,)).fetchall()

    # -- judgments ----------------------------------------------------------------

    def add_judgment(self, trunk: str, judge: str, step: int | None, reason: str, steps_shown: int,
                     introduced: int | None = None) -> None:
        self.db.execute("INSERT OR REPLACE INTO judgments (trunk, judge, step, reason, steps_shown, created_at, "
                        "introduced) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (trunk, judge, step, reason, steps_shown, time.time(), introduced))

    def continuations_at(self, trunk: str, step: int, policy: str) -> list[tuple[int, bool]]:
        """Scored continuations of a trunk's state at `step` by `policy`: (steps the run ended at, passed)."""
        return [(r[0], bool(r[1])) for r in self.db.execute(
            "SELECT r.steps, r.outcome FROM runs r JOIN branch_points b ON r.branch_point = b.id "
            "WHERE b.trunk = ? AND b.step = ? AND r.kind = 'cont' AND r.policy = ? AND r.outcome IS NOT NULL",
            (trunk, step, policy))]

    def judgments(self, trunk: str) -> dict[str, sqlite3.Row]:
        return {r["judge"]: r for r in self.db.execute("SELECT * FROM judgments WHERE trunk = ?", (trunk,))}

    def tokens_used(self, experiment: str) -> int:
        row = self.db.execute(
            "SELECT COALESCE(SUM(input_tokens + output_tokens + cache_read_tokens + cache_write_tokens), 0) "
            "FROM runs WHERE experiment = ?", (experiment,),
        ).fetchone()
        return int(row[0])

    def cost_usd(self, experiment: str) -> float:
        return float(self.db.execute("SELECT COALESCE(SUM(cost_usd), 0) FROM runs WHERE experiment = ?",
                                     (experiment,)).fetchone()[0])

    def states(self, experiment: str) -> list[sqlite3.Row]:
        """One row per branch point: its continuations' passes and count."""
        return self.db.execute(
            """
            SELECT t.task_id, bp.trunk, bp.id AS branch_point, bp.step, bp.true_value,
                   t.outcome AS trunk_outcome, t.steps AS trunk_steps,
                   COUNT(c.outcome) AS n, COALESCE(SUM(c.outcome), 0) AS passes,
                   SUM(c.id IS NOT NULL AND c.outcome IS NULL) AS unscored,
                   bp.fork_ms, bp.suspend_ms, bp.skipped
              FROM branch_points bp
              JOIN runs t ON t.id = bp.trunk
              LEFT JOIN runs c ON c.branch_point = bp.id AND c.kind = 'cont' AND c.policy = t.policy
             WHERE t.experiment = ?
             GROUP BY bp.id
             ORDER BY t.task_id, t.started_at, bp.step
            """,
            (experiment,),
        ).fetchall()

    def fresh(self, experiment: str) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT task_id, COUNT(outcome) AS n, COALESCE(SUM(outcome), 0) AS passes, "
            "SUM(outcome IS NULL) AS unscored FROM runs WHERE experiment = ? AND kind = 'fresh' GROUP BY task_id",
            (experiment,),
        ).fetchall()

    def done(self, experiment: str) -> dict[str, tuple[int, int]]:
        """Per task: (complete trunks, scored fresh runs). A trunk is complete when it was
        scored itself and every one of its branch points has at least two scored continuations
        (or was skipped), with none still kept on the platform waiting to be measured."""
        trunks = dict(self.db.execute(
            """
            SELECT t.task_id, COUNT(*) FROM runs t
             WHERE t.experiment = ? AND t.kind = 'trunk' AND t.outcome IS NOT NULL
               AND NOT EXISTS (
                   SELECT 1 FROM branch_points bp
                    WHERE bp.trunk = t.id AND bp.skipped = 0
                      AND (bp.kept = 1
                           OR (SELECT COUNT(c.outcome) FROM runs c WHERE c.branch_point = bp.id) < 2))
             GROUP BY t.task_id
            """, (experiment,)).fetchall())
        fresh = dict(self.db.execute(
            "SELECT task_id, COUNT(outcome) FROM runs WHERE experiment = ? AND kind = 'fresh' GROUP BY task_id",
            (experiment,)).fetchall())
        return {task: (trunks.get(task, 0), fresh.get(task, 0)) for task in set(trunks) | set(fresh)}
