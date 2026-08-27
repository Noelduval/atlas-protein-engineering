"""Append-only SQLite ledger with a human-inspectable JSONL event mirror."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterator

from atlas.adaptive.models import CandidateRecord, EvidenceRecord, FailureObservation


SCHEMA_VERSION = 1


class DuplicateCandidateError(ValueError):
    """A candidate sequence already exists in the scientific ledger."""


class LedgerIntegrityError(RuntimeError):
    """The SQLite source of truth and JSONL mirror disagree."""


class ScientificLedger:
    def __init__(self, database_path: Path, events_path: Path) -> None:
        self.database_path = database_path
        self.events_path = events_path
        self._connection = sqlite3.connect(database_path)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")

    @classmethod
    def create(cls, database_path: str | Path, events_path: str | Path) -> ScientificLedger:
        database = Path(database_path)
        events = Path(events_path)
        database.parent.mkdir(parents=True, exist_ok=True)
        events.parent.mkdir(parents=True, exist_ok=True)
        if database.exists() or events.exists():
            raise FileExistsError("ScientificLedger.create never overwrites existing state")
        ledger = cls(database, events)
        ledger._initialize_schema()
        events.touch()
        return ledger

    @classmethod
    def open(
        cls,
        database_path: str | Path,
        events_path: str | Path,
        *,
        synchronize_mirror: bool = True,
    ) -> ScientificLedger:
        database = Path(database_path)
        events = Path(events_path)
        if not database.is_file():
            raise FileNotFoundError(f"Scientific ledger does not exist: {database}")
        ledger = cls(database, events)
        version = ledger._connection.execute("PRAGMA user_version").fetchone()[0]
        if version != SCHEMA_VERSION:
            raise LedgerIntegrityError(
                f"Unsupported ledger schema {version}; expected {SCHEMA_VERSION}"
            )
        if synchronize_mirror:
            ledger._sync_jsonl()
        elif not events.exists():
            events.touch()
        return ledger

    def evidence_candidate_count(self) -> int:
        return int(
            self._connection.execute(
                "SELECT COUNT(DISTINCT candidate_id) FROM evidence"
            ).fetchone()[0]
        )

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> ScientificLedger:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _initialize_schema(self) -> None:
        self._connection.executescript(
            """
            PRAGMA user_version = 1;
            CREATE TABLE candidates (
                candidate_id TEXT PRIMARY KEY,
                sequence TEXT NOT NULL UNIQUE,
                record_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE evidence (
                evidence_id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT NOT NULL REFERENCES candidates(candidate_id),
                axis TEXT NOT NULL,
                record_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE failures (
                failure_id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT REFERENCES candidates(candidate_id),
                category TEXT NOT NULL,
                scope TEXT NOT NULL,
                record_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE decisions (
                decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT NOT NULL REFERENCES candidates(candidate_id),
                route TEXT NOT NULL CHECK(route IN ('REJECT', 'REVISE', 'PROMOTE')),
                rationale TEXT NOT NULL,
                round_index INTEGER NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE stages (
                stage_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                status TEXT NOT NULL,
                context_hash TEXT NOT NULL,
                artifact_hash TEXT,
                created_at TEXT NOT NULL
            );
            CREATE TABLE repair_proposals (
                repair_id INTEGER PRIMARY KEY AUTOINCREMENT,
                parent_id TEXT NOT NULL REFERENCES candidates(candidate_id),
                child_id TEXT NOT NULL UNIQUE REFERENCES candidates(candidate_id),
                proposal_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE repair_outcomes (
                outcome_id INTEGER PRIMARY KEY AUTOINCREMENT,
                parent_id TEXT NOT NULL REFERENCES candidates(candidate_id),
                child_id TEXT NOT NULL REFERENCES candidates(candidate_id),
                evidence_delta_json TEXT NOT NULL,
                disposition TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                previous_hash TEXT NOT NULL,
                event_hash TEXT NOT NULL UNIQUE
            );
            CREATE INDEX evidence_candidate_axis ON evidence(candidate_id, axis);
            CREATE INDEX failures_category_scope ON failures(category, scope);
            CREATE INDEX stages_resume ON stages(run_id, stage, context_hash, status);
            """
        )
        self._connection.commit()

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            yield self._connection
            self._connection.commit()
        except Exception:
            self._connection.rollback()
            raise

    def _append_event_in_transaction(
        self, connection: sqlite3.Connection, event_type: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        previous = connection.execute(
            "SELECT event_hash FROM events ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        previous_hash = "GENESIS" if previous is None else str(previous["event_hash"])
        timestamp = self._timestamp()
        payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        event_hash = hashlib.sha256(
            f"{previous_hash}|{timestamp}|{event_type}|{payload_json}".encode("utf-8")
        ).hexdigest()
        cursor = connection.execute(
            "INSERT INTO events(timestamp, event_type, payload_json, previous_hash, event_hash) "
            "VALUES (?, ?, ?, ?, ?)",
            (timestamp, event_type, payload_json, previous_hash, event_hash),
        )
        return {
            "sequence": int(cursor.lastrowid),
            "timestamp": timestamp,
            "event_type": event_type,
            "payload": payload,
            "previous_hash": previous_hash,
            "event_hash": event_hash,
        }

    def _mirror_event(self, event: dict[str, Any]) -> None:
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")

    def _mirror_events(self, events: list[dict[str, Any]]) -> None:
        if not events:
            return
        with self.events_path.open("a", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event, sort_keys=True) + "\n")

    def _sync_jsonl(self) -> None:
        if not self.events_path.exists():
            self.events_path.touch()
        lines = [line for line in self.events_path.read_text().splitlines() if line.strip()]
        rows = self._connection.execute("SELECT * FROM events ORDER BY sequence").fetchall()
        if len(lines) > len(rows):
            raise LedgerIntegrityError("JSONL contains events absent from SQLite source of truth")
        for index, line in enumerate(lines):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LedgerIntegrityError(f"Invalid JSONL event at line {index + 1}") from exc
            if event.get("event_hash") != rows[index]["event_hash"]:
                raise LedgerIntegrityError(f"JSONL diverges from SQLite at event {index + 1}")
        for row in rows[len(lines) :]:
            self._mirror_event(self._event_from_row(row))

    @staticmethod
    def _event_from_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "sequence": int(row["sequence"]),
            "timestamp": row["timestamp"],
            "event_type": row["event_type"],
            "payload": json.loads(row["payload_json"]),
            "previous_hash": row["previous_hash"],
            "event_hash": row["event_hash"],
        }

    def record_event(self, event_type: str, payload: dict[str, Any]) -> None:
        """Append a non-candidate scientific/orchestration observation."""
        if not event_type.strip():
            raise ValueError("event_type is required")
        with self._transaction() as connection:
            event = self._append_event_in_transaction(connection, event_type, payload)
        self._mirror_event(event)

    def add_candidate(self, candidate: CandidateRecord) -> None:
        payload = candidate.to_dict()
        try:
            with self._transaction() as connection:
                connection.execute(
                    "INSERT INTO candidates(candidate_id, sequence, record_json, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        candidate.candidate_id,
                        candidate.sequence,
                        json.dumps(payload, sort_keys=True),
                        self._timestamp(),
                    ),
                )
                event = self._append_event_in_transaction(
                    connection,
                    "candidate_added",
                    {"candidate_id": candidate.candidate_id, "record": payload},
                )
        except sqlite3.IntegrityError as exc:
            raise DuplicateCandidateError(
                f"Candidate sequence already exists: {candidate.candidate_id}"
            ) from exc
        self._mirror_event(event)

    def add_candidates(self, candidates: tuple[CandidateRecord, ...]) -> int:
        """Persist a candidate batch atomically while retaining one event per record."""
        if not candidates:
            return 0
        events: list[dict[str, Any]] = []
        try:
            with self._transaction() as connection:
                for candidate in candidates:
                    payload = candidate.to_dict()
                    connection.execute(
                        "INSERT INTO candidates(candidate_id, sequence, record_json, created_at) "
                        "VALUES (?, ?, ?, ?)",
                        (
                            candidate.candidate_id,
                            candidate.sequence,
                            json.dumps(payload, sort_keys=True),
                            self._timestamp(),
                        ),
                    )
                    events.append(
                        self._append_event_in_transaction(
                            connection,
                            "candidate_added",
                            {"candidate_id": candidate.candidate_id, "record": payload},
                        )
                    )
        except sqlite3.IntegrityError as exc:
            raise DuplicateCandidateError(
                "Candidate batch contains an existing ID or sequence"
            ) from exc
        self._mirror_events(events)
        return len(candidates)

    def get_candidate(self, candidate_id: str) -> CandidateRecord | None:
        row = self._connection.execute(
            "SELECT record_json FROM candidates WHERE candidate_id = ?", (candidate_id,)
        ).fetchone()
        return None if row is None else CandidateRecord.from_dict(json.loads(row[0]))

    def candidate_count(self) -> int:
        return int(self._connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0])

    def candidates(self) -> tuple[CandidateRecord, ...]:
        rows = self._connection.execute(
            "SELECT record_json FROM candidates ORDER BY rowid"
        ).fetchall()
        return tuple(CandidateRecord.from_dict(json.loads(row[0])) for row in rows)

    def add_evidence(self, evidence: EvidenceRecord) -> None:
        payload = evidence.to_dict()
        with self._transaction() as connection:
            connection.execute(
                "INSERT INTO evidence(candidate_id, axis, record_json, created_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    evidence.candidate_id,
                    evidence.axis.value,
                    json.dumps(payload, sort_keys=True),
                    self._timestamp(),
                ),
            )
            event = self._append_event_in_transaction(
                connection,
                "evidence_added",
                {"candidate_id": evidence.candidate_id, "record": payload},
            )
        self._mirror_event(event)

    def add_evidence_many(self, records: tuple[EvidenceRecord, ...]) -> None:
        """Persist an evidence batch atomically while preserving append order."""
        if not records:
            return
        events: list[dict[str, Any]] = []
        with self._transaction() as connection:
            for evidence in records:
                payload = evidence.to_dict()
                connection.execute(
                    "INSERT INTO evidence(candidate_id, axis, record_json, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        evidence.candidate_id,
                        evidence.axis.value,
                        json.dumps(payload, sort_keys=True),
                        self._timestamp(),
                    ),
                )
                events.append(
                    self._append_event_in_transaction(
                        connection,
                        "evidence_added",
                        {"candidate_id": evidence.candidate_id, "record": payload},
                    )
                )
        self._mirror_events(events)

    def evidence_for(self, candidate_id: str) -> tuple[EvidenceRecord, ...]:
        rows = self._connection.execute(
            "SELECT record_json FROM evidence WHERE candidate_id = ? ORDER BY evidence_id",
            (candidate_id,),
        ).fetchall()
        return tuple(EvidenceRecord.from_dict(json.loads(row[0])) for row in rows)

    def all_evidence(self) -> tuple[EvidenceRecord, ...]:
        rows = self._connection.execute(
            "SELECT record_json FROM evidence ORDER BY evidence_id"
        ).fetchall()
        return tuple(EvidenceRecord.from_dict(json.loads(row[0])) for row in rows)

    def all_decisions(self) -> tuple[dict[str, Any], ...]:
        rows = self._connection.execute(
            "SELECT candidate_id, route, rationale, round_index, created_at "
            "FROM decisions ORDER BY decision_id"
        ).fetchall()
        return tuple(dict(row) for row in rows)

    def stage_records(self, run_id: str) -> tuple[dict[str, Any], ...]:
        rows = self._connection.execute(
            "SELECT stage, status, context_hash, artifact_hash, created_at "
            "FROM stages WHERE run_id = ? ORDER BY stage_id",
            (run_id,),
        ).fetchall()
        return tuple(dict(row) for row in rows)

    def record_failure(self, failure: FailureObservation) -> None:
        payload = failure.to_dict()
        with self._transaction() as connection:
            connection.execute(
                "INSERT INTO failures(candidate_id, category, scope, record_json, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    failure.candidate_id,
                    failure.category,
                    failure.scope,
                    json.dumps(payload, sort_keys=True),
                    self._timestamp(),
                ),
            )
            event = self._append_event_in_transaction(
                connection, "failure_recorded", {"record": payload}
            )
        self._mirror_event(event)

    def failures(self, *, category: str | None = None) -> tuple[FailureObservation, ...]:
        if category is None:
            rows = self._connection.execute(
                "SELECT record_json FROM failures ORDER BY failure_id"
            ).fetchall()
        else:
            rows = self._connection.execute(
                "SELECT record_json FROM failures WHERE category = ? ORDER BY failure_id",
                (category,),
            ).fetchall()
        return tuple(FailureObservation(**json.loads(row[0])) for row in rows)

    def record_decision(
        self,
        *,
        candidate_id: str,
        route: str,
        rationale: str,
        round_index: int,
    ) -> None:
        route = route.upper()
        if route not in {"REJECT", "REVISE", "PROMOTE"}:
            raise ValueError(f"Unknown decision route: {route}")
        payload = {
            "candidate_id": candidate_id,
            "route": route,
            "rationale": rationale,
            "round_index": round_index,
        }
        with self._transaction() as connection:
            connection.execute(
                "INSERT INTO decisions(candidate_id, route, rationale, round_index, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (candidate_id, route, rationale, round_index, self._timestamp()),
            )
            event = self._append_event_in_transaction(
                connection, "decision_recorded", payload
            )
        self._mirror_event(event)

    def decisions_for(self, candidate_id: str) -> tuple[dict[str, Any], ...]:
        rows = self._connection.execute(
            "SELECT candidate_id, route, rationale, round_index, created_at "
            "FROM decisions WHERE candidate_id = ? ORDER BY decision_id",
            (candidate_id,),
        ).fetchall()
        return tuple(dict(row) for row in rows)

    def mark_stage(
        self,
        *,
        run_id: str,
        stage: str,
        status: str,
        context_hash: str,
        artifact_hash: str | None,
    ) -> None:
        payload = {
            "run_id": run_id,
            "stage": stage,
            "status": status,
            "context_hash": context_hash,
            "artifact_hash": artifact_hash,
        }
        with self._transaction() as connection:
            connection.execute(
                "INSERT INTO stages(run_id, stage, status, context_hash, artifact_hash, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, stage, status, context_hash, artifact_hash, self._timestamp()),
            )
            event = self._append_event_in_transaction(
                connection, "stage_recorded", payload
            )
        self._mirror_event(event)

    def stage_completed(self, run_id: str, stage: str, context_hash: str) -> bool:
        row = self._connection.execute(
            "SELECT status FROM stages WHERE run_id = ? AND stage = ? AND context_hash = ? "
            "ORDER BY stage_id DESC LIMIT 1",
            (run_id, stage, context_hash),
        ).fetchone()
        return row is not None and row["status"] == "completed"

    def record_repair(self, proposal: Any) -> None:
        payload = proposal.to_dict()
        with self._transaction() as connection:
            connection.execute(
                "INSERT INTO repair_proposals(parent_id, child_id, proposal_json, created_at) "
                "VALUES (?, ?, ?, ?)",
                (
                    payload["parent_id"],
                    payload["child_id"],
                    json.dumps(payload, sort_keys=True),
                    self._timestamp(),
                ),
            )
            event = self._append_event_in_transaction(
                connection, "repair_proposed", payload
            )
        self._mirror_event(event)

    def record_repair_outcome(
        self,
        *,
        parent_id: str,
        child_id: str,
        evidence_delta: dict[str, float],
        disposition: str,
    ) -> None:
        payload = {
            "parent_id": parent_id,
            "child_id": child_id,
            "evidence_delta": evidence_delta,
            "disposition": disposition,
        }
        with self._transaction() as connection:
            connection.execute(
                "INSERT INTO repair_outcomes(parent_id, child_id, evidence_delta_json, "
                "disposition, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    parent_id,
                    child_id,
                    json.dumps(evidence_delta, sort_keys=True),
                    disposition,
                    self._timestamp(),
                ),
            )
            event = self._append_event_in_transaction(
                connection, "repair_evaluated", payload
            )
        self._mirror_event(event)

    def repair_trajectory(self, parent_id: str) -> tuple[dict[str, Any], ...]:
        rows = self._connection.execute(
            """
            SELECT p.proposal_json, o.evidence_delta_json, o.disposition
            FROM repair_proposals p
            LEFT JOIN repair_outcomes o ON o.child_id = p.child_id
            WHERE p.parent_id = ?
            ORDER BY p.repair_id, o.outcome_id
            """,
            (parent_id,),
        ).fetchall()
        trajectory: list[dict[str, Any]] = []
        for row in rows:
            record = json.loads(row["proposal_json"])
            record["evidence_delta"] = (
                {} if row["evidence_delta_json"] is None else json.loads(row["evidence_delta_json"])
            )
            record["disposition"] = row["disposition"]
            trajectory.append(record)
        return tuple(trajectory)

    def delete_candidate(self, candidate_id: str) -> None:
        del candidate_id
        raise RuntimeError("ScientificLedger is append-only; candidates cannot be deleted")
