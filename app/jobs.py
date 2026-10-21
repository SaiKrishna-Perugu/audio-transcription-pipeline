"""Durable job queue for asynchronous transcription.

- Jobs and results live in SQLite (swap for Postgres in production; same schema).
- A small worker pool pulls queued jobs; uploads never block on transcription.
- Idempotency: the job id is derived from the audio's SHA-256 plus options, so
  re-uploading the same file returns the existing job instead of redoing the work.
- Retries: transient failures are retried with exponential backoff up to max_attempts;
  bad input (AudioError) fails immediately because retrying cannot fix it.
- Recovery: on startup, jobs left 'running' by a crashed process are re-queued.
"""
from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .audio import AudioError
from .pipeline import transcribe_file
from .transcriber import Transcriber

log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id           TEXT PRIMARY KEY,
    status       TEXT NOT NULL,          -- queued | running | succeeded | failed
    filename     TEXT NOT NULL,
    audio_path   TEXT NOT NULL,
    options      TEXT NOT NULL,
    attempts     INTEGER NOT NULL DEFAULT 0,
    error        TEXT,
    result_path  TEXT,
    next_run_at  REAL NOT NULL,
    created_at   REAL NOT NULL,
    updated_at   REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, next_run_at);
"""


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class JobStore:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._lock = threading.Lock()

    def create(self, job_id: str, filename: str, audio_path: Path, options: dict) -> tuple[dict, bool]:
        now = time.time()
        with self._lock:
            existing = self.get(job_id)
            if existing and existing["status"] != "failed":
                return existing, False
            self._conn.execute(
                "INSERT OR REPLACE INTO jobs (id,status,filename,audio_path,options,attempts,error,result_path,"
                "next_run_at,created_at,updated_at) VALUES (?,?,?,?,?,0,NULL,NULL,?,?,?)",
                (job_id, "queued", filename, str(audio_path), json.dumps(options), now, now, now),
            )
        return self.get(job_id), True

    def get(self, job_id: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return dict(row) if row else None

    def claim_next(self) -> dict | None:
        """Atomically move one due job from queued to running."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM jobs WHERE status='queued' AND next_run_at<=? ORDER BY created_at LIMIT 1",
                (time.time(),),
            ).fetchone()
            if not row:
                return None
            self._conn.execute(
                "UPDATE jobs SET status='running', attempts=attempts+1, updated_at=? WHERE id=?",
                (time.time(), row["id"]),
            )
        return self.get(row["id"])

    def finish(self, job_id: str, result_path: Path) -> None:
        self._conn.execute(
            "UPDATE jobs SET status='succeeded', result_path=?, error=NULL, updated_at=? WHERE id=?",
            (str(result_path), time.time(), job_id),
        )

    def fail(self, job_id: str, error: str, retry_in: float | None) -> None:
        if retry_in is None:
            self._conn.execute(
                "UPDATE jobs SET status='failed', error=?, updated_at=? WHERE id=?", (error, time.time(), job_id)
            )
        else:
            self._conn.execute(
                "UPDATE jobs SET status='queued', error=?, next_run_at=?, updated_at=? WHERE id=?",
                (error, time.time() + retry_in, time.time(), job_id),
            )

    def requeue_stale(self) -> int:
        cur = self._conn.execute(
            "UPDATE jobs SET status='queued', next_run_at=?, updated_at=? WHERE status='running'",
            (time.time(), time.time()),
        )
        return cur.rowcount


class Worker:
    def __init__(self, store: JobStore, engine: Transcriber, results_dir: Path, concurrency: int = 2,
                 max_attempts: int = 3, base_backoff: float = 2.0, poll_interval: float = 0.5):
        self.store, self.engine, self.results_dir = store, engine, results_dir
        self.concurrency, self.max_attempts = concurrency, max_attempts
        self.base_backoff, self.poll_interval = base_backoff, poll_interval
        self._stop = threading.Event()
        self._pool = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="stt-worker")
        results_dir.mkdir(parents=True, exist_ok=True)

    def start(self) -> None:
        n = self.store.requeue_stale()
        if n:
            log.warning("re-queued %d job(s) interrupted by a previous shutdown", n)
        for _ in range(self.concurrency):
            self._pool.submit(self._loop)

    def stop(self) -> None:
        self._stop.set()
        self._pool.shutdown(wait=True, cancel_futures=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            job = self.store.claim_next()
            if job is None:
                self._stop.wait(self.poll_interval)
                continue
            self.run_one(job)

    def run_one(self, job: dict) -> None:
        opts = json.loads(job["options"])
        try:
            transcript = transcribe_file(Path(job["audio_path"]), self.engine,
                                         language=opts.get("language"), word_timestamps=opts.get("word_timestamps", False))
            out = self.results_dir / f"{job['id']}.json"
            tmp = out.with_suffix(".tmp")
            tmp.write_text(json.dumps(transcript.to_dict(), ensure_ascii=False))
            tmp.replace(out)  # atomic: readers never see a half-written result
            self.store.finish(job["id"], out)
            log.info("job %s succeeded", job["id"])
        except AudioError as e:  # permanent: bad input
            self.store.fail(job["id"], f"invalid_audio: {e}", retry_in=None)
        except Exception as e:  # transient: engine crash, OOM, timeout...
            if job["attempts"] >= self.max_attempts:
                self.store.fail(job["id"], f"failed after {job['attempts']} attempts: {e}", retry_in=None)
            else:
                delay = self.base_backoff * 2 ** (job["attempts"] - 1)
                log.warning("job %s attempt %d failed (%s); retrying in %.1fs", job["id"], job["attempts"], e, delay)
                self.store.fail(job["id"], str(e), retry_in=delay)
