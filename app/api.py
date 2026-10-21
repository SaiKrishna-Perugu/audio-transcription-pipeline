"""HTTP API.

POST /v1/transcriptions            upload audio -> 202 + job (async, for anything non-trivial)
GET  /v1/transcriptions/{id}       job status (+ transcript when done)
GET  /v1/transcriptions/{id}/{fmt} transcript as json | srt | vtt | txt
POST /v1/transcriptions:sync       small files only, returns the transcript directly
GET  /healthz
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, PlainTextResponse

from . import formats
from .audio import SUPPORTED_EXTENSIONS, AudioError
from .config import settings
from .jobs import JobStore, Worker, file_sha256
from .pipeline import Transcript, transcribe_file
from .transcriber import Segment, Transcriber, WhisperTranscriber, Word

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

SYNC_MAX_SECONDS = 120


def create_app(engine: Transcriber | None = None, data_dir: Path | None = None, start_worker: bool = True) -> FastAPI:
    engine = engine or WhisperTranscriber(settings.model_size, settings.device, settings.compute_type)
    data_dir = data_dir or settings.data_dir
    uploads, results = data_dir / "uploads", data_dir / "results"
    uploads.mkdir(parents=True, exist_ok=True)
    store = JobStore(data_dir / "jobs.db")
    worker = Worker(store, engine, results, concurrency=settings.workers, max_attempts=settings.max_attempts)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if start_worker:
            worker.start()
        yield
        if start_worker:
            worker.stop()

    app = FastAPI(title="Transcription Pipeline", version="0.1.0", lifespan=lifespan)
    app.state.store, app.state.worker = store, worker

    async def _save_upload(file: UploadFile) -> Path:
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            raise HTTPException(415, f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}")
        limit = settings.max_upload_mb * 1024 * 1024
        tmp = uploads / f".incoming-{uuid.uuid4().hex}{suffix}"
        size = 0
        with open(tmp, "wb") as out:  # stream to disk in 1 MB pieces; never hold the file in memory
            while block := await file.read(1 << 20):
                size += len(block)
                if size > limit:
                    out.close(); tmp.unlink(missing_ok=True)
                    raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB")
                out.write(block)
        if size == 0:
            tmp.unlink(missing_ok=True)
            raise HTTPException(400, "Empty file")
        return tmp

    def _job_view(job: dict) -> dict:
        view = {k: job[k] for k in ("id", "status", "filename", "attempts", "error", "created_at", "updated_at")}
        if job["status"] == "succeeded" and job["result_path"]:
            view["result"] = json.loads(Path(job["result_path"]).read_text())
        return view

    def _load_transcript(job_id: str) -> Transcript:
        job = store.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        if job["status"] != "succeeded":
            raise HTTPException(409, f"Job is {job['status']}")
        d = json.loads(Path(job["result_path"]).read_text())
        segs = [Segment(s["start"], s["end"], s["text"], s.get("avg_logprob"),
                        [Word(**w) for w in s.get("words", [])]) for s in d["segments"]]
        return Transcript(d["language"], d["language_probability"], d["duration"], segs, d["text"],
                          d["engine"], d["processing_seconds"], d["source"])

    @app.get("/healthz")
    def healthz():
        return {"status": "ok", "engine": engine.name}

    @app.post("/v1/transcriptions", status_code=202)
    async def create_transcription(file: UploadFile = File(...), language: str | None = Form(None),
                                   word_timestamps: bool = Form(False)):
        tmp = await _save_upload(file)
        digest = await run_in_threadpool(file_sha256, tmp)
        options = {"language": language, "word_timestamps": word_timestamps}
        job_id = hashlib.sha256((digest + json.dumps(options, sort_keys=True)).encode()).hexdigest()[:24]
        final = uploads / f"{digest}{tmp.suffix}"  # content-addressed: identical uploads stored once
        if final.exists():
            tmp.unlink()
        else:
            tmp.replace(final)
        job, created = store.create(job_id, file.filename or final.name, final, options)
        return JSONResponse(_job_view(job), status_code=202 if created else 200,
                            headers={"Location": f"/v1/transcriptions/{job_id}"})

    @app.get("/v1/transcriptions/{job_id}")
    def get_transcription(job_id: str):
        job = store.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        return _job_view(job)

    @app.get("/v1/transcriptions/{job_id}/{fmt}")
    def export_transcription(job_id: str, fmt: str):
        if fmt not in formats.RENDERERS:
            raise HTTPException(400, f"Format must be one of {list(formats.RENDERERS)}")
        t = _load_transcript(job_id)
        return PlainTextResponse(formats.RENDERERS[fmt](t), media_type=formats.MEDIA_TYPES[fmt])

    @app.post("/v1/transcriptions:sync")
    async def transcribe_sync(file: UploadFile = File(...), language: str | None = Form(None),
                              word_timestamps: bool = Form(False)):
        tmp = await _save_upload(file)
        try:
            from .audio import probe
            info = await run_in_threadpool(probe, tmp)
            if info.duration > SYNC_MAX_SECONDS:
                raise HTTPException(413, f"Sync endpoint is for audio <= {SYNC_MAX_SECONDS}s; use POST /v1/transcriptions")
            t = await run_in_threadpool(transcribe_file, tmp, engine, language, word_timestamps)
            return t.to_dict()
        except AudioError as e:
            raise HTTPException(422, str(e))
        finally:
            tmp.unlink(missing_ok=True)

    return app

