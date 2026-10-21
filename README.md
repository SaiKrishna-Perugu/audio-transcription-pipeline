# Transcription Pipeline

A small, production-minded pipeline that turns an audio file (WAV, MP3, M4A, FLAC, OGG, WEBM, MP4…) into text with **timestamps per segment**, and exposes it as a **CLI** and an **async HTTP API**.

Speech-to-text runs locally on the open-source **Whisper** model via [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2, int8 on CPU), so there are no API keys or per-minute costs. The focus is on the engineering around the model: format handling, long files, concurrency, storage, retries and the API.

```
audio file ──► probe ──► normalize ──► split at pauses ──► transcribe chunks ──► merge + dedupe ──► clean up ──► JSON / SRT / VTT / TXT
            (ffprobe)  (16 kHz mono)   (long files only)    (faster-whisper)    (global timestamps)
```

## Quick start

```bash
# needs Python 3.10+ and ffmpeg on PATH
pip install -r requirements.txt

# CLI
python transcribe.py samples/jfk.mp3                 # JSON to stdout
python transcribe.py samples/jfk.mp3 -f srt -o out.srt
python transcribe.py talk.m4a --language en --words  # word-level timestamps too

# API
uvicorn app.api:create_app --factory --port 8000
curl -F file=@samples/jfk.mp3 localhost:8000/v1/transcriptions        # -> 202 {"id": "...", "status": "queued"}
curl localhost:8000/v1/transcriptions/<id>                           # status + transcript when done
curl localhost:8000/v1/transcriptions/<id>/srt                       # json | srt | vtt | txt

# Docker
docker build -t transcriber . && docker run -p 8000:8000 -v $PWD/data:/data transcriber

# Tests (use a fake engine, so they run in seconds without the model)
pytest -q
```

Sample output for `samples/jfk.mp3` is in `samples/jfk.transcript.json` and `samples/jfk.srt`:

```json
{
  "language": "en",
  "duration": 11.088,
  "segments": [{"start": 0.0, "end": 11.0, "text": "And so my fellow Americans, ask not what your country can do for you, ask what you can do for your country."}],
  "text": "And so my fellow Americans, ...",
  "word_count": 22,
  "source": {"codec": "mp3", "chunks": 1}
}
```

## Project layout

| File | Responsibility |
|---|---|
| `app/audio.py` | ffprobe validation, normalization to 16 kHz mono PCM, silence detection, chunk planning and splitting |
| `app/transcriber.py` | `Transcriber` interface, `WhisperTranscriber` (faster-whisper), `FakeTranscriber` for tests |
| `app/pipeline.py` | Orchestration, merging chunk results onto one timeline, deduping, text clean-up |
| `app/formats.py` | JSON / SRT / WebVTT / TXT renderers |
| `app/jobs.py` | Durable SQLite job queue, worker pool, retries with backoff, crash recovery, idempotency |
| `app/api.py` | FastAPI app: async + sync endpoints, streaming uploads, exports |
| `transcribe.py` | CLI |
| `tests/` | 19 tests: formats, chunking, merge/dedupe, retries, idempotency, bad input |

## Design decisions

### 1. Normalize everything first
Every input goes through `ffmpeg` once and becomes **16 kHz, mono, 16-bit PCM WAV**, which is Whisper's native input. After that, nothing downstream knows or cares what the original format was. That gives one code path to test, and accurate durations for chunking. ffprobe rejects files with no audio stream, zero length or a corrupt container *before* any work is queued. The normalized samples are passed to the model as a numpy array, so audio is decoded exactly once.

### 2. Long files: split at pauses, not at fixed times
Whisper works on 30 s windows internally, but holding a 2-hour file in one call is slow, memory-hungry and all-or-nothing if it fails. Files longer than `CHUNK_SECONDS` (default 600 s) are split:

* `ffmpeg silencedetect` finds pauses. The threshold is **relative to the file's own loudness** (mean − 12 dB), so it works for quiet phone audio and loud studio audio alike.
* For each boundary I look back up to 30 s from the ideal cut and **cut in the middle of the latest pause**, so no word is split and no overlap is needed.
* If there's no pause (music, nonstop speech) I fall back to a **hard cut with a 2 s overlap**. When merging, a segment belongs to the chunk whose zone holds its midpoint, and anything the previous chunk already covered is skipped, so each phrase appears exactly once.
* Each chunk's timestamps are shifted by its offset, so the output is one continuous timeline.
* Language is detected on the first chunk and pinned for the rest, so chunks never disagree.

Tested on a 66 s file cut into 25 s chunks: both cuts landed in pauses (24.9 s, 48.0 s) and every sentence appeared exactly once.

### 3. Engine behind an interface
`Transcriber` is a two-method protocol. faster-whisper is the default (open-source, local, about 4× faster than reference Whisper on CPU with int8). A hosted API (Deepgram, AssemblyAI, OpenAI) or a GPU worker can be dropped in without touching the pipeline. Tests use a deterministic fake, so the suite runs in seconds with no model download.

`vad_filter=True` skips silence before decoding. That's faster and removes most of Whisper's "hallucinated" text on silent stretches.

### 4. Output built for downstream use
* Segments with `start`/`end` in seconds, plus optional word-level timestamps and `avg_logprob` as a confidence signal.
* A clean full `text`, `word_count`, detected `language` and source metadata.
* Light clean-up: whitespace collapsed, punctuation-only segments dropped, and an exact repeat that *overlaps in time* (a stitching artefact) folded. Real repetition at different times is kept.
* Exports in JSON (machines), SRT/VTT (subtitles, players) and TXT (humans).

### 5. Async API with a durable job queue
Transcription takes seconds to minutes, so the main endpoint **returns `202 Accepted` with a job id immediately** and work happens in a background worker pool. Clients poll `GET /v1/transcriptions/{id}`, or a webhook could be added. `POST /v1/transcriptions:sync` exists for clips of 2 minutes or less, where a blocking call is more convenient.

Uploads are **streamed to disk in 1 MB pieces** with a size limit, never held in memory.

### 6. Idempotency and storage
The job id is `sha256(audio bytes + options)`. Uploading the same file twice returns the existing job (`200`) instead of transcribing it again. Audio is stored **content-addressed** (`uploads/<sha256>.<ext>`), so duplicates take no extra space. Results are written atomically (temp file + rename), so a reader never sees half a JSON file.

### 7. Retries and recovery
* **Permanent errors** (corrupt or unsupported audio) fail immediately with `invalid_audio`. Retrying can't fix bad input.
* **Transient errors** (engine crash, OOM, timeout) are retried with **exponential backoff** (2 s, 4 s, …) up to `MAX_ATTEMPTS` (default 3). After that the job is `failed` with the last error, and re-submitting the file restarts it.
* **Crash recovery:** on startup, jobs stuck in `running` from a dead process are put back in the queue. The input audio is stored durably, so nothing is lost.

## Scaling to production (not implemented: thoughts)

| Concern | Here | In production |
|---|---|---|
| Queue | SQLite + thread pool | SQS / Redis Streams / Pub/Sub, with visibility timeouts and a dead-letter queue |
| Workers | In-process threads | Separate autoscaled worker deployment (GPU nodes for large models), scaled on queue depth |
| Audio | Local disk | S3/GCS with lifecycle rules (e.g. delete raw audio after 30 days); clients upload via presigned URLs straight to the bucket |
| Metadata and transcripts | SQLite + JSON files | Postgres (jobs, status, segments as JSONB), and optionally a search index over text |
| Long files | Sequential chunks | Fan out chunks to many workers in parallel, then a merge step |
| API | Polling | Polling plus webhooks, auth (API keys/OAuth), per-tenant rate limits and quotas |
| Ops | Logs | Metrics (queue depth, real-time factor, failure rate), tracing, alerts |

## Configuration

| Env var | Default | Meaning |
|---|---|---|
| `WHISPER_MODEL` | `base` | `tiny`, `base`, `small`, `medium`, `large-v3` |
| `WHISPER_DEVICE` / `WHISPER_COMPUTE` | `cpu` / `int8` | e.g. `cuda` / `float16` on GPU |
| `CHUNK_SECONDS` / `CHUNK_OVERLAP` | `600` / `2` | Long-file splitting |
| `WORKERS` | `2` | Concurrent jobs per process |
| `MAX_ATTEMPTS` | `3` | Retries for transient failures |
| `MAX_UPLOAD_MB` / `MAX_DURATION_S` | `500` / `14400` | Input limits |
| `DATA_DIR` | `./data` | Uploads, results, job DB |

## License
MIT
