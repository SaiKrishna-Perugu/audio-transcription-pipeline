import time
from fastapi.testclient import TestClient
from app.api import create_app
from app.transcriber import FakeTranscriber


def _wait(client, job_id, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        job = client.get(f"/v1/transcriptions/{job_id}").json()
        if job["status"] in ("succeeded", "failed"):
            return job
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_async_flow_idempotency_and_exports(tmp_path, tone_factory):
    app = create_app(engine=FakeTranscriber(), data_dir=tmp_path / "data")
    src = tone_factory("clip.mp3", 2)
    with TestClient(app) as c:
        r = c.post("/v1/transcriptions", files={"file": ("clip.mp3", src.read_bytes(), "audio/mpeg")})
        assert r.status_code == 202
        job = _wait(c, r.json()["id"])
        assert job["status"] == "succeeded" and job["result"]["text"] == "hello world this is a test"
        again = c.post("/v1/transcriptions", files={"file": ("clip.mp3", src.read_bytes(), "audio/mpeg")})
        assert again.status_code == 200 and again.json()["id"] == job["id"]  # no duplicate work
        assert "-->" in c.get(f"/v1/transcriptions/{job['id']}/srt").text


def test_transient_failures_are_retried(tmp_path, tone_factory):
    app = create_app(engine=FakeTranscriber(fail_times=2), data_dir=tmp_path / "data")
    app.state.worker.base_backoff = 0.05
    with TestClient(app) as c:
        r = c.post("/v1/transcriptions", files={"file": ("a.wav", tone_factory("a.wav", 1).read_bytes(), "audio/wav")})
        job = _wait(c, r.json()["id"])
        assert job["status"] == "succeeded" and job["attempts"] == 3


def test_bad_audio_fails_without_retry(tmp_path):
    app = create_app(engine=FakeTranscriber(), data_dir=tmp_path / "data")
    with TestClient(app) as c:
        r = c.post("/v1/transcriptions", files={"file": ("bad.mp3", b"garbage" * 50, "audio/mpeg")})
        job = _wait(c, r.json()["id"])
        assert job["status"] == "failed" and job["attempts"] == 1 and job["error"].startswith("invalid_audio")


def test_rejects_unsupported_type(tmp_path):
    app = create_app(engine=FakeTranscriber(), data_dir=tmp_path / "data", start_worker=False)
    with TestClient(app) as c:
        assert c.post("/v1/transcriptions", files={"file": ("x.pdf", b"%PDF", "application/pdf")}).status_code == 415
