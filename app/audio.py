"""Audio I/O: probing, format normalization and chunking, all via ffmpeg.

Every input (WAV, MP3, M4A, FLAC, OGG, WEBM, even video containers) is converted to
one canonical format, 16 kHz mono 16-bit PCM WAV, which is what Whisper expects.
Downstream code therefore never has to care about the original format.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".webm", ".mp4", ".wma"}
TARGET_SAMPLE_RATE = 16_000


class AudioError(ValueError):
    """Raised for unreadable, unsupported or empty audio. Not retryable."""


@dataclass(frozen=True)
class AudioInfo:
    duration: float
    codec: str
    sample_rate: int
    channels: int
    container: str


@dataclass(frozen=True)
class Chunk:
    index: int
    path: Path
    start: float  # offset of this chunk in the original audio, seconds
    duration: float
    overlap_next: float = 0.0  # seconds shared with the next chunk (0 when cut at silence)


def _require_ffmpeg() -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("ffmpeg/ffprobe not found on PATH")


def probe(path: Path) -> AudioInfo:
    """Inspect a file with ffprobe; reject anything without an audio stream."""
    _require_ffmpeg()
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise AudioError(f"Unsupported extension '{path.suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}")
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise AudioError(f"Could not read audio file: {proc.stderr.strip()[:200]}")
    meta = json.loads(proc.stdout or "{}")
    audio = next((s for s in meta.get("streams", []) if s.get("codec_type") == "audio"), None)
    if audio is None:
        raise AudioError("File contains no audio stream")
    duration = float(meta.get("format", {}).get("duration") or audio.get("duration") or 0)
    if duration <= 0:
        raise AudioError("Audio has zero duration")
    return AudioInfo(
        duration=duration,
        codec=audio.get("codec_name", "unknown"),
        sample_rate=int(audio.get("sample_rate", 0) or 0),
        channels=int(audio.get("channels", 0) or 0),
        container=meta.get("format", {}).get("format_name", "unknown"),
    )


def normalize(src: Path, dst: Path) -> Path:
    """Convert any supported input to 16 kHz mono PCM WAV."""
    _require_ffmpeg()
    dst.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-v", "error", "-i", str(src),
         "-vn", "-ac", "1", "-ar", str(TARGET_SAMPLE_RATE), "-c:a", "pcm_s16le", str(dst)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise AudioError(f"ffmpeg failed to decode audio: {proc.stderr.strip()[:200]}")
    return dst


def mean_volume_db(wav: Path) -> float:
    proc = subprocess.run(["ffmpeg", "-nostdin", "-i", str(wav), "-af", "volumedetect", "-f", "null", "-"],
                          capture_output=True, text=True)
    m = re.search(r"mean_volume: (-?[\d.]+) dB", proc.stderr)
    return float(m.group(1)) if m else -30.0


def detect_silences(wav: Path, noise_db: float | None = None, min_silence: float = 0.3) -> list[tuple[float, float]]:
    """Return (start, end) of pauses using ffmpeg's silencedetect filter.

    The threshold is relative to the recording's own loudness (mean - 12 dB) rather than a
    fixed level, so it works for both quiet phone recordings and loud studio audio.
    """
    if noise_db is None:
        noise_db = min(-20.0, max(-50.0, mean_volume_db(wav) - 12.0))
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "info", "-i", str(wav), "-af",
         f"silencedetect=noise={noise_db}dB:d={min_silence}", "-f", "null", "-"],
        capture_output=True, text=True,
    )
    starts = [float(x) for x in re.findall(r"silence_start: (-?[\d.]+)", proc.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", proc.stderr)]
    return [(max(0.0, a), b) for a, b in zip(starts, ends)]


def plan_cuts(duration: float, chunk_seconds: int, silences: list[tuple[float, float]],
              search_window: float = 30.0) -> list[tuple[float, bool]]:
    """Choose where to cut a long recording.

    For each boundary we look back up to `search_window` seconds from the ideal cut point
    and take the latest pause, cutting in its middle so no word is split. Returns a list of
    (cut_time, at_silence). If there's no pause in the window we cut at the ideal point and
    flag it, so the caller adds an overlap there instead.
    """
    cuts, last = [], 0.0
    window = min(search_window, chunk_seconds / 2)
    while duration - last > chunk_seconds:
        target = last + chunk_seconds
        candidates = [(a + b) / 2 for a, b in silences if target - window <= (a + b) / 2 <= target]
        if candidates:
            cut, at_silence = max(candidates), True
        else:
            cut, at_silence = target, False
        cuts.append((cut, at_silence))
        last = cut
    return cuts


def split(wav: Path, out_dir: Path, duration: float, chunk_seconds: int, overlap: float) -> list[Chunk]:
    """Cut a normalized WAV into ~chunk_seconds pieces, preferring to cut at pauses.

    - Cut at a pause: chunks are contiguous, no overlap needed.
    - No pause nearby (music, nonstop speech): hard cut with `overlap` seconds shared with
      the next chunk, so a word straddling the cut is fully heard at least once; the merge
      step removes the duplicate.
    Files shorter than one window are returned as a single chunk (no copy).
    """
    if duration <= chunk_seconds:
        return [Chunk(0, wav, 0.0, duration)]
    out_dir.mkdir(parents=True, exist_ok=True)
    cuts = plan_cuts(duration, chunk_seconds, detect_silences(wav))
    bounds = [0.0] + [c for c, _ in cuts] + [duration]
    chunks: list[Chunk] = []
    for i in range(len(bounds) - 1):
        start = bounds[i]
        hard_cut_after = i < len(cuts) and not cuts[i][1]
        extra = overlap if hard_cut_after else 0.0
        end = min(duration, bounds[i + 1] + extra)
        path = out_dir / f"chunk_{i:04d}.wav"
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}",
             "-i", str(wav), "-c", "copy", str(path)],
            check=True, capture_output=True,
        )
        chunks.append(Chunk(i, path, start, end - start, overlap_next=extra))
    return chunks
