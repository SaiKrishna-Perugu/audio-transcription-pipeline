"""Orchestration: probe -> normalize -> chunk -> transcribe each chunk -> merge -> post-process."""
from __future__ import annotations

import logging
import re
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from . import audio
from .config import settings
from .transcriber import Segment, Transcriber

log = logging.getLogger(__name__)


@dataclass
class Transcript:
    language: str | None
    language_probability: float | None
    duration: float
    segments: list[Segment]
    text: str
    engine: str
    processing_seconds: float
    source: dict

    def to_dict(self) -> dict:
        d = asdict(self)
        d["word_count"] = len(self.text.split())
        d["segment_count"] = len(self.segments)
        return d


def merge_chunks(per_chunk: list[tuple[audio.Chunk, list[Segment]]], tolerance: float = 0.25) -> list[Segment]:
    """Shift each chunk's segments to global time and drop duplicates from the overlap.

    Two rules, applied in order of chunks:
      1. Chunk k keeps a segment only if its midpoint is before (start of chunk k+1) + overlap/2
         (with silence-aligned cuts the overlap is 0 and this is simply the chunk boundary),
         because anything later is heard by chunk k+1 with more context.
      2. Chunk k+1 skips any segment that ends at or before what is already kept (+ tolerance),
         because chunk k already produced it.
    Together they keep exactly one copy of each phrase spoken around a boundary, even
    though the two chunks place that phrase at slightly different timestamps.
    """
    merged: list[Segment] = []
    for i, (chunk, segs) in enumerate(per_chunk):
        nxt = per_chunk[i + 1][0] if i + 1 < len(per_chunk) else None
        cut = (nxt.start + chunk.overlap_next / 2) if nxt else float("inf")
        for s in segs:
            g = s.shifted(chunk.start)
            if (g.start + g.end) / 2 >= cut:
                continue
            if merged and g.end <= merged[-1].end + tolerance:
                continue
            merged.append(g)
    return merged


_WS = re.compile(r"\s+")


def postprocess(segments: list[Segment]) -> tuple[list[Segment], str]:
    """Light clean-up for downstream use: trim, collapse whitespace, drop punctuation-only
    segments, and fold an exact repeat that overlaps the previous segment in time (a
    stitching artefact). Genuine repetition at different times is kept."""
    cleaned: list[Segment] = []
    for s in segments:
        text = _WS.sub(" ", s.text).strip()
        if not text or re.fullmatch(r"[\W_]+", text):
            continue
        if cleaned and cleaned[-1].text == text and s.start < cleaned[-1].end:
            cleaned[-1].end = max(cleaned[-1].end, s.end)  # same words, overlapping time: a stitching artefact
            continue
        s.text = text
        cleaned.append(s)
    full_text = " ".join(s.text for s in cleaned)
    return cleaned, full_text


def transcribe_file(
    path: Path,
    engine: Transcriber,
    language: str | None = None,
    word_timestamps: bool = False,
    chunk_seconds: int | None = None,
    overlap: float | None = None,
) -> Transcript:
    t0 = time.perf_counter()
    chunk_seconds = chunk_seconds or settings.chunk_seconds
    overlap = settings.chunk_overlap_seconds if overlap is None else overlap

    info = audio.probe(path)
    if info.duration > settings.max_duration_seconds:
        raise audio.AudioError(f"Audio is {info.duration:.0f}s; limit is {settings.max_duration_seconds}s")

    work = Path(tempfile.mkdtemp(prefix="stt_"))
    try:
        wav = audio.normalize(path, work / "normalized.wav")
        chunks = audio.split(wav, work / "chunks", info.duration, chunk_seconds, overlap)
        log.info("transcribing %s: %.1fs in %d chunk(s)", path.name, info.duration, len(chunks))

        per_chunk, language_votes = [], {}
        for c in chunks:
            # Detect language on the first chunk, then pin it so every chunk agrees.
            res = engine.transcribe(c.path, language=language, word_timestamps=word_timestamps)
            if language is None and res.language:
                language = res.language
                language_votes[res.language] = res.language_probability
            per_chunk.append((c, res.segments))

        segments = merge_chunks(per_chunk)
        segments, text = postprocess(segments)
        return Transcript(
            language=language,
            language_probability=language_votes.get(language) if language else None,
            duration=round(info.duration, 3),
            segments=segments,
            text=text,
            engine=engine.name,
            processing_seconds=round(time.perf_counter() - t0, 2),
            source={"filename": path.name, "codec": info.codec, "sample_rate": info.sample_rate,
                    "channels": info.channels, "container": info.container, "chunks": len(chunks)},
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)
