"""Speech-to-text engines behind one small interface.

The pipeline depends only on `Transcriber`, so the engine can be swapped
(local faster-whisper, a hosted API, or a fake for tests) without touching anything else.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass
class Word:
    start: float
    end: float
    text: str
    probability: float | None = None


@dataclass
class Segment:
    start: float
    end: float
    text: str
    avg_logprob: float | None = None
    words: list[Word] = field(default_factory=list)

    def shifted(self, offset: float) -> "Segment":
        return Segment(
            start=round(self.start + offset, 3),
            end=round(self.end + offset, 3),
            text=self.text,
            avg_logprob=self.avg_logprob,
            words=[Word(round(w.start + offset, 3), round(w.end + offset, 3), w.text, w.probability) for w in self.words],
        )


@dataclass
class ChunkResult:
    segments: list[Segment]
    language: str | None
    language_probability: float | None


def load_pcm16(wav: Path):
    """Read a 16-bit PCM WAV into a float32 numpy array in [-1, 1]."""
    import wave

    import numpy as np

    with wave.open(str(wav), "rb") as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise ValueError("expected 16-bit mono PCM; run audio.normalize first")
        frames = w.readframes(w.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


class Transcriber(Protocol):
    name: str

    def transcribe(self, wav: Path, language: str | None = None, word_timestamps: bool = False) -> ChunkResult: ...


class WhisperTranscriber:
    """Local, open-source Whisper via faster-whisper (CTranslate2, int8 on CPU).

    The model is loaded lazily once per process and shared; a lock serialises inference
    because a single CTranslate2 model instance is not meant to be hammered concurrently
    from many threads. Throughput scales by running more worker processes.
    """

    name = "faster-whisper"

    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        self.model_size, self.device, self.compute_type = model_size, device, compute_type
        self._model = None
        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()

    def _get_model(self):
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    from faster_whisper import WhisperModel  # imported lazily: heavy

                    self._model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
        return self._model

    def transcribe(self, wav: Path, language: str | None = None, word_timestamps: bool = False) -> ChunkResult:
        model = self._get_model()
        samples = load_pcm16(wav)  # already 16 kHz mono from audio.normalize; no second decode
        with self._infer_lock:
            segments_iter, info = model.transcribe(
                samples,
                language=language,
                vad_filter=True,  # skip silence: faster, and fewer hallucinations on quiet audio
                word_timestamps=word_timestamps,
                beam_size=5,
            )
            segments = [
                Segment(
                    start=round(s.start, 3),
                    end=round(s.end, 3),
                    text=s.text.strip(),
                    avg_logprob=s.avg_logprob,
                    words=[Word(round(w.start, 3), round(w.end, 3), w.word.strip(), w.probability) for w in (s.words or [])],
                )
                for s in segments_iter  # generator: decoding happens here
            ]
        return ChunkResult(segments, info.language, info.language_probability)


class FakeTranscriber:
    """Deterministic engine for tests and offline demos."""

    name = "fake"

    def __init__(self, segments: list[Segment] | None = None, fail_times: int = 0):
        self.segments = segments or [Segment(0.0, 1.5, "hello world"), Segment(1.5, 3.0, "this is a test")]
        self.fail_times = fail_times
        self.calls = 0

    def transcribe(self, wav: Path, language: str | None = None, word_timestamps: bool = False) -> ChunkResult:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("simulated transient engine failure")
        return ChunkResult(list(self.segments), language or "en", 0.99)
