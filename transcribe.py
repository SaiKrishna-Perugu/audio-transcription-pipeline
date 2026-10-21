#!/usr/bin/env python3
"""CLI: transcribe an audio file and print/save the result.

    python transcribe.py samples/sample.mp3 --format srt -o out.srt
"""
import argparse
import logging
import sys
from pathlib import Path

from app import formats
from app.audio import AudioError
from app.config import settings
from app.pipeline import transcribe_file
from app.transcriber import WhisperTranscriber


def main() -> int:
    p = argparse.ArgumentParser(description="Transcribe audio to text with per-segment timestamps.")
    p.add_argument("audio", type=Path, help="WAV, MP3, M4A, FLAC, OGG, WEBM, MP4 ...")
    p.add_argument("--format", "-f", choices=list(formats.RENDERERS), default="json")
    p.add_argument("--output", "-o", type=Path, help="write to file instead of stdout")
    p.add_argument("--language", "-l", help="ISO code, e.g. en; auto-detected if omitted")
    p.add_argument("--model", default=settings.model_size, help="tiny | base | small | medium | large-v3")
    p.add_argument("--words", action="store_true", help="include word-level timestamps")
    p.add_argument("--chunk-seconds", type=int, default=settings.chunk_seconds)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)

    if not args.audio.exists():
        p.error(f"{args.audio} not found")
    try:
        t = transcribe_file(args.audio, WhisperTranscriber(args.model, settings.device, settings.compute_type),
                            language=args.language, word_timestamps=args.words, chunk_seconds=args.chunk_seconds)
    except AudioError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    out = formats.RENDERERS[args.format](t)
    if args.output:
        args.output.write_text(out, encoding="utf-8")
        print(f"wrote {args.output} ({len(t.segments)} segments, {t.duration:.1f}s audio, "
              f"{t.processing_seconds:.1f}s processing)", file=sys.stderr)
    else:
        print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
