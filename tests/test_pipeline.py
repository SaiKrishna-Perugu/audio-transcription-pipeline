from app import audio, formats
from app.pipeline import merge_chunks, postprocess, transcribe_file
from app.transcriber import FakeTranscriber, Segment


def test_merge_shifts_offsets_and_dedupes_overlap():
    c0 = audio.Chunk(0, None, 0.0, 10.0, overlap_next=2.0)  # hard cut at 8s, 2s overlap
    c1 = audio.Chunk(1, None, 8.0, 10.0)
    segs0 = [Segment(0, 4, "a"), Segment(4, 8.5, "b"), Segment(8.5, 10, "dup-in-overlap")]
    segs1 = [Segment(0, 1.5, "dup-in-overlap"), Segment(1.5, 6, "c")]
    out = merge_chunks([(c0, segs0), (c1, segs1)])
    assert [s.text for s in out] == ["a", "b", "dup-in-overlap", "c"]
    assert out[-1].start == 9.5 and out[-1].end == 14.0


def test_postprocess_cleans_repeats_and_noise():
    segs = [Segment(0, 1, "  hello   there "), Segment(1, 2, "..."), Segment(2, 3, "thanks"),
            Segment(2.8, 3.4, "thanks"), Segment(5, 6, "thanks")]
    cleaned, text = postprocess(segs)
    assert text == "hello there thanks thanks"  # overlapping copy folded, later real repeat kept
    assert cleaned[1].end == 3.4


def test_silence_aligned_chunks_need_no_overlap():
    contiguous = [audio.Chunk(0, None, 0.0, 9.6), audio.Chunk(1, None, 9.6, 8.0)]
    out = merge_chunks([(contiguous[0], [Segment(0, 9.0, "first")]), (contiguous[1], [Segment(0.2, 5, "second")])])
    assert [(s.text, s.start) for s in out] == [("first", 0.0), ("second", 9.8)]


def test_end_to_end_with_fake_engine(tone_factory):
    t = transcribe_file(tone_factory("x.mp3", 3), FakeTranscriber())
    assert t.text == "hello world this is a test"
    assert t.language == "en" and t.source["codec"] == "mp3"
    srt = formats.to_srt(t)
    assert "00:00:00,000 --> 00:00:01,500" in srt and "this is a test" in srt
    assert formats.to_vtt(t).startswith("WEBVTT")


def test_long_file_is_chunked(tone_factory):
    eng = FakeTranscriber()
    t = transcribe_file(tone_factory("long.wav", 25), eng, chunk_seconds=10, overlap=2)
    assert t.source["chunks"] == 3 and eng.calls == 3
