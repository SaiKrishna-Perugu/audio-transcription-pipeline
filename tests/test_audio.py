import subprocess
import pytest
from app import audio


@pytest.mark.parametrize("name,args", [("a.wav", ["-ar", "44100", "-ac", "2"]), ("a.mp3", []),
                                       ("a.flac", []), ("a.ogg", ["-c:a", "libvorbis"]), ("a.m4a", [])])
def test_any_format_normalizes_to_16k_mono(tone_factory, tmp_path, name, args):
    src = tone_factory(name, 2, args)
    info = audio.probe(src)
    assert info.duration == pytest.approx(2, abs=0.1)
    out = audio.normalize(src, tmp_path / "norm.wav")
    n = audio.probe(out)
    assert (n.sample_rate, n.channels, n.codec) == (16000, 1, "pcm_s16le")


def test_rejects_unsupported_extension(tmp_path):
    f = tmp_path / "notes.txt"; f.write_text("hi")
    with pytest.raises(audio.AudioError):
        audio.probe(f)


def test_rejects_corrupt_file(tmp_path):
    f = tmp_path / "broken.mp3"; f.write_bytes(b"not really audio" * 100)
    with pytest.raises(audio.AudioError):
        audio.probe(f)


def test_continuous_audio_falls_back_to_overlapping_hard_cuts(tone_factory, tmp_path):
    wav = audio.normalize(tone_factory("long.wav", 25), tmp_path / "n.wav")  # a steady tone has no pauses
    chunks = audio.split(wav, tmp_path / "chunks", 25, chunk_seconds=10, overlap=2)
    assert [c.start for c in chunks] == [0, 10, 20]
    assert [c.overlap_next for c in chunks] == [2, 2, 0]
    assert chunks[-1].start + chunks[-1].duration == pytest.approx(25, abs=0.05)


def test_cuts_prefer_pauses():
    silences = [(7.0, 7.6), (18.5, 19.1), (29.0, 29.4)]
    cuts = audio.plan_cuts(35, chunk_seconds=10, silences=silences)
    assert cuts[0] == (pytest.approx(7.3), True)
    assert cuts[1] == (pytest.approx(17.3), False)  # no pause in (12.3, 17.3]
    assert cuts[2] == (pytest.approx(27.3), False) or cuts[2][1] is True


def test_short_audio_is_single_chunk(tone_factory, tmp_path):
    wav = audio.normalize(tone_factory("s.wav", 3), tmp_path / "n.wav")
    assert len(audio.split(wav, tmp_path / "c", 3, 10, 2)) == 1
