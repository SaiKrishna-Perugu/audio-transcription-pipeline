import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def make_tone(path: Path, seconds: float, fmt_args: list[str] | None = None) -> Path:
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "lavfi", "-i",
                    f"sine=frequency=440:duration={seconds}", *(fmt_args or []), str(path)], check=True)
    return path


@pytest.fixture
def tone_factory(tmp_path):
    return lambda name, seconds=3, fmt_args=None: make_tone(tmp_path / name, seconds, fmt_args)
