"""Runtime configuration, overridable with environment variables."""
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model_size: str = field(default_factory=lambda: os.getenv("WHISPER_MODEL", "base"))
    device: str = field(default_factory=lambda: os.getenv("WHISPER_DEVICE", "cpu"))
    compute_type: str = field(default_factory=lambda: os.getenv("WHISPER_COMPUTE", "int8"))
    # Long files are cut into chunks of this many seconds (with overlap) before transcription.
    chunk_seconds: int = field(default_factory=lambda: int(os.getenv("CHUNK_SECONDS", "600")))
    chunk_overlap_seconds: float = field(default_factory=lambda: float(os.getenv("CHUNK_OVERLAP", "2")))
    max_upload_mb: int = field(default_factory=lambda: int(os.getenv("MAX_UPLOAD_MB", "500")))
    max_duration_seconds: int = field(default_factory=lambda: int(os.getenv("MAX_DURATION_S", str(4 * 3600))))
    workers: int = field(default_factory=lambda: int(os.getenv("WORKERS", "2")))
    max_attempts: int = field(default_factory=lambda: int(os.getenv("MAX_ATTEMPTS", "3")))
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "./data")))


settings = Settings()
