import os
from pathlib import Path


class Settings:
    KB_DIR: str = os.getenv("KB_DIR", str(
        Path(__file__).resolve().parents[1] / "dataset" / "kb"))
    TOP_K: int = int(os.getenv("TOP_K", "3"))
    MODEL_NAME: str = os.getenv("MODEL_NAME", os.getenv("LLM_MODEL", "stub"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    MAX_CONTEXT_CHARS: int = int(os.getenv("MAX_CONTEXT_CHARS", "2000"))


settings = Settings()
