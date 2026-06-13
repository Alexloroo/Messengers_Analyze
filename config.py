"""Application configuration loaded from environment variables."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "telethon_data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file(PROJECT_ROOT / ".env")

TELETHON_API_ID = os.getenv("TELETHON_API_ID")
TELETHON_API_HASH = os.getenv("TELETHON_API_HASH")
SESSION_NAME = os.getenv("TELETHON_SESSION", str(DATA_DIR / "telethon"))

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"sqlite+aiosqlite:///{DATA_DIR / 'messages.db'}",
)

# Comma-separated list of informational chat ids/usernames/titles allowed for collection.
# Empty list means no chats are collected (safe default).
ALLOWED_INFO_CHATS = [
    item.strip()
    for item in os.getenv("ALLOWED_INFO_CHATS", "").split(",")
    if item.strip()
]

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
