"""Single source of truth for the source and packaged Jarvis release."""
from pathlib import Path


VERSION_FILE = Path(__file__).resolve().parents[1] / "VERSION"
__version__ = VERSION_FILE.read_text(encoding="utf-8").strip()

if not __version__:
    raise RuntimeError("VERSION must not be empty")
