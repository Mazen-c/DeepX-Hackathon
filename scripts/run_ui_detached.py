"""Launch the ABSA web UI with file-backed stdout/stderr."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = PROJECT_ROOT / "ui_server.log"

sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout = LOG_PATH.open("a", encoding="utf-8", buffering=1)
sys.stderr = sys.stdout
sys.argv = ["app.py", "--host", "127.0.0.1", "--port", "8501"]

from app import main  # noqa: E402


raise SystemExit(main())
