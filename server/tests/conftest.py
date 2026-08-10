"""Test configuration & path setup."""

import sys
from pathlib import Path

CLIENT_DIR = Path(__file__).resolve().parent.parent.parent / "client"
if str(CLIENT_DIR) not in sys.path:
    sys.path.insert(0, str(CLIENT_DIR))
