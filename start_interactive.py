#!/usr/bin/env python3
"""Root Launcher for MedShield FL."""

import os

if __name__ == "__main__":
    os.system("PYTHONPATH=. uv run python client/launch_client.py")
