#!/usr/bin/env bash
# MedShield FL — Hospital Client Desktop Application Launcher

echo "============================================================"
echo " 🛡️   Starting MedShield FL Hospital Client Desktop App   🛡️"
echo "============================================================"

# Launch Standalone Hospital Client Node Desktop GUI Window
PYTHONPATH=. client/.venv/bin/python client/gui_client.py
