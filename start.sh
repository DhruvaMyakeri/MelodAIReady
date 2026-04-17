#!/usr/bin/env bash
# start.sh — Launch the Audio-Instruments-ML web server
# Usage: bash start.sh

cd "$(dirname "$0")"
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
