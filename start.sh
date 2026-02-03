#!/bin/bash
# Start bot_runner in background
python bot_runner.py &

# Start web server (foreground)
uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}
