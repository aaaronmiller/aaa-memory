#!/bin/bash
# Clawdi Daily Memory Update — Cron job
# Runs at 2:00 AM daily via: crontab -e → 0 2 * * * /home/misscheta/code/aaa-memory/scripts/daily-update.sh

set -euo pipefail

LOG_FILE="/home/misscheta/logs/memory-update-cron.log"
exec 1>>"$LOG_FILE" 2>&1

echo "=== $(date -Iseconds) Starting daily memory update ==="

# Start capture bridge in background if not running
if ! pgrep -f "capture-bridge.py" > /dev/null; then
    echo "Starting capture bridge..."
    nohup python3 /home/misscheta/code/aaa-memory/scripts/capture-bridge.py >> /home/misscheta/logs/capture-bridge.log 2>&1 &
    echo "Capture bridge started (PID $!)"
else
    echo "Capture bridge already running"
fi

# Run daily update
echo "Running ingestion scan..."
python3 /home/misscheta/code/aaa-memory/scripts/daily-update.py

echo "=== $(date -Iseconds) Daily update complete ==="
