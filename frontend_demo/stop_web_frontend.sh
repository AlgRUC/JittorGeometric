#!/bin/bash
# Stop the web frontend service - SAFE VERSION

echo "Stopping JittorGeometric web frontend..."

# Find and kill ONLY the JittorGeometric web frontend process
# This is safe because it only kills processes running our specific script
pkill -f "python simple_web_frontend.py"

# If pkill fails, use ps and kill with more precise filtering
if [ $? -ne 0 ]; then
    echo "pkill failed, trying ps..."
    # Use grep with -F for fixed strings to avoid regex issues
    # Only match processes running our exact script
    PIDS=$(ps aux | grep -F "python simple_web_frontend.py" | grep -v grep | awk '{print $2}')
    if [ -n "$PIDS" ]; then
        for PID in $PIDS; do
            kill $PID
            echo "Killed JittorGeometric process $PID"
        done
    else
        echo "No JittorGeometric web frontend process found"
    fi
else
    echo "JittorGeometric web frontend stopped successfully"
fi

# Optional: Check if any process is running on the specific port we use
# This is safer than scanning all ports 5000-6000
PORT=5000
PID=$(lsof -i :$PORT | grep LISTEN | awk '{print $2}')
if [ -n "$PID" ]; then
    # Verify it's actually our process before killing
    if ps -p $PID -o cmd= | grep -q "simple_web_frontend.py"; then
        kill -9 $PID
        echo "Killed JittorGeometric process $PID running on port $PORT"
    else
        echo "Warning: Process $PID is running on port $PORT but it's not our JittorGeometric frontend"
        echo "Not killing it to avoid affecting other services"
    fi
fi