#!/bin/bash
# Start the web frontend service

echo "Starting JittorGeometric web frontend..."
cd "$(dirname "$0")/.."

python frontend_demo/web_server.py