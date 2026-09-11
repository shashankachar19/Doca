#!/bin/bash
set -e

echo "Starting DoCA QA Pipeline..."

# 1. Environment Setup
echo "Installing system dependencies..."
sudo DEBIAN_FRONTEND=noninteractive apt-get update
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y tesseract-ocr ffmpeg xvfb libgl1 libxkbcommon-x11-0 libegl1 libxcb-cursor0 python3-tk scrot xdotool libmagic1

echo "Installing Python dependencies..."
# Remove invalid magic-bin
sed -i 's/python-magic-bin/python-magic/' requirements.txt || true
pip install -r requirements.txt
pip install pyautogui pydub fpdf scipy opencv-python python-docx pymupdf numpy requests

echo "Setting up CouchDB..."
# Ensure docker is installed (usually is in these sandboxes, or use apt)
if ! command -v docker &> /dev/null
then
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y docker.io
fi

# Stop/rm existing couchdb if exists
docker rm -f couchdb || true
sudo mkdir -p /etc/docker
echo '{"storage-driver": "vfs"}' | sudo tee /etc/docker/daemon.json
sudo systemctl restart docker
docker run -d --name couchdb -e COUCHDB_USER=admin -e COUCHDB_PASSWORD=admin -p 5984:5984 couchdb:latest

# Wait for CouchDB to start
sleep 5

# 2. Start Watchdog Service
echo "Starting Watchdog Service..."
mkdir -p monitored_folder organized_output
python3 watchdog_service.py > watchdog.log 2>&1 &
WATCHDOG_PID=$!

# Wait for handlers to initialize
sleep 5

# 3. Generate Data and Run Pipeline
echo "Generating Synthetic Dataset..."
rm -rf test_data/* monitored_folder/* organized_output/*
python3 generate_data.py

echo "Waiting for CouchDB to reach 80 documents..."
python3 wait_for_db.py

# 4. Launch Dashboard and Record
echo "Setting up Xvfb..."
export DISPLAY=:99
Xvfb :99 -screen 0 1280x1024x24 &
XVFB_PID=$!
sleep 2

# We need a dummy .Xauthority for Xlib
touch ~/.Xauthority

echo "Starting screen recording..."
ffmpeg -y -video_size 1280x1024 -framerate 25 -f x11grab -i :99.0 -c:v libvpx-vp9 -b:v 2M output.webm > ffmpeg.log 2>&1 &
FFMPEG_PID=$!

echo "Launching UI and running automation..."
python3 ui/main_dashboard.py > ui.log 2>&1 &
UI_PID=$!

python3 ui_automation.py

echo "Stopping screen recording gracefully..."
kill -INT $FFMPEG_PID
wait $FFMPEG_PID || true

echo "Stopping services..."
kill $UI_PID || true
kill $WATCHDOG_PID || true
kill $XVFB_PID || true

# 5. Generate Test Report
echo "Generating Test Report..."
python3 generate_report.py

echo "QA Pipeline Completed Successfully!"
