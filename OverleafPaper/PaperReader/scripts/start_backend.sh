#!/bin/bash
# ============================================
# Paper Reader - Backend Server Starter
# For Linux/Ubuntu
# Run in screen: screen -S paperreader-backend ./start_backend.sh
# ============================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$SCRIPT_DIR/../backend"

echo "╔════════════════════════════════════════════╗"
echo "║   PAPER READER - BACKEND SERVER            ║"
echo "╚════════════════════════════════════════════╝"
echo ""

cd "$BACKEND_DIR" || exit 1

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "[INFO] Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "[INFO] Installing dependencies..."
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

echo "[INFO] Starting FastAPI server on port 22222..."
echo ""

uvicorn main:app --host 0.0.0.0 --port 22222
