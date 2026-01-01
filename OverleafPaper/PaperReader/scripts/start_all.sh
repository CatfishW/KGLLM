#!/bin/bash
# ============================================
# Paper Reader - Complete Deployment Script
# Creates screen sessions for both backend and tunnel
# ============================================

echo "╔════════════════════════════════════════════╗"
echo "║   PAPER READER - SCREEN SESSION MANAGER    ║"
echo "╚════════════════════════════════════════════╝"
echo ""

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Kill existing sessions if they exist
echo "[INFO] Checking for existing sessions..."
screen -S paperreader-backend -X quit 2>/dev/null && echo "[INFO] Killed existing backend session"
screen -S paperreader-tunnel -X quit 2>/dev/null && echo "[INFO] Killed existing tunnel session"

sleep 1

echo ""
echo "[INFO] Starting backend server in screen session 'paperreader-backend'..."
screen -dmS paperreader-backend bash -c "cd $SCRIPT_DIR && chmod +x start_backend.sh && ./start_backend.sh; exec bash"

# Wait for backend to start
echo "[INFO] Waiting for backend to initialize..."
sleep 5

# Check if backend is running
if curl -s --max-time 2 http://127.0.0.1:22222/health >/dev/null 2>&1; then
    echo "[OK] Backend server is running!"
else
    echo "[WARN] Backend may not be ready yet. Check the screen session."
fi

echo ""
echo "[INFO] Starting SSH tunnel in screen session 'paperreader-tunnel'..."
screen -dmS paperreader-tunnel bash -c "cd $SCRIPT_DIR && chmod +x tunnel.sh && ./tunnel.sh; exec bash"

echo ""
echo "╔════════════════════════════════════════════╗"
echo "║   DEPLOYMENT COMPLETE                      ║"
echo "╚════════════════════════════════════════════╝"
echo ""
echo "Screen sessions started:"
screen -ls | grep paperreader || echo "  (no sessions found)"
echo ""
echo "Commands:"
echo "  Attach to backend: screen -r paperreader-backend"
echo "  Attach to tunnel:  screen -r paperreader-tunnel"
echo "  List sessions:     screen -ls"
echo "  Detach:            Ctrl+A, then D"
echo ""
echo "Your Paper Reader should be accessible at:"
echo "  http://game.agaii.org/paperreader/"
echo ""
