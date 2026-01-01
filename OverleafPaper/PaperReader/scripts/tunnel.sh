#!/bin/bash
# ============================================
# Paper Reader - Reverse SSH Tunnel Maintainer
# For Linux/Ubuntu
# Run in screen: screen -S paperreader-tunnel ./tunnel.sh
# ============================================

REMOTE_USER="lobin"
REMOTE_HOST="vpn.agaii.org"
REMOTE_PORT=22222
LOCAL_PORT=22222
RECONNECT_DELAY=5

echo "╔════════════════════════════════════════════╗"
echo "║   PAPER READER - SSH TUNNEL MAINTAINER     ║"
echo "╚════════════════════════════════════════════╝"
echo ""
echo "Forwarding: $REMOTE_HOST:$REMOTE_PORT -> 127.0.0.1:$LOCAL_PORT"
echo ""

# Check if local server is responding
check_local_server() {
    if curl -s --max-time 2 http://127.0.0.1:$LOCAL_PORT/health >/dev/null 2>&1; then
        echo "[OK] Local server is responding on port $LOCAL_PORT"
    else
        echo "[WARN] Local server not responding on port $LOCAL_PORT"
        echo "[INFO] Make sure to start the backend server first!"
    fi
}

check_local_server

# Main reconnection loop
ATTEMPT=0
while true; do
    ((ATTEMPT++))
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting tunnel (attempt #$ATTEMPT)..."
    
    ssh -R $REMOTE_PORT:127.0.0.1:$LOCAL_PORT \
        -o ServerAliveInterval=30 \
        -o ServerAliveCountMax=3 \
        -o ExitOnForwardFailure=yes \
        -o StrictHostKeyChecking=no \
        -o ConnectTimeout=10 \
        -N $REMOTE_USER@$REMOTE_HOST
    
    EXIT_CODE=$?
    
    if [[ $EXIT_CODE -eq 0 ]]; then
        echo "[INFO] Tunnel closed gracefully."
        break
    fi
    
    echo "[WARN] Tunnel disconnected (exit code: $EXIT_CODE). Reconnecting in ${RECONNECT_DELAY}s..."
    sleep $RECONNECT_DELAY
done

echo "[INFO] Tunnel maintainer stopped."
