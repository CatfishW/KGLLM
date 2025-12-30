#!/bin/bash
# Reverse SSH Tunnel Maintainer (Ubuntu)

REMOTE_USER="lobin"
REMOTE_HOST="vpn.agaii.org"
REMOTE_PORT=32026
LOCAL_PORT=32026
RECONNECT_DELAY=5

echo "=== Reverse SSH Tunnel ==="
echo "Forwarding: $REMOTE_HOST:$REMOTE_PORT -> 127.0.0.1:$LOCAL_PORT"

# Check local server (wait up to 30s)
for i in {1..15}; do
    curl -s --max-time 2 http://127.0.0.1:$LOCAL_PORT/health >/dev/null 2>&1 && break
    echo "[INFO] Waiting for local server on port $LOCAL_PORT... ($i/15)"
    sleep 2
done

# Final check
curl -s --max-time 2 http://127.0.0.1:$LOCAL_PORT/health >/dev/null 2>&1 || \
    echo "[WARN] Local server still not responding on port $LOCAL_PORT. Tunnel may be idle."

# Main loop
ATTEMPT=0
while true; do
    ((ATTEMPT++))
    echo "[$(date +%H:%M:%S)] Starting tunnel (attempt #$ATTEMPT)..."
    
    ssh -R $REMOTE_PORT:127.0.0.1:$LOCAL_PORT \
        -o ServerAliveInterval=30 \
        -o ServerAliveCountMax=3 \
        -o ExitOnForwardFailure=yes \
        -o StrictHostKeyChecking=no \
        -N $REMOTE_USER@$REMOTE_HOST
    
    [[ $? -eq 0 ]] && break
    echo "[WARN] Disconnected. Reconnecting in ${RECONNECT_DELAY}s..."
    sleep $RECONNECT_DELAY
done

