@echo off
REM ============================================
REM Paper Reader - Reverse SSH Tunnel Maintainer
REM For Windows
REM ============================================

setlocal enabledelayedexpansion

set REMOTE_USER=lobin
set REMOTE_HOST=vpn.agaii.org
set REMOTE_PORT=22222
set LOCAL_PORT=22222
set RECONNECT_DELAY=5

echo ========================================
echo   PAPER READER - SSH TUNNEL MAINTAINER
echo ========================================
echo.
echo Forwarding: %REMOTE_HOST%:%REMOTE_PORT% -^> 127.0.0.1:%LOCAL_PORT%
echo.

REM Check if local server is responding
curl -s --max-time 2 http://127.0.0.1:%LOCAL_PORT%/health >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Local server is responding on port %LOCAL_PORT%
) else (
    echo [WARN] Local server not responding on port %LOCAL_PORT%
    echo [INFO] Make sure to start the backend server first!
)

REM Main reconnection loop
set ATTEMPT=0

:loop
set /a ATTEMPT+=1
echo.
echo [%TIME%] Starting tunnel (attempt #%ATTEMPT%)...

ssh -R %REMOTE_PORT%:127.0.0.1:%LOCAL_PORT% ^
    -o ServerAliveInterval=30 ^
    -o ServerAliveCountMax=3 ^
    -o ExitOnForwardFailure=yes ^
    -o StrictHostKeyChecking=no ^
    -o ConnectTimeout=10 ^
    -N %REMOTE_USER%@%REMOTE_HOST%

if %errorlevel% equ 0 (
    echo [INFO] Tunnel closed gracefully.
    goto :end
)

echo [WARN] Tunnel disconnected. Reconnecting in %RECONNECT_DELAY%s...
timeout /t %RECONNECT_DELAY% /nobreak >nul
goto :loop

:end
echo [INFO] Tunnel maintainer stopped.
pause
