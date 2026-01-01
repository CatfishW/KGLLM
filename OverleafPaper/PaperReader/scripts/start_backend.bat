@echo off
REM ============================================
REM Paper Reader - Backend Server Starter
REM For Windows
REM ============================================

echo ========================================
echo   PAPER READER - BACKEND SERVER
echo ========================================
echo.

cd /d "%~dp0..\backend"

REM Check if virtual environment exists
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo [INFO] Installing dependencies...
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

echo [INFO] Starting FastAPI server on port 22222...
echo.

uvicorn main:app --host 0.0.0.0 --port 22222 --reload
