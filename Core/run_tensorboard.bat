@echo off
REM ============================================================
REM TensorBoard Launcher for KG Path Diffusion Training Logs
REM ============================================================

setlocal

set LOG_DIR=outputs_multipath_diffusion_relation_only
set PORT=6006

echo ============================================================
echo Starting TensorBoard
echo ============================================================
echo Log directory: %LOG_DIR%
echo Port: %PORT%
echo ============================================================
echo.
echo TensorBoard will be available at: http://localhost:%PORT%
echo Press Ctrl+C to stop TensorBoard
echo ============================================================
echo.

cd /d "%~dp0"
tensorboard --logdir %LOG_DIR% --port %PORT%

pause

