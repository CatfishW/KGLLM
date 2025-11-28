@echo off
REM ============================================================
REM KG Path Diffusion Model - Training Script (Windows)
REM ============================================================
REM Usage: run_train.bat [conda_env_name] [config_path]
REM   Example: run_train.bat Wu configs\flow_matching_base.yaml
REM ============================================================

setlocal

REM Default conda environment
set CONDA_ENV=%1
if "%CONDA_ENV%"=="" set CONDA_ENV=Wu

set TRAIN_CONFIG=.\configs\diffusion.yaml
@REM if "%TRAIN_CONFIG%"=="" set TRAIN_CONFIG=.\configs\flow_matching_base.yaml


echo Config File    : %TRAIN_CONFIG%
echo ============================================================
echo KG Path Diffusion Model - Training
echo ============================================================
echo Conda Environment: %CONDA_ENV%
echo ============================================================

REM Activate conda environment
call conda activate %CONDA_ENV%
if errorlevel 1 (
    echo ERROR: Failed to activate conda environment "%CONDA_ENV%"
    echo Please ensure conda is installed and the environment exists.
    exit /b 1
)

REM Run training with config-based settings
python train.py --config "%TRAIN_CONFIG%"

echo.
echo ============================================================
echo Training complete!
echo ============================================================
pause

