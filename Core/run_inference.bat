@echo off
REM ============================================================
REM KG Path Diffusion Model - Inference Script (Windows)
REM ============================================================
REM Usage: run_inference.bat [conda_env_name] [checkpoint_dir]
REM   Example: run_inference.bat Wu outputs_1
REM ============================================================

setlocal

REM Default values
set CONDA_ENV=%1
set CKPT_DIR=%2
if "%CONDA_ENV%"=="" set CONDA_ENV=Wu
if "%CKPT_DIR%"=="" set CKPT_DIR=outputs_improved

echo ============================================================
echo KG Path Diffusion Model - Inference
echo ============================================================
echo Conda Environment: %CONDA_ENV%
echo Checkpoint Dir: %CKPT_DIR%
echo ============================================================

REM Activate conda environment
call conda activate %CONDA_ENV%
if errorlevel 1 (
    echo ERROR: Failed to activate conda environment "%CONDA_ENV%"
    exit /b 1
)

REM Check if checkpoint exists
if not exist "%CKPT_DIR%\checkpoints\last.ckpt" (
    echo ERROR: Checkpoint not found at %CKPT_DIR%\checkpoints\last.ckpt
    echo Please train the model first or specify the correct checkpoint directory.
    exit /b 1
)

REM Run inference
python inference.py ^
    --checkpoint %CKPT_DIR%/checkpoints/last.ckpt ^
    --vocab %CKPT_DIR%/vocab.json ^
    --data ../Data/webqsp_combined/val.jsonl ^
    --output inference_results.jsonl ^
    --batch_size 8 ^
    --path_length 10 ^
    --temperature 0.7 ^
    --show_examples 10

echo.
echo ============================================================
echo Inference complete! Results saved to inference_results.jsonl
echo ============================================================
pause

