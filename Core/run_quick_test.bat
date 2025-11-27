@echo off
REM ============================================================
REM Quick Test - Run inference on a few samples (Windows)
REM ============================================================
REM Usage: run_quick_test.bat [conda_env_name]
REM ============================================================

setlocal

set CONDA_ENV=%1
if "%CONDA_ENV%"=="" set CONDA_ENV=Wu

echo ============================================================
echo Quick Test - Inference on 10 samples
echo ============================================================

call conda activate %CONDA_ENV%
if errorlevel 1 (
    echo ERROR: Failed to activate conda environment "%CONDA_ENV%"
    exit /b 1
)

python inference.py ^
    --checkpoint outputs_1/checkpoints/last.ckpt ^
    --vocab outputs_1/vocab.json ^
    --data ../Data/webqsp_combined/val.jsonl ^
    --output quick_test_results.jsonl ^
    --max_samples 10 ^
    --batch_size 2 ^
    --show_examples 10

pause

