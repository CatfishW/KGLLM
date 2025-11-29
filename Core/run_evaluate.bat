@echo off
REM ============================================================
REM Evaluation Script with Metrics (Windows)
REM ============================================================
REM Usage: run_evaluate.bat [conda_env_name] [checkpoint_path] [vocab_path] [test_data]
REM   Example: run_evaluate.bat Wu outputs_multipath_diffusion_relation_only_2/checkpoints/last.ckpt outputs_multipath_diffusion_relation_only_2/vocab.json ../Data/webqsp_final/val.parquet
REM ============================================================

setlocal

REM Default conda environment
set CONDA_ENV=%1
if "%CONDA_ENV%"=="" set CONDA_ENV=Wu

set CHECKPOINT=%2
if "%CHECKPOINT%"=="" set CHECKPOINT=outputs_multipath_diffusion_relation_only_2/checkpoints/last.ckpt

set VOCAB=%3
if "%VOCAB%"=="" set VOCAB=outputs_multipath_diffusion_relation_only_2/vocab.json

set TEST_DATA=%4
if "%TEST_DATA%"=="" set TEST_DATA=../Data/webqsp_final/val.parquet

echo ============================================================
echo KG Path Diffusion Model - Evaluation with Metrics
echo ============================================================
echo Conda Environment: %CONDA_ENV%
echo Checkpoint: %CHECKPOINT%
echo Vocabulary: %VOCAB%
echo Test Data: %TEST_DATA%
echo ============================================================

REM Activate conda environment
call conda activate %CONDA_ENV%
if errorlevel 1 (
    echo ERROR: Failed to activate conda environment "%CONDA_ENV%"
    echo Please ensure conda is installed and the environment exists.
    exit /b 1
)

REM Run evaluation
python evaluate_with_metrics.py --checkpoint "%CHECKPOINT%" --vocab "%VOCAB%" --test_data "%TEST_DATA%" --max_examples 100 --num_samples 5 --batch_size 8

echo.
echo ============================================================
echo Evaluation complete!
echo ============================================================
pause

