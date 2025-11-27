@echo off
REM ============================================================
REM KG Path Diffusion Model - Training Script (Windows)
REM ============================================================
REM Usage: run_train.bat [conda_env_name]
REM   Example: run_train.bat Wu
REM ============================================================

setlocal

REM Default conda environment
set CONDA_ENV=%1
if "%CONDA_ENV%"=="" set CONDA_ENV=Wu

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

REM Run training with recommended settings
python train.py ^
    --train_data ../Data/webqsp_final/train.parquet ^
    --val_data ../Data/webqsp_final/val.parquet ^
    --vocab_path ../Data/webqsp_final/vocab.json ^
    --batch_size 4 ^
    --hidden_dim 128 ^
    --num_graph_layers 1 ^
    --num_diffusion_layers 1 ^
    --num_diffusion_steps 25 ^
    --max_path_length 20 ^
    --gpus 1 ^
    --output_dir outputs_multipath ^
    --max_epochs 50

echo.
echo ============================================================
echo Training complete!
echo ============================================================
pause

