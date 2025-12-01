@echo off
REM ============================================================
REM GSR Training Pipeline - Windows Batch Script
REM ============================================================
REM This script runs the complete GSR training pipeline:
REM 1. Build subgraph index
REM 2. Prepare training data
REM 3. Train GSR model
REM ============================================================

setlocal

REM Configuration
set CONDA_ENV=%1
if "%CONDA_ENV%"=="" set CONDA_ENV=Wu

set DATA_DIR=..\Data\webqsp_final
set OUTPUT_DIR=outputs_gsr
set GSR_DATA_DIR=%DATA_DIR%\gsr_data

set TRAIN_DATA=%DATA_DIR%\train.parquet
set VAL_DATA=%DATA_DIR%\val.parquet
set TEST_DATA=%DATA_DIR%\test.parquet

echo ============================================================
echo GSR Training Pipeline
echo ============================================================
echo Conda Environment: %CONDA_ENV%
echo Train Data: %TRAIN_DATA%
echo Output Directory: %OUTPUT_DIR%
echo ============================================================

REM Activate conda environment
call conda activate %CONDA_ENV%
if errorlevel 1 (
    echo ERROR: Failed to activate conda environment "%CONDA_ENV%"
    exit /b 1
)

REM Step 1: Build subgraph index
echo.
echo [Step 1/3] Building subgraph index...
python gsr\subgraph_index.py ^
    --data_path "%TRAIN_DATA%" ^
    --output_path "%GSR_DATA_DIR%\subgraph_index.json" ^
    --min_frequency 1

if errorlevel 1 (
    echo ERROR: Failed to build subgraph index
    exit /b 1
)

REM Step 2: Prepare GSR training data
echo.
echo [Step 2/3] Preparing GSR training data...
python gsr\prepare_gsr_data.py ^
    --train_data "%TRAIN_DATA%" ^
    --output_dir "%GSR_DATA_DIR%"

if errorlevel 1 (
    echo ERROR: Failed to prepare GSR training data
    exit /b 1
)

REM Step 3: Train GSR model
echo.
echo [Step 3/3] Training GSR model...
python gsr\train_gsr.py ^
    --train_data "%GSR_DATA_DIR%\gsr_training_data.jsonl" ^
    --val_data "%GSR_DATA_DIR%\gsr_val_data.jsonl" ^
    --model_name t5-small ^
    --output_dir "%OUTPUT_DIR%" ^
    --batch_size 16 ^
    --num_epochs 10 ^
    --learning_rate 1e-4 ^
    --max_input_length 512 ^
    --max_target_length 128

if errorlevel 1 (
    echo ERROR: Training failed
    exit /b 1
)

echo.
echo ============================================================
echo Training complete!
echo ============================================================
echo Model saved to: %OUTPUT_DIR%
echo Subgraph index: %GSR_DATA_DIR%\subgraph_index.json
echo ============================================================
pause

