@echo off
REM ============================================================
REM GSR Model Training Script
REM ============================================================

setlocal

set CONDA_ENV=%1
if "%CONDA_ENV%"=="" set CONDA_ENV=Wu

set TRAIN_DATA=..\Data\webqsp_final\gsr_data\gsr_training_data.jsonl
set VAL_DATA=..\Data\webqsp_final\gsr_data\gsr_val_data.jsonl
set OUTPUT_DIR=outputs_gsr

echo ============================================================
echo GSR Model Training
echo ============================================================
echo Conda Environment: %CONDA_ENV%
echo Train Data: %TRAIN_DATA%
echo Val Data: %VAL_DATA%
echo Output: %OUTPUT_DIR%
echo ============================================================

call conda activate %CONDA_ENV%
if errorlevel 1 (
    echo ERROR: Failed to activate conda environment
    exit /b 1
)

python gsr\train_gsr.py ^
    --train_data "%TRAIN_DATA%" ^
    --val_data "%VAL_DATA%" ^
    --model_name t5-small ^
    --output_dir "%OUTPUT_DIR%" ^
    --batch_size 16 ^
    --num_epochs 10 ^
    --learning_rate 1e-4 ^
    --max_input_length 512 ^
    --max_target_length 128 ^
    --warmup_steps 1000

if errorlevel 1 (
    echo.
    echo ERROR: Training failed
    echo This may be due to torch DLL issues on Windows
    echo Try running in a different environment or check torch installation
    pause
    exit /b 1
)

echo.
echo ============================================================
echo Training complete!
echo ============================================================
echo Model saved to: %OUTPUT_DIR%
echo ============================================================
pause

