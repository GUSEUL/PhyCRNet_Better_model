@echo off
REM Complete PhyCRNet Training and Analysis Pipeline
REM This script runs the complete training pipeline in sequence

echo ========================================
echo PhyCRNet Complete Training Pipeline
echo ========================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo Step 1/3: Starting complete physics training...
echo ========================================
python train_complete_physics.py
if errorlevel 1 (
    echo ERROR: Training failed!
    pause
    exit /b 1
)
echo Training completed successfully!
echo.

echo Step 2/3: Creating training plots...
echo ========================================
python create_training_plots.py
if errorlevel 1 (
    echo ERROR: Training plots creation failed!
    pause
    exit /b 1
)
echo Training plots created successfully!
echo.

echo Step 3/3: Creating complete animations...
echo ========================================
python create_complete_animation.py
if errorlevel 1 (
    echo ERROR: Animation creation failed!
    pause
    exit /b 1
)
echo Animations created successfully!
echo.

echo ========================================
echo Pipeline completed successfully!
echo ========================================
echo All steps completed:
echo - Training: train_complete_physics.py ✓
echo - Plots: create_training_plots.py ✓  
echo - Animations: create_complete_animation.py ✓
echo.
echo Check the following directories for results:
echo - complete_physics_results/ (training results and plots)
echo - training_plots/ (training analysis plots)
echo - complete_animations/ (generated animations)
echo.
pause