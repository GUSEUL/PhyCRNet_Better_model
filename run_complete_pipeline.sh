#!/bin/bash

# Complete PhyCRNet Training and Analysis Pipeline
# This script runs the complete training pipeline in sequence

echo "========================================"
echo "PhyCRNet Complete Training Pipeline"
echo "========================================"
echo

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "ERROR: Python is not installed or not in PATH"
    exit 1
fi

# Function to check script execution
check_status() {
    if [ $? -ne 0 ]; then
        echo "ERROR: $1 failed!"
        exit 1
    fi
    echo "$1 completed successfully!"
    echo
}

echo "Step 1/3: Starting complete physics training..."
echo "========================================"
python train_complete_physics.py
check_status "Training"

echo "Step 2/3: Creating training plots..."
echo "========================================"
python create_training_plots.py
check_status "Training plots creation"

echo "Step 3/3: Creating complete animations..."
echo "========================================"
python create_complete_animation.py
check_status "Animation creation"

echo "========================================"
echo "Pipeline completed successfully!"
echo "========================================"
echo "All steps completed:"
echo "- Training: train_complete_physics.py ✓"
echo "- Plots: create_training_plots.py ✓"
echo "- Animations: create_complete_animation.py ✓"
echo
echo "Check the following directories for results:"
echo "- complete_physics_results/ (training results and plots)"
echo "- training_plots/ (training analysis plots)"
echo "- complete_animations/ (generated animations)"
echo