#!/bin/bash
echo "Starting One-Shot Learning Interactive Playground with memory optimizations..."

# Set environment variables to reduce memory usage
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=3
export PYTHONOPTIMIZE=1

# Run the application with reduced memory settings
python app.py