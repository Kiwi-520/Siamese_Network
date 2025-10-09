#!/bin/bash
echo "Starting One-Shot Learning Playground..."

# Clear any existing Python processes to free memory
echo "Cleaning up resources..."
pkill -f python || true
sleep 2

# Set memory-saving environment variables
export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_CPP_MIN_LOG_LEVEL=2

# Change to the script directory
cd "$(dirname "$0")"

# Run the application
echo "Starting application..."
python app.py

# If the application fails, try again with reduced features
if [ $? -ne 0 ]; then
  echo "Application failed to start. Trying with minimal mode..."
  export REDUCE_FEATURES=1
  python app.py
fi