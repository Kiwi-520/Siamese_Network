@echo off
echo Starting One-Shot Learning Interactive Playground with memory optimizations...

rem Set environment variables to reduce memory usage
set TF_FORCE_GPU_ALLOW_GROWTH=true
set TF_CPP_MIN_LOG_LEVEL=3
set PYTHONOPTIMIZE=1

rem Run the application with reduced memory settings
python app.py

pause