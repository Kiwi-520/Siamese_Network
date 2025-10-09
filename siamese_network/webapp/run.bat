@echo off
echo Starting One-Shot Learning Playground...

:: Clear any existing Python processes to free memory
echo Cleaning up resources...
taskkill /F /IM python.exe /T 2>NUL
timeout /T 2 /NOBREAK >NUL

:: Set memory-saving environment variables
set TF_FORCE_GPU_ALLOW_GROWTH=true
set TF_CPP_MIN_LOG_LEVEL=2

:: Change to the script directory
cd %~dp0

:: Run the application
echo Starting application...
python app.py

:: If the application fails, try again with reduced features
if %ERRORLEVEL% NEQ 0 (
  echo Application failed to start. Trying with minimal mode...
  set REDUCE_FEATURES=1
  python app.py
)