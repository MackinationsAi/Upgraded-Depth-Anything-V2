@echo off
REM Activate the virtual environment
CALL venv\Scripts\activate

REM Run the enhanced video depth processing script
python run_video-depth_enhanced.py %*

REM Keep the window open to see any output
pause