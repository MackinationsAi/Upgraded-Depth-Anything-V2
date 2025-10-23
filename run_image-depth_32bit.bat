@echo off
REM Activate the virtual environment
CALL venv\Scripts\activate

REM Run the 32-bit depth processing script
python run_image-depth_32bit.py %*

REM Keep the window open to see any output
pause