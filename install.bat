@echo off
REM Check if the virtual environment folder exists
IF NOT EXIST venv (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate the virtual environment
echo Activating virtual environment...
CALL venv\Scripts\activate

REM Upgrade pip to the latest version
python.exe -m pip install --upgrade pip --no-cache-dir

REM Install dependencies from requirements.txt
echo Installing dependencies...
pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir
pip install -r requirements.txt --no-cache-dir
pip install triton-2.1.0-cp310-cp310-win_amd64.whl --no-cache-dir

REM Deactivate the virtual environment
CALL venv\scripts\deactivate
pause
exit