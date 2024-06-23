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

REM Create checkpoints directory if it does not exist
IF NOT EXIST checkpoints (
    mkdir checkpoints
)

REM Download models and place them into the checkpoints folder & Download Triton wheel and place it in the main tree folder
echo Downloading Depth-Anything-V2-Small model...
curl -L -o checkpoints/depth_anything_v2_vits.safetensors https://huggingface.co/MackinationsAi/Depth-Anything-V2_Safetensors/resolve/main/depth_anything_v2_vits.safetensors?download=true
echo Downloading Depth-Anything-V2-Base model...
curl -L -o checkpoints/depth_anything_v2_vitb.safetensors https://huggingface.co/MackinationsAi/Depth-Anything-V2_Safetensors/resolve/main/depth_anything_v2_vitb.safetensors?download=true
echo Downloading Depth-Anything-V2-Large model...
curl -L -o checkpoints/depth_anything_v2_vitl.safetensors https://huggingface.co/MackinationsAi/Depth-Anything-V2_Safetensors/resolve/main/depth_anything_v2_vitl.safetensors?download=true
echo Downloading Triton==2.1.0 .whl...
curl -L -o triton-2.1.0-cp310-cp310-win_amd64.whl https://huggingface.co/MonsterMMORPG/SECourses/resolve/main/triton-2.1.0-cp310-cp310-win_amd64.whl?download=true

REM Installing torch, dependencies from requirements.txt & triton
echo Installing dependencies...
pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir
pip install -r requirements.txt --no-cache-dir
pip install triton-2.1.0-cp310-cp310-win_amd64.whl --no-cache-dir

REM Deactivate the virtual environment
CALL venv\scripts\deactivate
pause
exit
