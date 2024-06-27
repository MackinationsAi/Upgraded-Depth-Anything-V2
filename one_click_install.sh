#!/bin/bash

# Check if virtual environment folder exists
if [ ! -d "venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv venv
fi

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip to the latest version
python3 -m pip install --upgrade pip --no-cache-dir

# Create checkpoints directory if it does not exist
if [ ! -d "checkpoints" ]; then
  mkdir checkpoints
fi

# Function to download models
download_models() {
  echo "Downloading Depth-Anything-V2-Small model..."
  curl -L -o checkpoints/depth_anything_v2_vits.safetensors "https://huggingface.co/MackinationsAi/Depth-Anything-V2_Safetensors/resolve/main/depth_anything_v2_vits.safetensors?download=true"
  echo "Downloading Depth-Anything-V2-Base model..."
  curl -L -o checkpoints/depth_anything_v2_vitb.safetensors "https://huggingface.co/MackinationsAi/Depth-Anything-V2_Safetensors/resolve/main/depth_anything_v2_vitb.safetensors?download=true"
  echo "Downloading Depth-Anything-V2-Large model..."
  curl -L -o checkpoints/depth_anything_v2_vitl.safetensors "https://huggingface.co/MackinationsAi/Depth-Anything-V2_Safetensors/resolve/main/depth_anything_v2_vitl.safetensors?download=true"
}

# Detect the operating system
OS="$(uname -s)"
case "$OS" in
  Linux*)   
    echo "Linux detected"
    download_models
    echo "Installing PyTorch with CUDA support..."
    pip install torch==2.0.0+cu121 torchvision==0.18.0+cu121 torchaudio==2.0.6.post1 --index-url https://download.pytorch.org/whl/cu121 --no-cache-dir
    ;;
  Darwin*)  
    echo "macOS detected"
    download_models
    echo "Installing PyTorch without CUDA support..."
    pip install torch torchvision torchaudio --no-cache-dir
    ;;
  *)        
    echo "Unsupported OS"
    exit 1
    ;;
esac

# Install other dependencies from requirements.txt
echo "Installing other dependencies..."
pip install -r requirements.txt --no-cache-dir