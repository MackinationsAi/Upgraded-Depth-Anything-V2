# Upgraded Depth Anything V2 (UDAV2) AI Agent Instructions

## Project Overview
This is an enhanced version of Depth Anything V2 for monocular depth estimation, featuring safetensors models, upgraded Gradio WebUI, and both CLI/batch processing capabilities. The project supports multiple model sizes (Small/Base/Large) and provides both relative and metric depth estimation with **extended bit depth support (8/16/24/32-bit) and seamless processing capabilities**.

## Architecture Components

### Core Models (`depth_anything_v2/`)
- **`dpt.py`**: Main DepthAnythingV2 class with DPT (Dense Prediction Transformer) head
- **`dinov2.py`**: DINOv2 vision transformer backbone from Meta (Apache 2.0 licensed)
- **`dinov2_layers/`**: Transformer components (attention, blocks, MLPs) with SwiGLU activation
- Model configs define encoder variants: `vits` (48M), `vitb` (190M), `vitl` (655M), `vitg` (1.3B-WIP)

### Entry Points & Usage Patterns
- **Gradio WebUI**: `run_gradio.py` - Interactive web interface with tabs for single/batch image/video processing
- **CLI Scripts**: 
  - Standard: `run_image-depth_8bit.py`, `run_image-depth_16bit.py`, `run_video-depth.py`
  - **Enhanced**: `run_image-depth_24bit.py`, `run_image-depth_32bit.py`, `run_video-depth_enhanced.py`
- **Metric Depth**: `metric_depth/run.py` - For absolute depth values using fine-tuned models on Hypersim/VKITTI

### Model Loading Convention
```python
# Standard pattern used throughout codebase
from safetensors.torch import load_file
state_dict = load_file(f'checkpoints/depth_anything_v2_{encoder}.safetensors')
model.load_state_dict(state_dict)
```

### Device Management Pattern
```python
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
```

## Enhanced Features

### Bit Depth Support
- **8-bit**: `depth * 255.0` → `np.uint8` (range 0-255) - Standard PNG output
- **16-bit**: `depth * 65535.0` → `np.uint16` (range 0-65535) - High precision PNG
- **24-bit**: `depth * 16777215.0` → `np.uint32` (range 0-16M) - TIFF format output
- **32-bit**: `depth * 4294967295.0` → `np.uint32` or `np.float32` - Maximum precision TIFF

### Seamless Processing
```python
def apply_seamless_processing(depth_map, method='edge_blend', blend_width=32):
    # Edge blending for tiling capabilities
    # Methods: 'edge_blend', 'periodic', 'mirror', 'gaussian_blend'
```
- **edge_blend**: Blends opposing edges for seamless tiling
- **periodic**: Ensures periodic boundary conditions
- **mirror**: Mirror boundary conditions for smooth transitions
- **gaussian_blend**: Gaussian-weighted blending for smoother results

## Critical Workflows

### Installation (Windows/Cross-platform)
- **Windows**: Run `oc_install.bat` - Creates venv, downloads models to `checkpoints/`, installs PyTorch+CUDA
- **macOS/Linux**: Use `oc_install.sh` or manual `pip install requirements_macos.txt`
- Models auto-downloaded from HuggingFace (MackinationsAi/Depth-Anything-V2_Safetensors)

### Running Applications
- **CLI**: Double-click `.bat` files on Windows or run Python scripts directly
- **Enhanced CLI Options**:
  ```bash
  python run_image-depth_24bit.py --seamless --input-size 1024
  python run_image-depth_32bit.py --output-format float32 --seamless
  python run_video-depth_enhanced.py --bit-depth 16 --temporal-smoothing --seamless
  ```
- **Gradio**: `run_gradio.bat` or `python run_gradio.py` - Now includes bit depth and seamless controls

### Output Structure
- **Standard**: `vis_img_depth/`, `video_depth_vis/`, `outputs/`
- **Enhanced**: `vis_img_depth_24bit/`, `vis_img_depth_32bit/`, `video_depth_vis_enhanced/`
- **High Bit Depth**: TIFF format for 24/32-bit, PNG for 8/16-bit
- **Naming**: `{name}_depth_{bit}bit.tiff`, `{name}_depth_greyscale.png`, `{name}_combined.png`

## Key Patterns & Conventions

### Enhanced Model Configuration
```python
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
}
```

### Enhanced Depth Processing Pipeline
1. Load image with `cv2.imread()` (BGR format)
2. Call `model.infer_image(image, input_size)` - handles RGB conversion internally
3. **Optional**: Apply seamless processing: `apply_seamless_processing(depth, method)`
4. Normalize depth: `(depth - depth.min()) / (depth.max() - depth.min())`
5. **Bit Depth Scaling**: Scale by bit depth max value (255, 65535, 16777215, 4294967295)
6. **Format Selection**: PNG for ≤16-bit, TIFF for >16-bit
7. Apply colormap: `matplotlib.colormaps.get_cmap('Spectral_r')(depth_normalized)`

### Temporal Processing (Video)
```python
def apply_temporal_smoothing(current_depth, previous_depth, smoothing_factor=0.3):
    # Reduces flickering in video sequences
    return (1 - smoothing_factor) * current_depth + smoothing_factor * previous_depth
```

### Path Handling
- Scripts handle both single files and directory batch processing
- Use `glob.glob(pattern, recursive=True)` for directory traversal
- Quote removal utility: `remove_double_quotes()` for user input sanitization

## Dependencies & Environment
- **PyTorch**: 2.3.0+cu121 with CUDA support
- **Triton**: Custom wheel (triton-2.1.0-cp310-cp310-win_amd64.whl) for Windows
- **safetensors**: Preferred over .pth for model loading
- **Gradio**: 4.29.0 with imageslider component for UI
- **Optional**: scipy (for gaussian_blend seamless processing)

## Debugging & Development
- CUDA environment variables set to avoid cuDNN warnings
- Warning filters applied for cudnnStatus messages
- Models expect specific input sizes (default 518, configurable up to 1024)
- Larger models (vitl) provide better temporal consistency for videos
- **High bit depth files**: Use image viewers that support TIFF format for >16-bit

## Extension Points
- Metric depth training: `metric_depth/train.py` with Hypersim/VKITTI datasets
- Point cloud generation: `metric_depth/depth_to_pointcloud.py`
- A1111/Forge extensions available separately (sd-webui-udav2)
- **New**: Custom seamless processing methods can be added to `apply_seamless_processing()`
- **New**: Additional bit depth formats can be extended in processing functions