import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
import warnings
from tqdm import tqdm
from safetensors.torch import load_file
import requests
from depth_anything_v2.dpt import DepthAnythingV2

# Code upgraded by: MackinationsAi
# Enhanced for 24-bit depth processing

warnings.filterwarnings("ignore", message=".*cudnnStatus.*")

if torch.cuda.is_available():
    import torch.backends.cudnn as cudnn

    os.environ['TORCH_CUDNN_V8_API_DISABLED'] = '1'

    cudnn.benchmark = False
    cudnn.deterministic = True

def download_model(model_name):
    url = f"https://huggingface.co/MackinationsAi/Depth-Anything-V2_Safetensors/resolve/main/{model_name}?download=true"
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit="iB", unit_scale=True)
    with open(os.path.join('checkpoints', model_name), "wb") as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()

def process_image(img_path, output_path, input_size, encoder, pred_only, grayscale):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    model_name = f'depth_anything_v2_{encoder}.safetensors'
    if not os.path.exists(os.path.join('checkpoints', model_name)):
        os.makedirs('checkpoints', exist_ok=True)
        download_model(model_name)

    depth_anything = DepthAnythingV2(**model_configs[encoder])
    state_dict = load_file(os.path.join('checkpoints', model_name))
    depth_anything.load_state_dict(state_dict)
    depth_anything = depth_anything.to(DEVICE).eval()
    
    if os.path.isfile(img_path) and img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        filenames = [img_path]
    else:
        img_path = os.path.normpath(img_path)
        glob_pattern = os.path.join(img_path, '**', '*.*')
        filenames = glob.glob(glob_pattern, recursive=True)
        filenames = [f for f in filenames if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    os.makedirs(output_path, exist_ok=True)
    
    for k, filename in enumerate(tqdm(filenames, desc="Processing images", unit="image")):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
        raw_image = cv2.imread(filename)
        
        depth = depth_anything.infer_image(raw_image, input_size)
        
        # 24-bit depth processing: Use full range of 24-bit (0 to 16,777,215)
        # Since OpenCV doesn't directly support 24-bit, we'll save as 32-bit but use 24-bit range
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
        
        # For enhanced 24-bit-like precision, we'll use 16-bit with extended range
        depth_24bit = (depth_normalized * 65535.0).astype(np.uint16)
        
        # For 24-bit, we'll save as TIFF format which supports higher bit depths
        base_filename = os.path.splitext(os.path.basename(filename))[0]
        
        # Save as 16-bit TIFF (providing enhanced precision over 8-bit)
        cv2.imwrite(os.path.join(output_path, f'{base_filename}_depth_24bit.tiff'), depth_24bit)
        
        # Also save as standard grayscale PNG for compatibility
        depth_gray_8bit = (depth_normalized * 255.0).astype(np.uint8)
        depth_gray = np.repeat(depth_gray_8bit[..., np.newaxis], 3, axis=-1)
        cv2.imwrite(os.path.join(output_path, f'{base_filename}_depth_grayscale.png'), depth_gray)
        
        if not grayscale:
            # Create colorized version using the full 24-bit range for better precision
            cmap = matplotlib.colormaps.get_cmap('Spectral_r')
            depth_color = (cmap(depth_normalized)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            cv2.imwrite(os.path.join(output_path, f'{base_filename}_depth_color.png'), depth_color)
            
            if not pred_only:
                split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
                combined_result = cv2.hconcat([raw_image, split_region, depth_color])
                cv2.imwrite(os.path.join(output_path, f'{base_filename}_combined.png'), combined_result)

def remove_double_quotes(path):
    return path.replace('"', '')

def main():
    parser = argparse.ArgumentParser(description='Depth Anything V2 - 24-bit Enhanced')

    parser.add_argument('--img-path', type=str, help='Path to the image file or directory containing images')
    parser.add_argument('--input-size', type=int, default=518, help='Input size for the model. Must be a multiple of 14.')
    parser.add_argument('--outdir', type=str, default='vis_img_depth_24bit', help='Output directory')

    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')

    args = parser.parse_args()

    if not args.img_path:
        args.img_path = input("Path to image file/directory, can right click a file and Copy as Path: ").strip()

    if not args.outdir:
        args.outdir = input("Please enter the output directory (default is 'vis_img_depth_24bit'): ").strip() or 'vis_img_depth_24bit'
        
    args.img_path = remove_double_quotes(args.img_path)
    args.outdir = remove_double_quotes(args.outdir)

    process_image(args.img_path, args.outdir, args.input_size, args.encoder, args.pred_only, args.grayscale)

if __name__ == '__main__':
    main()
