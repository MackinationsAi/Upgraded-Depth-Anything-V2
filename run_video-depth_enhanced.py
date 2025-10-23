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

# Code upgraded by: MackinationsAi
# Enhanced for high bit depth and seamless video processing

from depth_anything_v2.dpt import DepthAnythingV2

warnings.filterwarnings("ignore", message=".*cudnnStatus.*")
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '0'

if torch.cuda.is_available():
    import torch.backends.cudnn as cudnn

    os.environ['TORCH_CUDNN_V8_API_DISABLED'] = '1'

    cudnn.benchmark = False
    cudnn.deterministic = True

def apply_temporal_smoothing(current_depth, previous_depth, smoothing_factor=0.3):
    """Apply temporal smoothing to reduce flickering between frames"""
    if previous_depth is None:
        return current_depth
    return (1 - smoothing_factor) * current_depth + smoothing_factor * previous_depth

def apply_seamless_processing(depth_map, method='edge_blend', blend_width=32):
    """Apply seamless processing for tiling capabilities"""
    h, w = depth_map.shape
    processed_depth = depth_map.copy()
    
    if method == 'edge_blend':
        for i in range(blend_width):
            alpha = i / blend_width
            processed_depth[i, :] = alpha * depth_map[i, :] + (1 - alpha) * depth_map[-(blend_width-i), :]
            processed_depth[-(i+1), :] = alpha * depth_map[-(i+1), :] + (1 - alpha) * depth_map[blend_width-i-1, :]
        
        for j in range(blend_width):
            alpha = j / blend_width
            processed_depth[:, j] = alpha * depth_map[:, j] + (1 - alpha) * depth_map[:, -(blend_width-j)]
            processed_depth[:, -(j+1)] = alpha * depth_map[:, -(j+1)] + (1 - alpha) * depth_map[:, blend_width-j-1]
    
    return processed_depth

def process_video(video_path, output_path, input_size, encoder, pred_only, grayscale, bit_depth=8, seamless_mode=False, temporal_smoothing=False):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[encoder])
    state_dict = load_file(f'checkpoints/depth_anything_v2_{encoder}.safetensors')
    depth_anything.load_state_dict(state_dict)
    depth_anything = depth_anything.to(DEVICE).eval()
    
    if os.path.isfile(video_path) and video_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        filenames = [video_path]
    else:
        video_path = os.path.normpath(video_path)
        glob_pattern = os.path.join(video_path, '**', '*.*')
        filenames = glob.glob(glob_pattern, recursive=True)
        filenames = [f for f in filenames if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
    
    os.makedirs(output_path, exist_ok=True)
    
    margin_width = 13
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    # Set up bit depth parameters
    if bit_depth == 8:
        max_val = 255.0
        dtype = np.uint8
        suffix = "8bit"
    elif bit_depth == 16:
        max_val = 65535.0
        dtype = np.uint16
        suffix = "16bit"
    elif bit_depth == 24:
        max_val = 16777215.0
        dtype = np.uint32
        suffix = "24bit"
    elif bit_depth == 32:
        max_val = 4294967295.0
        dtype = np.uint32
        suffix = "32bit"
    else:
        raise ValueError(f"Unsupported bit depth: {bit_depth}")
    
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
        raw_video = cv2.VideoCapture(filename)
        
        frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
        output_width = frame_width * 2 + margin_width
        
        base_filename = os.path.splitext(os.path.basename(filename))[0]
        
        # Set up output video writers
        combined_output_path = os.path.join(output_path, f'{base_filename}_combined_{suffix}.mp4')
        greyscale_depth_output_path = os.path.join(output_path, f'{base_filename}_depth_greyscale_{suffix}.mp4')
        colourized_depth_output_path = os.path.join(output_path, f'{base_filename}_depth_colourized_{suffix}.mp4')
        
        # For higher bit depths, we'll save as separate image sequences and then combine to video
        if bit_depth > 16:
            frame_dir = os.path.join(output_path, f'{base_filename}_frames_{suffix}')
            os.makedirs(frame_dir, exist_ok=True)
        
        combined_out = cv2.VideoWriter(combined_output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (output_width, frame_height))
        greyscale_depth_out = cv2.VideoWriter(greyscale_depth_output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (frame_width, frame_height))
        colourized_depth_out = cv2.VideoWriter(colourized_depth_output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (frame_width, frame_height))
        
        frame_count = int(raw_video.get(cv2.CAP_PROP_FRAME_COUNT))
        previous_depth = None
        first_original_frame_path = None
        first_colourized_frame_path = None
        
        for frame_index in tqdm(range(frame_count), desc="Processing frames", unit="frame"):
            ret, raw_frame = raw_video.read()
            if not ret:
                break
            
            depth = depth_anything.infer_image(raw_frame, input_size)
            
            # Apply seamless processing if enabled
            if seamless_mode:
                depth = apply_seamless_processing(depth, method='edge_blend')
            
            # Apply temporal smoothing if enabled
            if temporal_smoothing:
                depth = apply_temporal_smoothing(depth, previous_depth)
                previous_depth = depth
            
            # Normalize depth
            depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())
            
            # Process for different bit depths
            if bit_depth <= 16:
                # Standard processing for 8-bit and 16-bit
                depth_scaled = (depth_normalized * max_val).astype(dtype)
                
                if bit_depth == 8:
                    depth_grey = np.repeat(depth_scaled[..., np.newaxis], 3, axis=-1)
                else:  # 16-bit
                    # For 16-bit video output, we need to convert to 8-bit for video writer
                    depth_8bit = (depth_normalized * 255.0).astype(np.uint8)
                    depth_grey = np.repeat(depth_8bit[..., np.newaxis], 3, axis=-1)
                
                greyscale_depth_out.write(depth_grey)
                
                # Colorized depth for video
                depth_colour = (cmap(depth_normalized)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
                colourized_depth_out.write(depth_colour)
                
                # Combined frame
                split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255
                combined_frame = cv2.hconcat([raw_frame, split_region, depth_colour])
                combined_out.write(combined_frame)
                
            else:
                # For 24-bit and 32-bit, save individual frames as TIFF and create 8-bit video
                depth_high_bit = (depth_normalized * max_val).astype(dtype)
                
                # Save high bit depth frame
                frame_filename = os.path.join(frame_dir, f'frame_{frame_index:06d}_depth_{suffix}.tiff')
                cv2.imwrite(frame_filename, depth_high_bit)
                
                # Create 8-bit version for video
                depth_8bit = (depth_normalized * 255.0).astype(np.uint8)
                depth_grey = np.repeat(depth_8bit[..., np.newaxis], 3, axis=-1)
                greyscale_depth_out.write(depth_grey)
                
                depth_colour = (cmap(depth_normalized)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
                colourized_depth_out.write(depth_colour)
                
                split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255
                combined_frame = cv2.hconcat([raw_frame, split_region, depth_colour])
                combined_out.write(combined_frame)
            
            # Save first frame for preview
            if frame_index == 0:
                original_frame_rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
                first_original_frame_path = os.path.join(output_path, f'{base_filename}_first_frame_original.png')
                cv2.imwrite(first_original_frame_path, raw_frame)
                
                first_colourized_frame_path = os.path.join(output_path, f'{base_filename}_first_frame_depth_{suffix}.png')
                cv2.imwrite(first_colourized_frame_path, depth_colour)
        
        raw_video.release()
        combined_out.release()
        greyscale_depth_out.release()
        colourized_depth_out.release()
        
        print(f"Video processing complete for {base_filename}")
        if bit_depth > 16:
            print(f"High bit depth frames saved in: {frame_dir}")

def remove_double_quotes(path):
    return path.replace('"', '')

def main():
    while True:
        parser = argparse.ArgumentParser(description='Depth Anything V2 - Enhanced Video Processing')
        
        parser.add_argument('--video-path', type=str, help='Path to the video file or directory containing videos')
        parser.add_argument('--input-size', type=int, default=518)
        parser.add_argument('--outdir', type=str, default='video_depth_vis_enhanced', help='Output directory')
        
        parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])     
        parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
        parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
        parser.add_argument('--bit-depth', type=int, default=8, choices=[8, 16, 24, 32], help='Output bit depth')
        parser.add_argument('--seamless', dest='seamless', action='store_true', help='apply seamless processing for tiling')
        parser.add_argument('--temporal-smoothing', dest='temporal_smoothing', action='store_true', help='apply temporal smoothing to reduce flickering')
        
        args = parser.parse_args()
        
        if not args.video_path:
            args.video_path = input("Path to video file/directory, can right click a file and Copy as Path: ").strip()
        
        if not args.outdir:
            args.outdir = input("Please enter the output directory (default is 'video_depth_vis_enhanced'): ").strip() or 'video_depth_vis_enhanced'
            
        args.video_path = remove_double_quotes(args.video_path)
        args.outdir = remove_double_quotes(args.outdir)
        
        process_video(args.video_path, args.outdir, args.input_size, args.encoder, args.pred_only, args.grayscale, 
                     args.bit_depth, args.seamless, args.temporal_smoothing)
        
        again = input("Convert another video? Y/N: ").strip().lower()
        if again not in ['y', 'yes']:
            break

if __name__ == '__main__':
    main()