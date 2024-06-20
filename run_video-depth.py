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

from depth_anything_v2.dpt import DepthAnythingV2

warnings.filterwarnings("ignore", message=".*cudnnStatus.*")
os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '0'

def process_video(video_path, output_path, input_size, encoder, pred_only, grayscale):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[encoder])
    state_dict = load_file(f'checkpoints/depth_anything_v2_{encoder}.safetensors')  # Use load_file to load safetensors
    depth_anything.load_state_dict(state_dict)
    depth_anything = depth_anything.to(DEVICE).eval()
    
    if os.path.isfile(video_path) and video_path.endswith('.mp4'):
        filenames = [video_path]
    else:
        filenames = glob.glob(os.path.join(video_path, '**/*.mp4'), recursive=True)
    os.makedirs(output_path, exist_ok=True)
    
    margin_width = 26
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
        raw_video = cv2.VideoCapture(filename)
        frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
        output_width = frame_width * 2 + margin_width

        base_filename = os.path.splitext(os.path.basename(filename))[0]
        combined_output_path = os.path.join(output_path, base_filename + '_combined.mp4')
        depth_output_path = os.path.join(output_path, base_filename + '_depth.mp4')
        grayscale_depth_output_path = os.path.join(output_path, base_filename + '_depth_grayscale.mp4')
        
        combined_out = cv2.VideoWriter(combined_output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (output_width, frame_height))
        depth_out = cv2.VideoWriter(depth_output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (frame_width, frame_height))
        grayscale_depth_out = cv2.VideoWriter(grayscale_depth_output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (frame_width, frame_height))
        
        frame_count = int(raw_video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        for _ in tqdm(range(frame_count), desc="Processing frames", unit="frame"):
            ret, raw_frame = raw_video.read()
            if not ret:
                break
            
            depth = depth_anything.infer_image(raw_frame, input_size)
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.astype(np.uint8)
            
            depth_gray = np.repeat(depth[..., np.newaxis], 3, axis=-1)
            grayscale_depth_out.write(depth_gray)
            
            if not grayscale:
                depth_color = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
                depth_out.write(depth_color)
                
                if not pred_only:
                    split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255
                    combined_frame = cv2.hconcat([raw_frame, split_region, depth_color])
                    combined_out.write(combined_frame)
        
        raw_video.release()
        combined_out.release()
        depth_out.release()
        grayscale_depth_out.release()

def main():
    while True:
        parser = argparse.ArgumentParser(description='Depth Anything V2')

        parser.add_argument('--video-path', type=str, help='Path to the video file or directory containing videos')
        parser.add_argument('--input-size', type=int, default=518)
        parser.add_argument('--outdir', type=str, default='vis_vid_depth', help='Output directory')
        parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
        parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
        parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
        args = parser.parse_args()
        
        if not args.video_path:
            args.video_path = input("Please enter the path to the video file or directory containing videos: ").strip()
        
        if not args.outdir:
            args.outdir = input("Please enter the output directory (default is 'vis_vid_depth'): ").strip() or 'vis_video_depth'
        
        process_video(args.video_path, args.outdir, args.input_size, args.encoder, args.pred_only, args.grayscale)
        
        again = input("Would you like to convert another video? Y/N: ").strip().lower()
        if again not in ['y', 'yes']:
            break

if __name__ == '__main__':
    main()