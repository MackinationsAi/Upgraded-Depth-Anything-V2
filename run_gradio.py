import glob
import gradio as gr
import matplotlib
import numpy as np
from PIL import Image
import torch
import os
from gradio_imageslider import ImageSlider
from safetensors.torch import load_file
from depth_anything_v2.dpt import DepthAnythingV2
import cv2
from tqdm import tqdm

# Upgraded code by: MackinationsAi
# Original underlying code by: DepthAnything

css = """
#img-display-container {
    max-height: 100vh;
}
#img-display-input {
    max-height: 80vh;
}
#img-display-output {
    max-height: 80vh;
}
#download {
    height: 62px;
}
button[aria-label="Clear"] {
    display: none !important;
}
"""

os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '0'

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}

def get_user_input():
    encoder_list = list(model_configs.keys())
    print("Select the encoder for the Gradio app:")
    for i, key in enumerate(encoder_list):
        print(f"{i+1}. {key}")
    try:
        input_value = input(f"Enter the number corresponding to the encoder (1-{len(encoder_list)}) or press Enter to use default (3): ").strip()
        if input_value == "":
            encoder_index = 2  # Default to 3rd option which is 'vitl'
        else:
            encoder_index = int(input_value) - 1
        if encoder_index not in range(len(encoder_list)):
            raise ValueError
    except ValueError:
        print(f"Invalid selection. Please enter a number between 1 and {len(encoder_list)}.")
        exit(1)
    return encoder_list[encoder_index]

def get_next_filename(base_path, prefix, suffix, extension):
    i = 1
    while os.path.exists(f"{base_path}/{i:04d}{prefix}{suffix}{extension}"):
        i += 1
    return f"{base_path}/{i:04d}{prefix}{suffix}{extension}"

encoder = get_user_input()

checkpoint_path = f'checkpoints/depth_anything_v2_{encoder}.safetensors'
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

model = DepthAnythingV2(**model_configs[encoder])
state_dict = load_file(checkpoint_path)
model.load_state_dict(state_dict)
model = model.to(DEVICE).eval()

def predict_depth(image):
    return model.infer_image(image)

def save_image_with_structure(image, prefix, suffix):
    base_path = 'outputs'
    filename = get_next_filename(base_path, prefix, suffix, '.png')
    image.save(filename)
    return filename

def save_colourized_image_with_structure(image, prefix, suffix, colour_map_method):
    base_path = 'outputs'
    filename = get_next_filename(base_path, prefix, f'_{colour_map_method}{suffix}', '.png')
    image.save(filename)
    return filename

def save_video_with_structure(base_path, base_filename, suffix):
    filename = get_next_filename(base_path, f'_{base_filename}', suffix, '.mp4')
    return filename

def save_video_first_frame_with_structure(image, prefix, suffix, output_path):
    base_path = os.path.join(output_path, 'initial_frames')
    os.makedirs(base_path, exist_ok=True)
    filename = get_next_filename(base_path, prefix, suffix, '.png')
    image.save(filename)
    return filename

def process_image(image, colour_map_method):
    original_image = image.copy()
    h, w = image.shape[:2]
    depth = predict_depth(image[:, :, ::-1])
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    grey_depth = Image.fromarray(depth)
    grey_depth_filename = save_image_with_structure(grey_depth, '', '_greyscale_depth_map')
    
    if colour_map_method == 'All':
        colourized_filenames = []
        for method in colour_map_methods[:-1]:  # Exclude 'All' from the methods list
            coloured_depth = (matplotlib.colormaps.get_cmap(method)(depth)[:, :, :3] * 255).astype(np.uint8)
            coloured_depth_image = Image.fromarray(coloured_depth)
            depth_filename = save_colourized_image_with_structure(coloured_depth_image, '', '_coloured_depth_map', method)
            colourized_filenames.append(depth_filename)
        return colourized_filenames, grey_depth_filename
    else:
        coloured_depth = (matplotlib.colormaps.get_cmap(colour_map_method)(depth)[:, :, :3] * 255).astype(np.uint8)
        coloured_depth_image = Image.fromarray(coloured_depth)
        depth_filename = save_colourized_image_with_structure(coloured_depth_image, '', '_coloured_depth_map', colour_map_method)
        return [depth_filename], grey_depth_filename

def process_video(video_paths, output_path, input_size, encoder, colour_map_method, greyscale):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    depth_anything = DepthAnythingV2(**model_configs[encoder])
    state_dict = load_file(f'checkpoints/depth_anything_v2_{encoder}.safetensors')
    depth_anything.load_state_dict(state_dict)
    depth_anything = depth_anything.to(DEVICE).eval()

    os.makedirs(output_path, exist_ok=True)

    margin_width = 13

    for k, filename in enumerate(video_paths):
        print(f'Progress {k+1}/{len(video_paths)}: {filename}')

        raw_video = cv2.VideoCapture(filename)

        frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))
        output_width = frame_width * 2 + margin_width
        base_filename = os.path.splitext(os.path.basename(filename))[0]
        combined_output_path = save_video_with_structure(output_path, base_filename, '_combined')
        greyscale_depth_output_path = save_video_with_structure(output_path, base_filename, '_depth_greyscale')
        colourized_depth_output_path = save_video_with_structure(output_path, base_filename, '_depth_colourized')
        combined_out = cv2.VideoWriter(combined_output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (output_width, frame_height))
        greyscale_depth_out = cv2.VideoWriter(greyscale_depth_output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (frame_width, frame_height))
        colourized_depth_out = cv2.VideoWriter(colourized_depth_output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (frame_width, frame_height))
        
        frame_count = int(raw_video.get(cv2.CAP_PROP_FRAME_COUNT))
        first_original_frame_path = None
        first_colourized_frame_path = None

        for frame_index in tqdm(range(frame_count), desc="Processing frames", unit="frame"):
            ret, raw_frame = raw_video.read()
            if not ret:
                break

            depth = depth_anything.infer_image(raw_frame, input_size)
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.astype(np.uint8)
            depth_grey = np.repeat(depth[..., np.newaxis], 3, axis=-1)
            greyscale_depth_out.write(depth_grey)
            
            cmap = matplotlib.colormaps.get_cmap(colour_map_method)           
            
            depth_colour = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            colourized_depth_out.write(depth_colour)
            
            split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255
            combined_frame = cv2.hconcat([raw_frame, split_region, depth_colour])
            combined_out.write(combined_frame)

            if frame_index == 0:
                original_frame_rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
                first_original_frame_path = save_video_first_frame_with_structure(Image.fromarray(original_frame_rgb), ''.format(k+1), '_original-frame', output_path)
                first_colourized_frame_path = save_video_first_frame_with_structure(Image.fromarray(depth_colour), ''.format(k+1), '-colourized-depth-frame_{}'.format(colour_map_method), output_path)

        raw_video.release()
        combined_out.release()
        greyscale_depth_out.release()
        colourized_depth_out.release()
        
        if first_original_frame_path is not None and first_colourized_frame_path is not None:
            return [first_original_frame_path, first_colourized_frame_path], combined_output_path
    return [], None

def get_colour_map_methods(selection):
    return full_colour_map_methods if selection == "Full" else colour_map_methods

title = """
# <span style="font-size: 1em; color: #FF5733;">Upgraded Depth Anything V2 🚀 </span> <span style="font-size: 1em;"> </span> <span style="font-size: 0.25em;">Please refer to our [paper](https://arxiv.org/abs/2406.09414), [project page](https://depth-anything-v2.github.io), or [github](https://github.com/MackinationsAi/Upgraded-Depth-Anything-V2) for more details.</span>
"""

with gr.Blocks(css=css) as demo:
    gr.Markdown(title)

    colour_map_selections = ["Default", "Full"]
    
    full_colour_map_methods = [
        'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'GnBu', 'GnBu_r', 'Grays', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 
        'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_grey', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gist_yerg', 
        'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'grey', 'hot', 'hot_r', 'hsv', 'hsv_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r',  'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'winter', 'winter_r', 'All'
    ]
    
    colour_map_methods = [
        'Spectral', 'terrain', 'viridis', 'plasma', 'inferno', 'magma', 'cividis',
        'twilight', 'rainbow', 'gist_rainbow', 'gist_ncar', 'gist_earth', 'turbo',
        'jet', 'afmhot', 'copper', 'seismic', 'hsv', 'brg', 'All'
    ]

    with gr.Tab("Single Image Processing"):
        with gr.Row():
            input_image = gr.Image(label="Input Image", type='numpy', elem_id='img-display-input')
            depth_image_slider = ImageSlider(label="Depth Map with Slider View", elem_id='img-display-output', position=0.5)
        with gr.Row():
            with gr.Column(scale=8):
                colour_map_dropdown_single = gr.Dropdown(label="Select Colour Map Method:", choices=colour_map_methods, value='Spectral')            
            with gr.Column(scale=1):
                colour_map_selection_single = gr.Dropdown(label="Colour Map Method Selection:", choices=colour_map_selections, value='Default')     
        submit_single = gr.Button(value="Compute Depth for Single Image", variant="primary")
        grey_depth_file_single = gr.File(label="greyscale depth map", elem_id="download")

        def on_submit_single(image, colour_map_method):
            original_image = image.copy()
            colourized_filenames, grey_depth_filename = process_image(original_image, colour_map_method)
            first_colourized_image = Image.open(colourized_filenames[0])
            first_colourized_image_np = np.array(first_colourized_image)
            return [(original_image, first_colourized_image_np), grey_depth_filename]
        
        colour_map_selection_single.change(fn=lambda selection: gr.update(choices=get_colour_map_methods(selection)), inputs=colour_map_selection_single, outputs=colour_map_dropdown_single)
        submit_single.click(on_submit_single, inputs=[input_image, colour_map_dropdown_single], outputs=[depth_image_slider, grey_depth_file_single])

    with gr.Tab("Batch Image Processing"):
        with gr.Row():
            input_images = gr.Files(label="Upload Images", type="filepath", elem_id="img-display-input")
        with gr.Row():
            with gr.Column(scale=8):
                colour_map_dropdown_batch = gr.Dropdown(label="Select Colour Map Method:", choices=colour_map_methods, value='Spectral')            
            with gr.Column(scale=1):
                colour_map_selection_batch = gr.Dropdown(label="Colour Map Method Selection:", choices=colour_map_selections, value='Default')      
        submit_batch = gr.Button(value="Compute Depth for Batch", variant="primary")
        output_message = gr.Textbox(label="Output", lines=1, interactive=False)

        def on_batch_submit(files, colour_map_method):
            if not os.path.exists('outputs'):
                os.makedirs('outputs')
            results = []
            for file in files:
                image = np.array(Image.open(file))
                colourized_filenames, grey_filename = process_image(image, colour_map_method)
                results.append(f"Processed {file}: {', '.join(colourized_filenames)}, {grey_filename}")
            return "\n".join(results)
        
        colour_map_selection_batch.change(fn=lambda selection: gr.update(choices=get_colour_map_methods(selection)), inputs=colour_map_selection_batch, outputs=colour_map_dropdown_batch)
        submit_batch.click(on_batch_submit, inputs=[input_images, colour_map_dropdown_batch], outputs=[output_message])

    with gr.Tab("Single Video Processing"):
        with gr.Row():
            input_video_single = gr.Video(label="Input Video", elem_id='img-display-input')
            depth_image_slider_video = ImageSlider(label="Depth Map with Slider View", elem_id='img-display-output', position=0.5)
        with gr.Row():
            with gr.Column(scale=8):
                colour_map_dropdown_video_single = gr.Dropdown(label="Select Colour Map Method:", choices=colour_map_methods, value='Spectral')            
            with gr.Column(scale=1):
                colour_map_selection_video_single = gr.Dropdown(label="Colour Map Method Selection:", choices=colour_map_selections, value='Default')
        with gr.Accordion(open=False, label="Advanced Options:"):    
            with gr.Row():   
                with gr.Column(scale=1):
                    output_dir_single = gr.Textbox(label="Output Directory:", value='outputs/vis_vid_depths')
                with gr.Column(scale=1):
                    input_size_single = gr.Slider(label="Input Size:", minimum=256, maximum=1024, step=1, value=1024)
        greyscale_single = gr.State(value=False)
        submit_video_single = gr.Button(value="Compute Depth for Single Video", variant="primary")
        output_message_video_single = gr.Textbox(label="Outputs", lines=1, interactive=False)

        def on_submit_video_single(video, colour_map_method, output_dir, input_size, greyscale):
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            image_paths, combined_output_path = process_video([video], output_dir, input_size, encoder, colour_map_method, greyscale)
            if image_paths:
                return image_paths, combined_output_path
            return [], None

        colour_map_selection_video_single.change(fn=lambda selection: gr.update(choices=get_colour_map_methods(selection)), inputs=colour_map_selection_video_single, outputs=colour_map_dropdown_video_single)
        submit_video_single.click(on_submit_video_single, inputs=[input_video_single, colour_map_dropdown_video_single, output_dir_single, input_size_single, greyscale_single], outputs=[depth_image_slider_video, output_message_video_single])

    with gr.Tab("Batch Video Processing"):
        with gr.Row():
            input_videos = gr.Files(label="Input Videos", type="filepath", elem_id='img-display-input')
        with gr.Row():
            with gr.Column(scale=5):
                colour_map_dropdown_video_batch = gr.Dropdown(label="Select Colour Map Method:", choices=colour_map_methods, value='Spectral')
            with gr.Column(scale=1):
                colour_map_selection_video_batch = gr.Dropdown(label="Colour Map Method Selection:", choices=colour_map_selections, value='Default')
            with gr.Column(scale=1):
                output_dir = gr.Textbox(label="Output Directory", value='outputs/vis_vid_depths')
        submit_video_batch = gr.Button(value="Compute Depth for Video(s)", variant="primary")
        output_message_video = gr.Textbox(label="Outputs", lines=1, interactive=False)

        def on_submit_video_batch(videos, colour_map_method, output_dir):
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            results = []
            for video in videos:
                image_paths, combined_output_path = process_video([video], output_dir, 1024, encoder, colour_map_method, False)
                result_message = f"Processed {os.path.basename(video)}: "
                if image_paths:
                    result_message += f"Initial frames: {', '.join(image_paths)}, Combined video: {combined_output_path}"
                else:
                    result_message += "No output generated."
                results.append(result_message)
            return "\n".join(results)
        
        colour_map_selection_video_batch.change(fn=lambda selection: gr.update(choices=get_colour_map_methods(selection)), inputs=colour_map_selection_video_batch, outputs=colour_map_dropdown_video_batch)
        submit_video_batch.click(on_submit_video_batch, inputs=[input_videos, colour_map_dropdown_video_batch, output_dir], outputs=output_message_video)

    with gr.Tab("Examples Gallery"):
        gr.Markdown()
        example_files = glob.glob('assets/examples/*')
        gallery = gr.Gallery(value=example_files, label="", columns=3, height="849px")

        def display_selected_image(img_paths):
            img_path = img_paths[0][0] if isinstance(img_paths[0], tuple) else img_paths[0]
            return Image.open(img_path)

        gallery.select(display_selected_image, inputs=[gallery])

if __name__ == '__main__':
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    demo.queue().launch()
