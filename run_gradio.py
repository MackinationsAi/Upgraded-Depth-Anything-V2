import glob
import gradio as gr
import matplotlib
import numpy as np
from PIL import Image
import torch
from gradio_imageslider import ImageSlider
import os
from safetensors.torch import load_file
from depth_anything_v2.dpt import DepthAnythingV2

# Code upgraded by: MackinationsAi
# Original code by: DepthAnything

os.environ['TORCH_CUDNN_V8_API_ENABLED'] = '0'

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
"""

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
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
    while os.path.exists(f"{base_path}/{prefix}{i:04d}{suffix}{extension}"):
        i += 1
    return f"{base_path}/{prefix}{i:04d}{suffix}{extension}"

encoder = get_user_input()

checkpoint_path = f'checkpoints/depth_anything_v2_{encoder}.safetensors'
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

model = DepthAnythingV2(**model_configs[encoder])
state_dict = load_file(checkpoint_path)
model.load_state_dict(state_dict)
model = model.to(DEVICE).eval()

title = "# Upgraded Depth Anything V2"
description = """Please refer to our [paper](https://arxiv.org/abs/2406.09414), [project page](https://depth-anything-v2.github.io), or [github](https://github.com/MackinationsAi/Upgraded-Depth-Anything-V2) for more details."""

def predict_depth(image):
    return model.infer_image(image)

def save_image_with_structure(image, prefix, suffix):
    base_path = 'outputs'
    filename = get_next_filename(base_path, prefix, suffix, '.png')
    image.save(filename)
    return filename

def process_image(image):
    original_image = image.copy()
    h, w = image.shape[:2]
    depth = predict_depth(image[:, :, ::-1])
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    colored_depth = (matplotlib.colormaps.get_cmap('Spectral_r')(depth)[:, :, :3] * 255).astype(np.uint8)
    colored_depth_image = Image.fromarray(colored_depth)
    depth_filename = save_image_with_structure(colored_depth_image, '', '_coloured_depth_map')
    gray_depth = Image.fromarray(depth)
    gray_depth_filename = save_image_with_structure(gray_depth, '', '_greyscale_depth_map')
    return depth_filename, gray_depth_filename

def on_submit(image):
    if isinstance(image, list):
        results = [process_image(img) for img in image]
        return results
    else:
        return process_image(image)

with gr.Blocks(css=css) as demo:
    gr.Markdown(title)
    gr.Markdown(description)

    with gr.Tab("Single Image"):
        with gr.Row():
            input_image = gr.Image(label="Input Image", type='numpy', elem_id='img-display-input')
            depth_image_slider = ImageSlider(label="Depth Map with Slider View", elem_id='img-display-output', position=0.5)
        submit_single = gr.Button(value="Compute Depth for Single Image")
        gray_depth_file_single = gr.File(label="Grayscale depth map", elem_id="download")
        cmap = matplotlib.colormaps.get_cmap('Spectral_r')

        def on_submit_single(image):
            original_image = image.copy()
            h, w = image.shape[:2]
            depth = predict_depth(image[:, :, ::-1])
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.astype(np.uint8)
            colored_depth = (cmap(depth)[:, :, :3] * 255).astype(np.uint8)
            colored_depth_image = Image.fromarray(colored_depth)
            depth_filename = save_image_with_structure(colored_depth_image, '', '_coloured_depth_map')
            gray_depth = Image.fromarray(depth)
            gray_depth_filename = save_image_with_structure(gray_depth, '', '_greyscale_depth_map')
            return [(original_image, colored_depth), gray_depth_filename]
        submit_single.click(on_submit_single, inputs=[input_image], outputs=[depth_image_slider, gray_depth_file_single])

    with gr.Tab("Batch Processing"):
        with gr.Row():
            input_images = gr.Files(label="Upload Images", type="filepath", elem_id="img-display-input")
        submit_batch = gr.Button(value="Compute Depth for Batch")
        output_message = gr.Textbox(label="Output", lines=5, interactive=False)

        def on_batch_submit(files):
            if not os.path.exists('outputs'):
                os.makedirs('outputs')
            results = []
            for file in files:
                image = np.array(Image.open(file))
                colored_filename, gray_filename = process_image(image)
                results.append(f"Processed {file}: {colored_filename}, {gray_filename}")
            return "\n".join(results)
        submit_batch.click(on_batch_submit, inputs=[input_images], outputs=[output_message])

    with gr.Tab("Examples Gallery"):
        input_examples = gr.Image(label="Input Examples", type='numpy', elem_id='img-display-input')
        example_files = glob.glob('assets/examples/*')
        examples = gr.Examples(examples=example_files, inputs=[input_examples], outputs=[depth_image_slider])

if __name__ == '__main__':
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    demo.queue().launch()