<div align="center">
<h1>Upgraded Depth Anything V2</h1>
</div>

This work presents Depth Anything V2. It significantly outperforms [V1](https://github.com/LiheYoung/Depth-Anything) in fine-grained details & robustness. Compared with SD-based models, it enjoys faster inference speed, fewer parameters, higher depth accuracy, & a robust upgraded Gradio WebUI as well as both image & video .bat scripts for more intuitive CLI usage (if that is your more preferred method of use).

### UDAV2 Outputs

![DepthV2_Outputs](https://github.com/MackinationsAi/Upgraded-Depth-Anything-V2/assets/133395980/46cdb302-3b34-4226-8920-372dfb4a0adc)

### Gradio Example

![Single_Image_Processing](https://github.com/MackinationsAi/Upgraded-Depth-Anything-V2/assets/133395980/ba7f4653-bc58-465c-8701-bb1d2ec27651)

## News

- **2024-06-14:** Paper, project page, code, models, demo, & benchmark are all released.
- **2024-06-20:** The repo has been upgraded & is also now running on .safetensors models instead of .pth models.
- **2024-06-23:** Updated installation process to be a simpler one_click_install.bat file. It automatically downloads the depth models into a 'checkpoints' folder, the triton wheel into the repo's main folder & installs all of the dependencies needed. *[Also updated this README.md file to provide more clarity!]*

## Windows Installation

All you need to do is copy & paste (or right-click), each of the following lines in-order into cmd & everything will be installed properly.

```
git clone https://github.com/MackinationsAi/Upgraded-Depth-Anything-V2.git
cd Upgraded-Depth-Anything-V2
one_click_install.bat
```
That's it! All you have to do now is pick one of the run_-------.bat files, double-click & you're off to depthing!

## Usage

### Gradio WebUi

To use the upgraded gradio webui locally:

```bash
run_gradio.bat
```
You can also try the [online gradio demo](https://huggingface.co/spaces/Depth-Anything/Depth-Anything-V2), though it is FAR less capable than this Upgraded Depth Anything V2 repo.

### Running run_image-depth.py CLI script

***It works for both single image depth processing & batch image depth processing.***

```bash
run_image-depth.bat
```
or

```bash
python run_image-depth.py --encoder <vits | vitb | vitl> --img-path <path> --outdir <outdir> [--input-size <size>] [--pred-only] [--grayscale]
```

Options:
- `--img-path`: You can either 1.) point it to an image directory storing all interested images, 2.) point it to a single image, or 3.) point it to a text file storing all image paths.
- `--input-size` (optional): By default, we use input size `518` for model inference. **You can increase the size for even more fine-grained results.**
- `--pred-only` (optional): Only save the predicted depth map, without raw image.
- `--grayscale` (optional): Save the grayscale depth map, without applying color palette.

For example:
```bash
python run_image-depth.py --encoder vitl --img-path assets/examples --outdir depth_vis
```

### Running run_video-depth.py CLI script

***It works for both single video depth processing & batch video depth processing.***

```bash
run_video-depth.bat
```
or

```bash
python run_video-depth.py --encoder vitl --video-path assets/examples_video --outdir video_depth_vis
```

## Pre-trained Models *[.safetensors]*

We provide **three models** of varying scales for robust relative depth estimation (the fourth model is still a WIP):

***All three models are automatically downloaded to a 'checkpoints' folder in your repo when you run the one_click_install.bat. (I only provided the download link here incase you want to download them elsewhere for use outside this repo)***


| Models | Params | Checkpoints |
|:-|-:|:-:|
| Depth-Anything-V2-Small model | 48.4M | [Download](https://huggingface.co/MackinationsAi/Depth-Anything-V2_Safetensors/resolve/main/depth_anything_v2_vits.safetensors?download=true) |
| Depth-Anything-V2-Base model | 190.4M | [Download](https://huggingface.co/MackinationsAi/Depth-Anything-V2_Safetensors/resolve/main/depth_anything_v2_vitb.safetensors?download=true) |
| Depth-Anything-V2-Large model | 654.9M | [Download](https://huggingface.co/MackinationsAi/Depth-Anything-V2_Safetensors/resolve/main/depth_anything_v2_vitl.safetensors?download=true) |
| Depth-Anything-V2-Giant model | 1.3B | *Coming soon* | [Download Doesn't Work - Model is still a WIP](https://huggingface.co/MackinationsAi/Depth-Anything-V2_Safetensors/resolve/main/depth_anything_v2_vitg.safetensors?download=true) |


*Please note that the larger (vitl) model has better temporal consistency on videos.*

## Triton Dependency Wheel

***This dependency .whl is automatically downloaded to the main/tree repo-folder when you run the one_click_install.bat. (I only provided the download link here incase you want to download it elsewhere for use outside this repo.)***


| Dependency | Params | Wheel |
|:-|-:|:-:|
| Triton==2.1.0 | 306.7M | [Download](https://huggingface.co/MonsterMMORPG/SECourses/blob/main/triton-2.1.0-cp310-cp310-win_amd64.whl?download=true) |


*(Once it has been installed & the gradio webui is running properly, you can delete it or use it elsewhere in a similar fashion.)*

### *Notes:*
- Compared to V1, we have made a minor modification to the DINOv2-DPT architecture (originating from this [issue](https://github.com/LiheYoung/Depth-Anything/issues/81)). In V1, we *unintentionally* used features from the last four layers of DINOv2 for decoding. In V2, we use [intermediate features](https://github.com/DepthAnything/Depth-Anything-V2/blob/2cbc36a8ce2cec41d38ee51153f112e87c8e42d8/depth_anything_v2/dpt.py#L164-L169) instead. Although this modification did not improve details or accuracy, we decided to follow this common practice. 
- **I will be updating the training scripts to support .safetensors output pre-trained models in the coming weeks so stay-tuned for more UDAV2 depthing updates!**

## Original DAV2 Github Repo Creds
<div align="center">

[**Lihe Yang**](https://liheyoung.github.io/)<sup>1</sup> · [**Bingyi Kang**](https://bingykang.github.io/)<sup>2&dagger;</sup> · [**Zilong Huang**](http://speedinghzl.github.io/)<sup>2</sup> · [**Zhen Zhao**](http://zhaozhen.me/) · [**Xiaogang Xu**](https://xiaogang00.github.io/) · [**Jiashi Feng**](https://sites.google.com/site/jshfeng/)<sup>2</sup> · [**Hengshuang Zhao**](https://hszhao.github.io/)<sup>1*</sup>

Legend <sup>Keys</sup> - [ HKU <sup>1</sup>  ·  TikTok <sup>2</sup>  ·  project-lead &dagger;  ·  corresponding author * ]
</div>

<div align="center">
<a href="https://arxiv.org/abs/2406.09414"><img src='https://img.shields.io/badge/arXiv-Depth Anything V2-red' alt='Paper PDF'></a>
<a href='https://depth-anything-v2.github.io'><img src='https://img.shields.io/badge/Project_Page-Depth Anything V2-green' alt='Project Page'></a>
<a href='https://huggingface.co/spaces/depth-anything/Depth-Anything-V2'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue'></a>
<a href='https://huggingface.co/datasets/depth-anything/DA-2K'><img src='https://img.shields.io/badge/Benchmark-DA--2K-yellow' alt='Benchmark'></a>
</div>

![teaser](assets/teaser.png)

## Fine-tuned to Metric Depth Estimation & DA-2K Evaluation Benchmark

Please refer to [metric depth estimation](./metric_depth) &/or to [DA-2K benchmark](./DA-2K.md).

## LICENSE

Depth-Anything-V2-Small model is under the Apache-2.0 license. Depth-Anything-V2-Base/Large/Giant models are under the CC-BY-NC-4.0 license.

## Citation

If you find this project useful, please consider citing below, give this upgraded repo a star & share it w/ others in the community!

```bibtex
@article{depth_anything_v2,
  title={Depth Anything V2},
  author={Yang, Lihe & Kang, Bingyi & Huang, Zilong & Zhao, Zhen & Xu, Xiaogang & Feng, Jiashi & Zhao, Hengshuang},
  journal={arXiv:2406.09414},
  year={2024}
}

@inproceedings{depth_anything_v1,
  title={Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data}, 
  author={Yang, Lihe & Kang, Bingyi & Huang, Zilong & Xu, Xiaogang & Feng, Jiashi & Zhao, Hengshuang},
  booktitle={CVPR},
  year={2024}
}
```
