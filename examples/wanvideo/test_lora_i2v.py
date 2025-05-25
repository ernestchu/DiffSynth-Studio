import io
import os

from datasets import load_dataset
import pandas as pd
import torch
import torchvision
from diffsynth import ModelManager, WanVideoPipeline, save_video, VideoData
from modelscope import snapshot_download, dataset_snapshot_download
from PIL import Image

num_samples = 20
model_variant = '1.3b_inp'
use_lora = True
# use_lora = False


# Download models
snapshot_download("PAI/Wan2.1-Fun-1.3B-InP", local_dir="models/PAI/Wan2.1-Fun-1.3B-InP")

# Load models
if model_variant == '1.3b_inp':
    model_manager = ModelManager(device="cpu")
    model_manager.load_models(
        [
            "models/PAI/Wan2.1-Fun-1.3B-InP/diffusion_pytorch_model.safetensors",
            "models/PAI/Wan2.1-Fun-1.3B-InP/models_t5_umt5-xxl-enc-bf16.pth",
            "models/PAI/Wan2.1-Fun-1.3B-InP/Wan2.1_VAE.pth",
            "models/PAI/Wan2.1-Fun-1.3B-InP/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        ],
        torch_dtype=torch.bfloat16, # You can set `torch_dtype=torch.float8_e4m3fn` to enable FP8 quantization.
    )
    if use_lora:
        model_manager.load_lora("models/lightning_logs/i2v_1.3b/checkpoints/epoch=9-step=1670.ckpt", lora_alpha=1.0)
else:
    raise ValueError(f"Unsupported model variant: {model_variant}")
pipe = WanVideoPipeline.from_model_manager(model_manager, torch_dtype=torch.bfloat16, device="cuda")
pipe.enable_vram_management(num_persistent_param_in_dit=None)

ds_url = "https://huggingface.co/datasets/SwayStar123/CelebV-HQ/resolve/main/videos.tar"
for i, data in enumerate(load_dataset("webdataset", data_files={"train": ds_url}, split="train", streaming=True)):
    if i == num_samples:
        break
    video = torchvision.io.read_video(io.BytesIO(data['mp4']), pts_unit='sec')[0]
    resize_and_crop = torchvision.transforms.Compose([
        torchvision.transforms.Resize(384, interpolation=torchvision.transforms.InterpolationMode.BILINEAR),
        torchvision.transforms.CenterCrop(384),
    ])
    image = resize_and_crop(video[0].permute(2, 0, 1) / 255)
    image = torchvision.transforms.functional.to_pil_image(image)
    video = pipe(
        prompt="A person is talking",
        negative_prompt="bad quality, still image",
        width=384,
        height=384,
        num_inference_steps=50,
        input_image=image,
        # You can input `end_image=xxx` to control the last frame of the video.
        # The model will automatically generate the dynamic content between `input_image` and `end_image`.
        seed=i, tiled=True
    )
    video[0] = image # Ensure the first frame aligns with the input image
    dirname = f"test_out_{model_variant}{'_ft' if use_lora else ''}"
    os.makedirs(dirname, exist_ok=True)
    save_video(video, os.path.join(dirname, f"video_{i:02}.mp4"), fps=15)

