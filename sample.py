# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import os

import torch

from utils.image_tools import ImageTools

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from models.DiT import DiT_models
from models.control_DiT import ControlDiT_models
import argparse


imgTools = ImageTools()


def load_dit_model(model_ckpt):
    assert os.path.isfile(model_ckpt), f'Could not find DiT checkpoint at {model_ckpt}'
    checkpoint = torch.load(model_ckpt, map_location=lambda storage, loc: storage)
    if "ema" in checkpoint:  # supports checkpoints from train.py
        checkpoint = checkpoint["ema"]
    return checkpoint


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    if args.dit_ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    dit_model = DiT_models[args.model_type](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    dit_ckpt_path = args.dit_ckpt
    dit_state_dict = load_dit_model(dit_ckpt_path)
    dit_model.load_state_dict(dit_state_dict)
    dit_model.eval()  # important!

    control_dit_model = ControlDiT_models[args.model_type](
        dit=dit_model,
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    control_dit_ckpt_path = args.control_dit_ckpt
    control_dit_state_dict = load_dit_model(control_dit_ckpt_path)
    control_dit_model.load_state_dict(control_dit_state_dict)
    control_dit_model.eval()  # important!

    diffusion = create_diffusion(str(args.num_sampling_steps))

    midas, transform = imgTools.load_deep_model(model_type="DPT_Large", cuda=False,
                                                model_repo_or_path=args.deep_model, source=args.deep_model_source)
    vae = AutoencoderKL.from_pretrained(args.vae_model).to(device)

    # Labels to condition the model with (feel free to change):
    class_labels = [0]

    # Create sampling noise:
    n = len(class_labels)
    x = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    # Setup classifier-free guidance:
    x = torch.cat([x, x], 0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)

    ix = imgTools.read(args.test_image)
    z = imgTools.to_deep(ix, False, midas, transform)
    z = z.expand_as(x[0])
    z = z.expand_as(x)
    z = z.to(device)
    with torch.no_grad():
        # Map input images to latent space + normalize latents:
        z = vae.encode(z).latent_dist.sample().mul_(0.18215)
        z = z.detach().cpu().numpy()
    model_kwargs = dict(y=y, z=z, cfg_scale=args.cfg_scale)

    # Sample images:
    samples = diffusion.p_sample_loop(
        control_dit_model.forward_with_cfg, x.shape, x, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample

    # Save and display images:
    save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dit-ckpt", type=str, default=None)
    parser.add_argument("--control-dit-ckpt", type=str, default=None)
    parser.add_argument("--deep-model", type=str, default="intel-isl/MiDaS")
    parser.add_argument("--vae-model", type=str, default="stabilityai/sd-vae-ft-ema")
    parser.add_argument("--test-image", type=str, default=None)
    args = parser.parse_args()
    main(args)
