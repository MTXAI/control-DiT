import argparse
import json

import cv2
from diffusers.models import AutoencoderKL
from torch import Tensor
from torchvision.utils import save_image

from src.diffusion import create_diffusion
from src.models import *
from src.utils.img_tools import *
from src.utils.model import load_model, load_depth_model, load_pretrained_dit_model


def get_img_depths(img_list: [str], midas, transform, latent_size, device) -> [Tensor]:
    z = torch.randn(len(img_list), 4, latent_size, latent_size, device=device)
    for i in range(len(img_list)):
        img_path = img_list[i]
        if img_path == '':
            continue
        ix = cv2_to_pil(cv2.imread(img_path))
        ix = pil_to_cv2(center_crop_arr(ix, args.image_size))
        iz = cv2_to_depth(ix, midas, transform, device)
        iz = iz.unsqueeze(0).to(device)
        iz = depth_to_map(iz, latent_size, 1, False)
        z[i] = iz
    return z


def main(args):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.model_ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = create_control_dit_model_baseline_v2(args.model_type)(
        input_size=latent_size,
        num_classes=args.num_classes,
        in_channels=args.in_channels
    ).to(device)

    if args.model_ckpt != '':
        model_dict = load_model(args.model_ckpt)
        model.load_state_dict(model_dict['ema'])
    elif args.dit_model_ckpt != '':
        state_dict = load_pretrained_dit_model(args.dit_model_ckpt)
        model.load_state_dict(state_dict, strict=False)

    vae = AutoencoderKL.from_pretrained(args.vae_model).to(device)

    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    midas, transform = load_depth_model(model_type="DPT_Large", model_repo_or_path=args.deep_model,
                                        source=args.deep_model_source, device=device)

    # Labels to condition the model with (feel free to change):
    image_config = dict()
    with open(args.image_config, encoding="utf-8") as f:
        image_config = json.load(f)

    class_labels = image_config["class_labels"]
    n = len(class_labels)

    # Create sampling noise:
    x = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    assert len(image_config["origin_images"]) == n, f"length not match: class_labels vs origin_images"
    z = get_img_depths(image_config["origin_images"], midas, transform, latent_size, device)

    # Setup classifier-free guidance:
    x = torch.cat([x, x], dim=0)
    z = torch.cat([z, z], dim=0)
    y_null = torch.tensor([1000] * n, device=device)
    y = torch.cat([y, y_null], 0)
    print(x.shape, z.shape, y.shape)

    model_kwargs = dict(y=y, z=z, cfg_scale=args.cfg_scale)
    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, x.shape, x, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample

    # Save and display images:
    save_image(samples, args.output + '_result.png', nrow=4, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", type=str, default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--in-channels", type=int, default=4)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dit-model-ckpt", type=str, default="/gemini/pretrain/checkpoints/DiT-XL-2-256x256.pt",
                        help="dit model path")
    parser.add_argument("--model-ckpt", type=str, default="")
    parser.add_argument("--deep-model", type=str, default="intel-isl/MiDaS")
    parser.add_argument("--deep-model-source", type=str, default="github")
    parser.add_argument("--vae-model", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--image-config", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    main(args)
