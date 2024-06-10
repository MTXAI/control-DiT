import argparse

import cv2
from diffusers.models import AutoencoderKL
from torchvision.utils import save_image

from src.diffusion import create_diffusion
from src.models import create_dit_model
from src.utils.img_tools import *
from src.utils.model import load_model, load_depth_model


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
    in_channels = 5
    model = create_dit_model(args.model_type)(
        input_size=latent_size,
        num_classes=args.num_classes,
        in_channels=in_channels
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    model_dict = load_model(args.model_ckpt)
    model.load_state_dict(model_dict['ema'])
    model.eval()  # important!

    diffusion = create_diffusion(str(args.num_sampling_steps))
    midas, transform = load_depth_model(model_type="DPT_Large", model_repo_or_path=args.deep_model,
                                        source=args.deep_model_source, device=device)
    vae = AutoencoderKL.from_pretrained(args.vae_model).to(device)

    # Labels to condition the model with (feel free to change):
    class_label = 0

    # Create sampling noise:
    x = torch.randn(1, 4, latent_size, latent_size, device=device)
    y = torch.tensor([class_label], device=device)
    ix = cv2_to_pil(cv2.imread(args.test_image))
    ix = pil_to_cv2(center_crop_arr(ix, args.image_size))
    z = cv2_to_depth(ix, midas, transform, device)
    z = z.unsqueeze(0).to(device)
    z = depth_to_map(z, latent_size, 1, False)
    # Setup classifier-free guidance:
    x = torch.cat([x, z], dim=1)
    x = torch.cat([x, x], 0)
    y_null = torch.tensor([1000], device=device)
    y = torch.cat([y, y_null], 0)

    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
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
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-ckpt", type=str, default=None)
    parser.add_argument("--deep-model", type=str, default="intel-isl/MiDaS")
    parser.add_argument("--deep-model-source", type=str, default="github")
    parser.add_argument("--vae-model", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--test-image", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    main(args)
