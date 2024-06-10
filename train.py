import argparse
from copy import deepcopy
from glob import glob
from time import time

from torch.utils.data import DataLoader

from src.diffusion import create_diffusion
from src.models import create_dit_model
from src.utils.dataset import CustomDataset
from src.utils.log import init_logger
from src.utils.model import *


def prepare_output_dir(output, model_type):
    os.makedirs(output, exist_ok=True)  # Make results folder (holds all experiment subfolders)
    experiment_index = len(glob(f"{output}/*"))
    model_string_name = model_type.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
    experiment_dir = f"{output}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    return experiment_dir, checkpoint_dir


def main(args):
    """
    Trains a new DiT model.
    """
    assert cuda, "Training currently requires at least one GPU."

    # 1. Setup accelerator:
    accelerator, device = setup_accelerator(not cuda)

    model_ckpt = ''
    if args.model_ckpt != '' and args.model_ckpt != none_model:
        model_ckpt = args.model_ckpt
    batch_size = int(args.batch_size // accelerator.num_processes)
    in_channels = 5
    # 2. Setup data:
    features_dir = os.path.join(args.input, f'imagenet{args.image_size}_features')
    labels_dir = os.path.join(args.input, f'imagenet{args.image_size}_labels')
    conditions_dir = os.path.join(args.input, f'imagenet{args.image_size}_conditions')
    dataset = CustomDataset(features_dir, labels_dir, conditions_dir)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # 3. Prepare output dir and log train args
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    if accelerator.is_main_process:
        # Setup an experiment folder and init logger
        experiment_dir, checkpoint_dir = prepare_output_dir(args.output, args.model_type)
        logger = init_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        logger.info(f"Input information: " +
                    f"\n\t - experiment_dir: {experiment_dir}" +
                    f"\n\t - checkpoint_dir: {checkpoint_dir}" +
                    f"\n\t - features_dir: {features_dir}" +
                    f"\n\t - labels_dir: {labels_dir}" +
                    f"\n\t - conditions_dir: {conditions_dir}" +
                    f"\n\t - dit_model_path: {args.dit_model_path}" +
                    f"\n\t - model_ckpt: {model_ckpt}")
        logger.info(f"Train options: " +
                    f"\n\t - model_type: {args.model_type}" +
                    f"\n\t - epochs: {args.epochs}" +
                    f"\n\t - lr: {args.lr}" +
                    f"\n\t - decay: {args.decay}" +
                    f"\n\t - batch_size: {batch_size}" +
                    f"\n\t - latent_size: {latent_size}" +
                    f"\n\t - in_channels: {in_channels}" +
                    f"\n\t - num_workers: {args.num_workers}" +
                    f"\n\t - ckpt_every: {args.ckpt_every}" +
                    f"\n\t - only_ema: {args.only_ema}" +
                    f"\n\t - log_every: {args.log_every}")
        logger.info(f"Dataset abstract: " +
                    f"\n\t - length: {len(dataset)}" +
                    f"\n\t - image_size: {args.image_size}" +
                    f"\n\t - num_classes: {args.num_classes}")

    # 4. Load dit model from checkpoint or pretrained checkpoint
    model = create_dit_model(args.model_type)(
        input_size=latent_size,
        num_classes=args.num_classes,
        in_channels=in_channels,
    ).to(device)

    if model_ckpt != '':
        model_dict = try_load_model(model_ckpt)
        model.load_state_dict(model_dict['model'])
    elif args.dit_model_path != '':
        state_dict = load_dit_model(args.dit_model_path)
        model.load_state_dict(state_dict, strict=False)

    # 5. Create diffusion pipeline and Setup optimizer
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    if accelerator.is_main_process:
        logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.decay)
    if model_ckpt != '':
        opt.load_state_dict(model_dict['opt'])

    # 6. Prepare models for training
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    model, opt, loader = accelerator.prepare(model, opt, loader)

    # variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    # 7. Begin training
    first_epoch = 0
    if model_ckpt != '':
        first_epoch = model_dict['epoch'] + 1
    if accelerator.is_main_process:
        logger.info(f"Training for {args.epochs} epochs, First epoch is {first_epoch}...")
    for epoch in range(first_epoch, args.epochs):
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")
        for x, y, z in loader:
            x = x.to(device)
            y = y.to(device)
            z = z.to(device)

            x = x.squeeze(dim=0)
            y = y.squeeze(dim=0).long()
            y = y.squeeze(dim=0)
            z = z.squeeze(dim=0)
            # cat x and z, [4, 32, 32] -> [5, 32, 32]
            x = torch.cat([x, z], dim=1)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()
            update_ema(ema, model)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                avg_loss = avg_loss.item() / accelerator.num_processes
                if accelerator.is_main_process:
                    logger.info(
                        f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "epoch": epoch,
                        "ema": ema.state_dict(),
                    }
                    if not args.only_ema:
                        checkpoint = {
                            "epoch": epoch,
                            "model": model.state_dict(),
                            "ema": ema.state_dict(),
                            "opt": opt.state_dict(),
                            "args": args
                        }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

    # final save
    if accelerator.is_main_process:
        if train_steps % args.ckpt_every != 0 and train_steps > 0:
            checkpoint = {
                "ema": ema.state_dict(),
            }
            if not args.only_ema:
                checkpoint = {
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args
                }
            checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
    model.eval()  # important! This disables randomized embedding dropout

    if accelerator.is_main_process:
        logger.info("Done!")


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="/gemini/data-1/imagenette_processed_train", help="input")
    parser.add_argument("--output", type=str, default="/gemini/output/control-dit_train_baseline-v4", help="output")
    parser.add_argument("--model-type", type=str, default="DiT-XL/2")
    parser.add_argument("--dit-model-path", type=str, default="/gemini/pretrain/checkpoints/DiT-XL-2-256x256.pt",
                        help="dit model path")
    parser.add_argument("--model-ckpt", type=str, default="")

    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)

    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--decay", type=float, default=1e-1)
    parser.add_argument("--batch-size", type=int, default=256, help="batch size should >= 6*2")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--ckpt-every", type=int, default=10_000)
    parser.add_argument("--only-ema", action='store_true')
    parser.add_argument("--log-every", type=int, default=100)

    args = parser.parse_args()
    main(args)
