# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT.
"""
import torch

from models.DiT import DiT_models
from models.control_DiT import ControlDiT_models, operator_add
from utils.dataset import CustomDataset

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from accelerate import Accelerator

from diffusion import create_diffusion
from diffusers.models import AutoencoderKL

logger = logging.Logger('')


#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def init_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    return logging.getLogger(__name__)


def prepare(output, model_type):
    os.makedirs(output, exist_ok=True)  # Make results folder (holds all experiment subfolders)
    experiment_index = len(glob(f"{output}/*"))
    model_string_name = model_type.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
    experiment_dir = f"{output}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    return experiment_dir, checkpoint_dir


def load_dit_model(model_ckpt):
    assert os.path.isfile(model_ckpt), f'Could not find DiT checkpoint at {model_ckpt}'
    checkpoint = torch.load(model_ckpt, map_location=lambda storage, loc: storage)
    if "ema" in checkpoint:  # supports checkpoints from train.py
        checkpoint = checkpoint["ema"]
    return checkpoint


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup accelerator:
    accelerator = Accelerator()
    device = accelerator.device

    # Setup data:
    model_type = args.model_type
    dit_model_path = args.dit_model_path
    image_size = args.image_size
    num_classes = args.num_classes
    epochs = args.epochs
    batch_size = int(args.batch_size // accelerator.num_processes)
    num_workers = args.num_workers
    ckpt_every = args.ckpt_every
    log_every = args.log_every

    features_dir = os.path.join(args.input, f'imagenet{image_size}_features')
    labels_dir = os.path.join(args.input, f'imagenet{image_size}_labels')
    conditions_dir = os.path.join(args.input, f'imagenet{image_size}_conditions')
    dataset = CustomDataset(features_dir, labels_dir, conditions_dir)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    if accelerator.is_main_process:
        # Setup an experiment folder and init logger
        experiment_dir, checkpoint_dir = prepare(args.output, model_type)
        logger = init_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
        logger.info(f"Input information: " +
                    f"\n\t - experiment_dir: {experiment_dir}" +
                    f"\n\t - checkpoint_dir: {checkpoint_dir}" +
                    f"\n\t - features_dir: {features_dir}" +
                    f"\n\t - labels_dir: {labels_dir}" +
                    f"\n\t - conditions_dir: {conditions_dir}" +
                    f"\n\t - dit_model_path: {dit_model_path}")
        logger.info(f"Train options: " +
                    f"\n\t - model_type: {model_type}" +
                    f"\n\t - epochs: {epochs}" +
                    f"\n\t - batch_size: {batch_size}" +
                    f"\n\t - num_workers: {num_workers}" +
                    f"\n\t - ckpt_every: {ckpt_every}" +
                    f"\n\t - log_every: {log_every}")
        logger.info(f"Dataset abstract: " +
                    f"\n\t - length: {len(dataset)}" +
                    f"\n\t - image_size: {image_size}" +
                    f"\n\t - num_classes: {num_classes}")

    assert image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = image_size // 8

    # Load dit model
    dit_model = DiT_models[model_type](
        input_size=latent_size,
        num_classes=num_classes,
    ).to(device)
    # state_dict = load_dit_model(dit_model_path)
    # dit_model.load_state_dict(state_dict)
    dit_model.eval()

    # Create model:
    model = ControlDiT_models[model_type](
        dit=dit_model,
        input_size=latent_size,
        num_classes=num_classes
    )
    # Note that parameter initialization is done within the DiT constructor
    model = model.to(device)
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    if accelerator.is_main_process:
        logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)

    # Prepare models for training:
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    model, opt, loader = accelerator.prepare(model, opt, loader)

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    if accelerator.is_main_process:
        logger.info(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        if accelerator.is_main_process:
            logger.info(f"Beginning epoch {epoch}...")
        for x, y, z in loader:
            x = x.to(device)
            y = y.to(device)
            z = z.to(device)
            x = x[0].squeeze(dim=1)
            y = y[0].squeeze(dim=1).long()
            z = z[0].squeeze(dim=1)
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y, z=z)
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
            if train_steps % log_every == 0:
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
            if train_steps % ckpt_every == 0 and train_steps > 0:
                if accelerator.is_main_process:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    if accelerator.is_main_process:
        logger.info("Done!")


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="/gemini/data-1/imagenette_processed_train", help="input")
    parser.add_argument("--output", type=str, default="/gemini/output/control-dit_train_baseline-v4", help="output")
    parser.add_argument("--model-type", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--dit-model-path", type=str, default="/gemini/pretrain/checkpoints/DiT-XL-2-256x256.pt",
                        help="dit model path")

    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)

    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)

    args = parser.parse_args()
    main(args)
