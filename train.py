import argparse
import logging
from copy import deepcopy
from glob import glob
from time import time
from pathlib import Path

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from src.diffusion import create_diffusion
from src.utils.dataset import CustomDataset
from src.utils.log import init_logger
from src.models import *
from src.utils.model import *

logger = logging.Logger('')
experiment_dir = ''
checkpoint_dir = ''



def log_info(s, in_main_process=False):
    if in_main_process:
        logger.info(s)


def log_step_metrics(prefix,
                     current_step,
                     steps,
                     start_time,
                     loss,
                     num_processes,
                     device,
                     sync_cuda=True,
                     in_main_process=False):
    # Measure training speed:
    torch.cuda.synchronize()
    end_time = time()
    steps_per_sec = steps / (end_time - start_time)
    datas_per_sec = steps * args.batch_size / (end_time - start_time)
    # Reduce loss history over all processes:
    avg_loss = torch.tensor(loss / steps, device=device)
    avg_loss = avg_loss.item() / num_processes
    log_info(f"[{prefix}](step={current_step:07d}) " +
             f"Train Loss: {avg_loss:.4f}, " +
             f"Train Steps/Sec: {steps_per_sec:.2f}, " +
             f"Process data/Sec: {datas_per_sec:.2f}",
             in_main_process=in_main_process)

    return float(f'{avg_loss:.4f}')


def save_loss_plot(output, epoch, losses, in_main_process=False):
    if not in_main_process:
        return
    x = range(len(losses))
    y = losses
    plt.plot(x, y, label='train loss', linewidth=2, color='r', marker='o', markerfacecolor='r', markersize=5)
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    path = os.path.join(output, f'loss-{epoch}-{int(time())}.jpg')
    plt.savefig(path)
    log_info(f"[Plot](epoch={epoch}) save loss plot in {path}", in_main_process=in_main_process)

    plot_files = os.listdir(output)
    for file in plot_files:
        items = file.split('-')
        if len(items) != 3:
            continue
        if items[0] != 'loss':
            continue
        old_epoch = items[1]
        if epoch > int(old_epoch):
            path = os.path.join(output, file)
            os.remove(path)
            log_info(f"Removed loss plot in {path}")


def save_and_clean_checkpoint(epoch, checkpoint, suffix, auto_clean_ckpt, in_main_process=False):
    checkpoint_path = f"{checkpoint_dir}/{epoch:07d}-{suffix}.pt"
    torch.save(checkpoint, checkpoint_path)
    log_info(f"Saved checkpoint to {checkpoint_path}", in_main_process=in_main_process)

    if auto_clean_ckpt:
        files = os.listdir(checkpoint_dir)
        remove_ckpt_names = []
        for file in files:
            items = file.split('-')
            if len(items) != 2:
                continue
            old_epoch, old_suffix = items[0], items[1].replace('.pt', '')
            if old_suffix != suffix:
                continue
            if epoch > int(old_epoch):
                remove_ckpt_names.append(file)
        if len(remove_ckpt_names) == 0:
            log_info('No checkpoint to remove')
            return
        for file in remove_ckpt_names:
            checkpoint_path = os.path.join(checkpoint_dir, file)
            os.remove(checkpoint_path)
            log_info(f"Removed checkpoint in {checkpoint_path}")


def prepare_output_dir(output, model_type):
    os.makedirs(output, exist_ok=True)  # Make results folder (holds all experiment subfolders)
    experiment_index = len(glob(f"{output}/*"))
    model_string_name = model_type.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
    experiment_dir = f"{output}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    os.makedirs(checkpoint_dir, exist_ok=True)
    return experiment_dir, checkpoint_dir


def prepare_all(args) -> (Accelerator, DataLoader, str):
    global logger
    global experiment_dir
    global checkpoint_dir

    # Setup accelerator and device
    accelerator, device = setup_accelerator(not cuda)

    # Setup dataloader
    batch_size = int(args.batch_size // accelerator.num_processes)
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

    # Prepare output dirs
    if accelerator.is_main_process:
        # Setup an experiment folder and init logger
        experiment_dir, checkpoint_dir = prepare_output_dir(args.output, args.model_type)
        logger = init_logger(experiment_dir)
        log_info(f"Experiment directory created at {experiment_dir}", True)
        log_info(f"Input information: " +
                 f"\n\t - experiment_dir: {experiment_dir}" +
                 f"\n\t - checkpoint_dir: {checkpoint_dir}" +
                 f"\n\t - features_dir: {features_dir}" +
                 f"\n\t - labels_dir: {labels_dir}" +
                 f"\n\t - conditions_dir: {conditions_dir}" +
                 f"\n\t - dit_model_ckpt: {args.dit_model_ckpt}" +
                 f"\n\t - model_ckpt: {args.model_ckpt}", True)
        log_info(f"Train options: " +
                 f"\n\t - device: {device}" +
                 f"\n\t - model_type: {args.model_type}" +
                 f"\n\t - epochs: {args.epochs}" +
                 f"\n\t - lr: {args.lr}" +
                 f"\n\t - decay: {args.decay}" +
                 f"\n\t - batch_size: {batch_size}" +
                 f"\n\t - vae_scale_factor: {args.vae_scale_factor}" +
                 f"\n\t - in_channels: {args.in_channels}" +
                 f"\n\t - num_workers: {args.num_workers}" +
                 f"\n\t - ckpt_every_epoch: {args.ckpt_every_epoch}" +
                 f"\n\t - ckpt_ema_every_epoch: {args.ckpt_ema_every_epoch}" +
                 f"\n\t - auto_clean_ckpt: {args.auto_clean_ckpt}" +
                 f"\n\t - log_every: {args.log_every}", True)
        log_info(f"Dataset abstract: " +
                 f"\n\t - length: {len(dataset)}" +
                 f"\n\t - image_size: {args.image_size}" +
                 f"\n\t - num_classes: {args.num_classes}", True)

    return accelerator, device, loader


def main(args):
    """
    Trains a new DiT model.
    """
    # Setup accelerator, device and dataloader:
    assert cuda, "Training currently requires at least one GPU."
    accelerator, device, loader = prepare_all(args)

    # Create dit model
    assert args.image_size % args.vae_scale_factor == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // args.vae_scale_factor
    model = create_control_dit_model_baseline_v2(args.model_type)(
        input_size=latent_size,
        num_classes=args.num_classes,
        in_channels=args.in_channels,
    ).to(device)
    model_ckpt = ''
    if args.model_ckpt != '' and args.model_ckpt != none_model:
        model_ckpt = args.model_ckpt

    # Load dit model from checkpoint or pretrained checkpoint
    if model_ckpt != '':
        log_info(f"Loading model from {args.model_ckpt}", True)
        model_dict = load_model(model_ckpt)
        model.load_state_dict(model_dict['model'])
    elif args.dit_model_ckpt != '':
        log_info(f"Loading dit model from {args.dit_model_ckpt}", True)
        state_dict = load_pretrained_dit_model(args.dit_model_ckpt)
        # pop unmatched params
        # state_dict.pop('x_embedder.proj.weight')
        # state_dict.pop('final_layer.linear.weight')
        # state_dict.pop('final_layer.linear.bias')
        model.load_state_dict(state_dict, strict=False)

    # Create diffusion pipeline and optimizer
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    log_info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}",
             in_main_process=accelerator.is_main_process)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.decay)
    if model_ckpt != '':
        opt.load_state_dict(model_dict['opt'])

    # Prepare models for training
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode
    model, opt, loader = accelerator.prepare(model, opt, loader)

    # Begin training
    train_steps = 0

    log_steps = 0
    running_loss = 0
    start_time = time()

    epoch_steps = 0
    epoch_loss = 0
    epoch_start_time = time()

    epoch_losses_10 = []
    epoch_losses = []

    first_epoch = 0
    if model_ckpt != '':
        first_epoch = model_dict['epoch'] + 1
    log_info(f"Training for {args.epochs} epochs, First epoch is {first_epoch}...",
             in_main_process=accelerator.is_main_process)
    for epoch in range(first_epoch, first_epoch + args.epochs):
        log_info(f"Beginning epoch {epoch}...", in_main_process=accelerator.is_main_process)

        for x, y, z in loader:
            x = x.to(device)
            y = y.to(device)
            z = z.to(device)

            x = x.squeeze(dim=1)
            y = y.squeeze(dim=1).long()
            y = y.squeeze(dim=1)
            z = z.squeeze(dim=1)

            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y, z=z)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()
            update_ema(ema, model)

            # Log loss values:
            train_steps += 1
            last_loss = loss.item()

            running_loss += last_loss
            log_steps += 1

            epoch_loss += last_loss
            epoch_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                log_step_metrics(prefix='Batch', current_step=train_steps, steps=log_steps, start_time=start_time,
                                 loss=running_loss, num_processes=accelerator.num_processes, device=device,
                                 sync_cuda=False, in_main_process=accelerator.is_main_process)
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

        # Save DiT checkpoint:
        if epoch % args.ckpt_every_epoch == 0 and epoch > 0:
            if accelerator.is_main_process:
                checkpoint = {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args
                }
                save_and_clean_checkpoint(epoch, checkpoint, 'full',
                                          args.auto_clean_ckpt, in_main_process=True)

        # Save DiT ema checkpoint:
        if epoch % args.ckpt_ema_every_epoch == 0 and epoch > 0:
            if accelerator.is_main_process:
                checkpoint_ema = {
                    "epoch": epoch,
                    "ema": ema.state_dict(),
                    "args": args
                }
                save_and_clean_checkpoint(epoch, checkpoint_ema, 'ema',
                                          args.auto_clean_ckpt, in_main_process=True)
        # Measure training speed of epoch:
        avg_loss = log_step_metrics(prefix='Epoch', current_step=train_steps, steps=epoch_steps, start_time=epoch_start_time,
                         loss=epoch_loss, num_processes=accelerator.num_processes, device=device,
                         sync_cuda=True, in_main_process=accelerator.is_main_process)
        if epoch >= 10:
            epoch_losses.append(avg_loss)
            save_loss_plot(args.output, epoch, epoch_losses, in_main_process=accelerator.is_main_process)
        else:
            epoch_losses_10.append(avg_loss)
            save_loss_plot(args.output, epoch, epoch_losses_10, in_main_process=accelerator.is_main_process)


        # todo 根据 loss 自动停止训练，并且保存最终 checkpoint+ema

        # Reset monitoring variables:
        epoch_loss = 0
        epoch_steps = 0
        epoch_start_time = time()

    # final save
    if accelerator.is_main_process:
        checkpoint_ema = {
            "epoch": args.epochs,
            "ema": ema.state_dict(),
            "args": args
        }
        save_and_clean_checkpoint(args.epochs, checkpoint_ema, 'ema_final',
                                  args.auto_clean_ckpt, in_main_process=True)

        checkpoint = {
            "epoch": args.epochs,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "opt": opt.state_dict(),
            "args": args
        }
        save_and_clean_checkpoint(args.epochs, checkpoint, 'full_final',
                                  args.auto_clean_ckpt, in_main_process=True)
    model.eval()  # important! This disables randomized embedding dropout

    log_info("Done!", in_main_process=accelerator.is_main_process)


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()

    # Input and output
    parser.add_argument("--input", type=str, default="/gemini/data-1/imagenette_processed_train", help="input")
    parser.add_argument("--output", type=str, default="/gemini/output/control-dit_train_baseline-v4", help="output")
    parser.add_argument("--dit-model-ckpt", type=str, default="/gemini/pretrain/checkpoints/DiT-XL-2-256x256.pt")
    parser.add_argument("--model-ckpt", type=str, default="")

    # Train options
    parser.add_argument("--model-type", type=str, default="DiT-XL/2")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--decay", type=float, default=1e-1)
    parser.add_argument("--batch-size", type=int, default=256, help="batch size should >= 6*2")
    parser.add_argument("--vae-scale-factor", type=int, default=8)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--in-channels", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--ckpt-every-epoch", type=int, default=200)
    parser.add_argument("--ckpt-ema-every-epoch", type=int, default=25, help="save ema and generate a sample")
    parser.add_argument("--auto-clean-ckpt", action='store_true', help="auto clean all old checkpoint with same suffix")
    parser.add_argument("--log-every", type=int, default=100)

    # Dataset abstract
    parser.add_argument("--num-classes", type=int, default=1000)

    args = parser.parse_args()
    main(args)
