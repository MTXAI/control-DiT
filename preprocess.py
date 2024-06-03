# -- coding: utf-8 --

import argparse
import json
import os

import numpy as np
import torch
import torchvision.transforms as transforms
import torch.distributed as dist
from PIL import Image
from accelerate import Accelerator
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder, ImageNet

from utils.image_tools import ImageTools

CUDA = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor
imgTools = ImageTools()
midas, transform = None, None


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def create_class_index(class_index_path):
    index_dict = {}
    with open(class_index_path, 'r', encoding='utf-8') as f:
        items = json.load(f)
        for idx, cls in items.items():
            index_dict[cls[0]] = [idx, cls[1]]
    return index_dict


def main(args):
    # assert CUDA, "Training currently requires at least one GPU."

    # Setup accelerator:
    accelerator = Accelerator(not CUDA)
    device = accelerator.device
    # device = "cpu"

    print(f'[CUDA] == {CUDA}\n[Derive] == {device}')

    root = args.input
    new_root = args.output
    features_dir = os.path.join(new_root, f'imagenet{args.image_size}_features')
    labels_dir = os.path.join(new_root, f'imagenet{args.image_size}_labels')
    conditions_dir = os.path.join(new_root, f'imagenet{args.image_size}_conditions')

    if accelerator.is_main_process:
    # if True:
        os.makedirs(new_root, exist_ok=True)
        os.makedirs(features_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        os.makedirs(conditions_dir, exist_ok=True)

    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."

    # Setup data:
    def resize(pil_image):
        return center_crop_arr(pil_image, args.image_size)

    dataset = ImageFolder(str(root), transform=transforms.Compose([
        transforms.Lambda(resize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ]))

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    midas, transform = imgTools.load_deep_model(model_type="DPT_Large", cuda=CUDA,
                                                model_repo_or_path=args.deep_model, source=args.deep_model_source)
    vae = AutoencoderKL.from_pretrained(args.vae_model).to(device)

    # create index
    index = create_class_index(args.class_index)

    idx = -1
    for x, y in loader:
        idx += 1

        y = y.to(device)
        y = y.detach().cpu().numpy()
        index_key = dataset.classes[y[0]]

        filename = f'{index[index_key][0]}_{index_key}_{index[index_key][1]}_{idx}.npy'
        features_path = os.path.join(features_dir, filename)
        labels_path = os.path.join(labels_dir, filename)
        conditions_path = os.path.join(conditions_dir, filename)

        print(f'===Process: [{idx + 1}/{len(dataset)}]:')

        if not os.path.exists(labels_path):
            print(f'Class: {y}-{index_key}-{index[index_key]}')
            y = np.asarray([[index[index_key][0]]], dtype=np.int16)
            np.save(labels_path, y)
            print(f'y filepath: {labels_path}, shape: {y.shape}')
        else:
            print(f'y filepath: {labels_path}, has existed')

        if not os.path.exists(features_path) or not os.path.exists(conditions_path):
            x = x.to(device)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                feature = vae.encode(x).latent_dist.sample().mul_(0.18215)
                feature = feature.detach().cpu().numpy()
                np.save(features_path, feature)
                print(f'feature filepath: {features_path}, shape: {feature.shape}')

            z = imgTools.to_deep(imgTools.pil_tensor_to_cv2(x[0]), CUDA, midas, transform)
            z = z.expand_as(x[0])
            z = z.expand_as(x)
            z = z.to(device)
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                condition = vae.encode(z).latent_dist.sample().mul_(0.18215)
                condition = condition.detach().cpu().numpy()
                np.save(features_path, feature)
                np.save(conditions_path, condition)
                print(f'condition filepath: {conditions_path}, shape: {condition.shape}')
        else:
            print(f'feature filepath: {features_path} and condition filepath: {conditions_path}, has existed')

        if len(index[index_key]) == 0:
            print(index_key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="/gemini/data-1/train")
    parser.add_argument("--output", type=str, default="/gemini/output/preprocessed_train")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="number of workers, default is 0, means using main process")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--class-index", type=str, default="./annotations/imagenet_class_index.json")
    parser.add_argument("--deep-model", type=str, default="intel-isl/MiDaS")
    parser.add_argument("--vae-model", type=str, default="stabilityai/sd-vae-ft-ema")
    parser.add_argument("--deep-model-source", type=str, default="github")

    args = parser.parse_args()
    main(args)
