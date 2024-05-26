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
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder, ImageNet

from utils.image_tools import ImageTools

CUDA = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor
imgTools = ImageTools()


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

    transform = transforms.Compose([
        transforms.Lambda(resize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(str(root), transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # create index
    index = create_class_index(args.class_index)

    idx = -1
    for x, y in loader:
        idx += 1

        index_key = dataset.classes[y[0]]
        print(f'==={idx}/{len(dataset)}, class: {y[0]}-{index_key}-{index[index_key]}')

        filename = f'{index[index_key][0]}_{index_key}_{index[index_key][1]}_{idx}.npy'
        features_path = os.path.join(features_dir, filename)
        labels_path = os.path.join(labels_dir, filename)
        conditions_path = os.path.join(conditions_dir, filename)

        if not os.path.exists(features_path) or not os.path.exists(conditions_path):
            x = x.to(device)
            x = x.detach().cpu().numpy()
            np.save(features_path, x)
            print(f'x filepath: {features_path}, shape: {x.shape}')
        else:
            print(f'x filepath: {features_path}, has existed')

        if not os.path.exists(labels_path):
            y = np.asarray([index[index_key][0]], dtype=np.int16)
            np.save(labels_path, y)
            print(f'y filepath: {labels_path}, shape: {y.shape}')
        else:
            print(f'y filepath: {labels_path}, has existed')

        if not os.path.exists(conditions_path):
            c = imgTools.to_deep(imgTools.pil_tensor_to_cv2(torch.from_numpy(x[0])), model_type="DPT_Large",
                                 cuda=CUDA, model_repo_or_path=args.deep_model, source="local")
            np.save(conditions_path, c)
            print(f'c filepath: {conditions_path}, shape: {c.shape}')
        else:
            print(f'c filepath: {conditions_path}, has existed')

        if len(index[index_key]) == 0:
            print(index_key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="/gemini/data-1/train")
    parser.add_argument("--output", type=str, default="/gemini/output/preprocessed_train")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="number of workers, default is 0, means using main process")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--class-index", type=str, default="./annotations/imagenet_class_index.json")
    parser.add_argument("--deep-model", type=str, default="intel-isl/MiDaS")
    main(parser.parse_args())
