# -- coding: utf-8 --

import argparse
import json
import os

import numpy as np
import torch
import torchvision.transforms as transforms
import torch.distributed as dist
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder, ImageNet

from utils.image_tools import ImageTools

CUDA = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor
imgTools = ImageTools()


def create_class_index(class_index_path):
    index_dict = {}
    with open(class_index_path, 'r', encoding='utf-8') as f:
        items = json.load(f)
        for idx, cls in items.items():
            index_dict[cls[0]] = [idx, cls[1]]
    return index_dict


def main(args):
    # assert CUDA, "Training currently requires at least one GPU."

    # # Setup accelerator:
    # accelerator = Accelerator(not CUDA)
    # device = accelerator.device
    device = "cuda" if CUDA else "cpu"

    print(f'[CUDA] == {CUDA}\n[Derive] == {device}')


    root = os.path.join(args.root, args.split)
    new_root = args.root + '_processed_' + args.split
    features_dir = os.path.join(new_root, f'imagenet{args.image_size}_features')
    labels_dir = os.path.join(new_root, f'imagenet{args.image_size}_labels')
    conditions_dir = os.path.join(new_root, f'imagenet{args.image_size}_conditions')

    os.makedirs(new_root, exist_ok=True)
    os.makedirs(features_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(conditions_dir, exist_ok=True)

    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."

    # Setup data:
    # todo 添加对原始 image size 范围的筛选，如 256 筛选范围为 160 -- 320， 512 筛选范围为 360 -- 640
    dataset = ImageFolder(str(root), transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Resize([args.image_size, args.image_size]),
    ]))

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # create index
    index = create_class_index(args.class_index)

    idx = 0
    for x, y in loader:
        features_path = os.path.join(features_dir, f'{idx}.npy')
        labels_path = os.path.join(labels_dir, f'{idx}.npy')
        conditions_path = os.path.join(conditions_dir, f'{idx}.npy')

        index_key = dataset.classes[y[0]]
        print(f'==={idx}/{len(dataset)}, class: {index_key}-{index[index_key]}')

        if not os.path.exists(features_path) or not os.path.exists(conditions_path):
            x = x.to(device)
            x = x.detach().numpy()
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
            c = imgTools.to_deep(imgTools.pil_tensor_to_cv2(torch.from_numpy(x[0])), model_type="DPT_Large")
            np.save(conditions_path, c)
            print(f'c filepath: {conditions_path}, shape: {c.shape}')
        else:
            print(f'c filepath: {conditions_path}, has existed')

        idx += 1

        if len(index[index_key]) == 0:
            print(index_key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./datasets/tiny-imagenet-200")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="number of workers, default is 0, means using main process")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--class-index", type=str, default="./datasets/tiny-imagenet-200/imagenet_class_index.json")
    main(parser.parse_args())
