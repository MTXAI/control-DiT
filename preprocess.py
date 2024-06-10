import argparse
import json
import os

from diffusers.models import AutoencoderKL
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.utils import *
from src.utils.img_tools import *
from src.utils.model import load_depth_model


def create_class_index(class_index_path):
    index_dict = {}
    with open(class_index_path, 'r', encoding='utf-8') as f:
        items = json.load(f)
        for idx, cls in items.items():
            index_dict[cls[0]] = [idx, cls[1]]
    return index_dict


def main(args):
    # assert cuda, "Training currently requires at least one GPU."

    # 1. Setup accelerator:
    accelerator, device = setup_accelerator(not cuda)

    root = args.input
    new_root = args.output
    features_dir = os.path.join(new_root, f'imagenet{args.image_size}_features')
    labels_dir = os.path.join(new_root, f'imagenet{args.image_size}_labels')
    conditions_dir = os.path.join(new_root, f'imagenet{args.image_size}_conditions')

    if accelerator.is_main_process:
        os.makedirs(new_root, exist_ok=True)
        os.makedirs(features_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        os.makedirs(conditions_dir, exist_ok=True)

    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8

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

    midas, transform = load_depth_model(model_type="DPT_Large", model_repo_or_path=args.deep_model,
                                        source=args.deep_model_source, device=device)
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
            label = np.asarray([[index[index_key][0]]], dtype=np.int16)  # (1, 1)
            np.save(labels_path, label)
            print(f'label filepath: {labels_path}, shape: {label.shape}')
        else:
            print(f'label filepath: {labels_path}, has existed')

        if not os.path.exists(features_path) or not os.path.exists(conditions_path):
            x = x.to(device)
            with torch.no_grad():
                z = cv2_to_depth(tensor_to_cv2(x[0]), midas, transform, device)
                z = z.unsqueeze(0).to(device)
                condition = depth_to_map(z, latent_size, 1, False)
                condition = condition.detach().cpu().numpy()  # (1, 1, 32, 32)
                np.save(conditions_path, condition)
                print(f'condition filepath: {conditions_path}, shape: {condition.shape}')

                # Map input images to latent space + normalize latents:
                feature = vae.encode(x).latent_dist.sample().mul_(0.18215)
                feature = feature.detach().cpu().numpy()  # (1, 4, 32, 32)
                np.save(features_path, feature)
                print(f'feature filepath: {features_path}, shape: {feature.shape}')
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
    parser.add_argument("--vae-model", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--deep-model-source", type=str, default="github")

    args = parser.parse_args()
    main(args)
