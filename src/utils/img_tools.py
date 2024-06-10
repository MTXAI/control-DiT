import PIL
import cv2 as cv
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


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


def cv2_to_gray(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def cv2_to_depth(img, midas, transform, device):
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    input_batch = transform(img).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    return prediction


def depth_to_map(depth_img, latent_size, batch_size, do_classifier_free_guidance):
    depth_map = torch.nn.functional.interpolate(
        depth_img.unsqueeze(1),
        size=(latent_size, latent_size),
        mode="bicubic",
        align_corners=False,
    )

    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = 2.0 * (depth_map - depth_min) / (depth_max - depth_min) - 1.0

    # duplicate mask and masked_image_latents for each generation per prompt, using mps friendly method
    if depth_map.shape[0] < batch_size:
        repeat_by = batch_size // depth_map.shape[0]
        depth_map = depth_map.repeat(repeat_by, 1, 1, 1)

    depth_map = torch.cat([depth_map] * 2) if do_classifier_free_guidance else depth_map
    return depth_map


def depth_2_cv2(depth_img, alpha=7):
    return cv.applyColorMap(cv.convertScaleAbs(depth_img, alpha=alpha), cv.COLORMAP_JET)


def cv2_to_tensor(img):
    # HWC to CHW
    img = img.transpose((2, 0, 1))
    return torch.from_numpy(img)


def cv2_to_pil(img):
    return Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))


def tensor_to_cv2(tensor):
    # CHW to HWC
    tensor = tensor.permute(1, 2, 0)
    return tensor.cpu().numpy()


def tensor_to_pil(tensor):
    return transforms.ToPILImage()(tensor).convert('RGB')


def pil_to_pil_tensor(img):
    return transforms.ToTensor()(img)


def pil_to_cv2(img):
    return cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)
