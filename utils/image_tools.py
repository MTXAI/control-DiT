import shutil
import re
import os

from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from PIL import Image
import torch
import numpy as np
import cv2 as cv


class ImageTools(object):
    def __init__(self, ):
        pass

    @classmethod
    def gray(cls, img):
        return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    @classmethod
    def gamma_convert(cls, gray_img):
        # fi = gray_img / 255
        gamma = 0.5
        new_img = np.power(gray_img, gamma)
        new_img = np.uint8(np.clip(new_img, 0, 255))
        return new_img

    @classmethod
    def enhance(cls, gray_img, clip_limit=4.0, grid_size=(4, 4)):
        clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)

        enhanced_img = clahe.apply(gray_img)
        return enhanced_img

    @classmethod
    def blur(cls, gray_img, ksize=(11, 11), sigma=(0, 0)):
        return cv.GaussianBlur(gray_img, ksize=ksize, sigmaX=sigma[0], sigmaY=sigma[1])

    @classmethod
    def resize(cls, img, dsize=(256, 256)):
        img = cv.resize(img, dsize=dsize, interpolation=cv.INTER_LINEAR)
        return img

    @classmethod
    def read(cls, filename):
        return cv.imread(filename)

    @classmethod
    def write(cls, img, filename):
        cv.imwrite(filename, img)

    @classmethod
    def display(cls, img):
        plt.imshow(img)

    @classmethod
    def to_binary(cls, gray_img):
        (_, image_binary) = cv.threshold(gray_img, 0, 255, cv.THRESH_OTSU)
        return image_binary

    @classmethod
    def to_deep(cls, img, model_type="MiDaS_small", cuda=False, model_repo_or_path="intel-isl/MiDaS", ):
        # gray_img = cls.resize(gray_img, (256, 256))
        # model_type = "DPT_Large"  # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
        # model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
        # model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
        midas = torch.hub.load("intel-isl/MiDaS", model_type)
        device = torch.device("cuda") if cuda else torch.device("cpu")
        midas.to(device)
        midas.cuda() if cuda else midas.eval()

        # https://github.com/isl-org/MiDaS#Accuracy
        # https://pytorch.org/hub/intelisl_midas_v2/
        midas_transforms = torch.hub.load(repo_or_dir=model_repo_or_path, model="transforms")

        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            transform = midas_transforms.dpt_transform
        else:
            transform = midas_transforms.small_transform
        input_batch = transform(img).to(device)
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        output = prediction.cuda().numpy() if cuda else prediction.cpu().numpy()
        return output

    @classmethod
    def deep_2_cv2(cls, deep_img):
        return cv.applyColorMap(cv.convertScaleAbs(deep_img, alpha=7), cv.COLORMAP_JET)

    @classmethod
    def cv2_to_tensor(cls, img):
        pass

    @classmethod
    def cv2_to_pil(cls, img):
        return Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))

    @classmethod
    def tensor_to_cv2(cls, tensor):
        pass

    @classmethod
    def tensor_to_pil(cls, img):
        pass

    @classmethod
    def pil_to_tensor(cls, img):
        pass

    @classmethod
    def pil_to_cv2(cls, img):
        return cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)

