import cv2 as cv
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms


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
    def load_deep_model(cls, model_type="MiDaS_small", cuda=False, model_repo_or_path="intel-isl/MiDaS", source="github"):
        # gray_img = cls.resize(gray_img, (256, 256))
        # model_type = "DPT_Large"  # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
        # model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
        # model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
        midas = torch.hub.load(repo_or_dir=model_repo_or_path, source=source, model=model_type)
        device = torch.device("cuda") if cuda else torch.device("cpu")
        midas.to(device)
        midas.cuda() if cuda else midas.eval()

        # https://github.com/isl-org/MiDaS#Accuracy
        # https://pytorch.org/hub/intelisl_midas_v2/
        midas_transforms = torch.hub.load(repo_or_dir=model_repo_or_path, source=source, model="transforms")

        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            transform = midas_transforms.dpt_transform
        else:
            transform = midas_transforms.small_transform

        return midas, transform

    @classmethod
    def to_deep(cls, img, cuda, model, transform):
        device = torch.device("cuda") if cuda else torch.device("cpu")
        input_batch = transform(img).to(device)
        with torch.no_grad():
            prediction = model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        output = prediction.cpu().numpy()
        return output

    @classmethod
    def deep_2_cv2(cls, deep_img):
        return cv.applyColorMap(cv.convertScaleAbs(deep_img, alpha=7), cv.COLORMAP_JET)

    @classmethod
    def cv2_to_pil_tensor(cls, img):
        # HWC to CHW
        img = img.transpose((2, 0, 1))
        return torch.from_numpy(img)

    @classmethod
    def cv2_to_pil(cls, img):
        return Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))

    @classmethod
    def pil_tensor_to_cv2(cls, tensor):
        # CHW to HWC
        # img = tensor.permute((1, 2, 0)).numpy()
        array = tensor.numpy()  # 将tensor数据转为numpy数据
        array = array * 255 / array.max()  # normalize，将图像数据扩展到[0,255]
        mat = np.uint8(array)  # float32-->uint8
        mat = mat.transpose(1, 2, 0)
        return cv.cvtColor(mat, cv.COLOR_BGR2RGB)

    @classmethod
    def pil_tensor_to_pil(cls, tensor):
        return transforms.ToPILImage()(tensor).convert('RGB')

    @classmethod
    def pil_to_pil_tensor(cls, img):
        return transforms.ToTensor()(img)

    @classmethod
    def pil_to_cv2(cls, img):
        return cv.cvtColor(np.array(img), cv.COLOR_RGB2BGR)

