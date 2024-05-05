import argparse
import concurrent.futures
import os
import cv2 as cv

import torch
import torchvision.transforms as transforms
from PIL.Image import Image

from utils.dataset import CustomCocoDataset
from utils.image_tools import ImageTools

Transform = transforms.PILToTensor()
CUDA = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor
Device = "cuda" if CUDA else "cpu"
imgTools = ImageTools()


def main():
    os.remove("./output")
    os.mkdir("./output")
    # train: 118287
    # val: 5000
    dataset = "./datasets/COCO_Captions/train2017"
    annotation = "./datasets/COCO_Captions/annotations_trainval2017/annotations/instances_train2017.json"
    coco_dataset = CustomCocoDataset(root=dataset, annotation_file=annotation, make_index=True,
                                     transform=Transform)
    print('Number of samples: ', len(coco_dataset))

    img, anno = coco_dataset[101]
    img_name = coco_dataset.get_filename_by_id(anno[0]["image_id"])
    print(img_name, img.shape)

    pil_img = imgTools.pil_tensor_to_pil(img)
    Image.save(pil_img, "./output/pil.jpg")

    cv_img = imgTools.pil_to_cv2(pil_img)
    imgTools.write(cv_img, "./output/cv.jpg")

    i = 1
    for a in anno:
        print(a["category_id"])
        print(a["bbox"])
        print(a["area"])
        x, y, w, h = int(a["bbox"][0]), int(a["bbox"][1]), int(a["bbox"][2]), int(a["bbox"][3])
        bbox_img = cv_img[y:y+h, x:x+w]
        cate = a["category_id"]
        imgTools.write(bbox_img, f"./output/{cate}-{i}.jpg")
        i += 1

        # 对于 256*256 和 512*512 训练任务来说，选择 512 精度为基准
        # 过滤area小于 512*512*0.5的
        # 并且过滤w/h或h/w大于1.5的

        # 然后对每一个图块做一个 resize，再生成深度图
        # 并且生成新的 annotation：{image_id, file_name, deep_file_name, category_id}


if __name__ == "__main__":
    print(CUDA)
    main()
