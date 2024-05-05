import argparse
import concurrent.futures
import os

import torch
import torchvision.transforms as transforms

from utils.dataset import CustomCocoDataset
from utils.image_tools import ImageTools

Transform = transforms.PILToTensor()
CUDA = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor
Device = "cuda" if CUDA else "cpu"
imgTools = ImageTools()


def get_resized_image_file_name(origin_file_name) -> (str, str):
    resized_file_name = os.path.splitext(origin_file_name)[0] + "_resized.jpg"
    resized_deep_file_name = os.path.splitext(origin_file_name)[0] + "_deep_resized.jpg"
    return resized_file_name, resized_deep_file_name


def get_resized_image_file_dir(dataset) -> str:
    abs_dir = os.path.abspath(dataset)
    dir_name = os.path.basename(abs_dir)
    return dir_name + "_resized"


def get_resized_image_file_path(dataset, origin_file_name) -> (str, str):
    resized_file_name, resized_deep_file_name = get_resized_image_file_name(origin_file_name)
    resized_file_dir = get_resized_image_file_dir(dataset)

    resized_file_path = os.path.join(resized_file_dir, resized_file_name)
    resized_deep_file_path = os.path.join(resized_file_dir, resized_deep_file_name)
    return resized_file_path, resized_deep_file_path


def generate_deep_graph_batch(dataset: CustomCocoDataset, begin, end, args):
    image_size = args.image_size
    for i in range(begin, end):
        tensor, annotation = dataset[i]
        print(tensor.shape, annotation)

        cv_img = imgTools.pil_tensor_to_cv2(tensor)
        cv_img_resized = imgTools.resize(cv_img, (image_size, image_size))

        deep_img_resized = imgTools.to_deep(cv_img_resized, model_type="DPT_Large")
        cv_deep_img_resized = imgTools.deep_2_cv2(deep_img_resized)

        print(f"origin size: {cv_img.shape}, new size: {cv_deep_img_resized.shape}")

        # get output file path
        img_id = dataset.ids[i]
        origin_file_name = dataset.get_filename_by_id(img_id)
        resized_file_path, resized_deep_file_path = get_resized_image_file_path(args.dataset, origin_file_name)

        # save resized img
        imgTools.write(cv_img_resized, resized_file_path)
        imgTools.write(cv_deep_img_resized, resized_deep_file_path)


def generate_deep_graph_parallel(dataset: CustomCocoDataset, args):
    batch = args.batch
    parallel = args.parallel
    if parallel <= 0:
        parallel = 1
    with concurrent.futures.ProcessPoolExecutor(max_workers=parallel) as executor:
        begin, end = 0, 0
        for i in range(len(dataset)):
            end += 1
            if (end-begin) == batch:
                executor.submit(generate_deep_graph_batch, dataset, begin, end, args)
                begin = end


def main(args):
    annotation_path = str(os.path.join(args.dataset, args.annotation))
    coco_dataset = CustomCocoDataset(root=args.dataset, annotation_file=annotation_path, make_index=True, transform=Transform)
    print('Number of samples: ', len(coco_dataset))
    generate_deep_graph_parallel(coco_dataset, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="./datasets/example/trainval")
    parser.add_argument("--annotation", type=str,
                        default="captions_example_trainval2017.json")
    parser.add_argument("--batch", type=int, default=32,
                        help="size of preprocessing image batch, estimated based on image size and memory capacity")
    parser.add_argument("--parallel", type=int, default=0,
                        help="number of parallel processes number, default is 0, means using main process")
    parser.add_argument("--image-size", type=int, default=256)
    main(parser.parse_args())

