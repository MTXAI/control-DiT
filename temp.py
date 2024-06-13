import cv2

from src.utils.img_tools import *
from src.utils.model import load_depth_model

midas, transform = load_depth_model(model_type="DPT_Large", device='cpu')

x = cv2.imread("hack/chores/5-train-baseline-2-opt/sample-64-4090-4-epoch900_result.png")

z = cv2_to_depth(x, midas, transform, 'cpu')
z = z.detach().cpu().numpy()
cv2.imwrite('sample-64-4090-4-epoch900_result-depth.png', z)


