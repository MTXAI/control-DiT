import cv2

from src.utils.img_tools import *
from src.utils.model import load_depth_model

x = cv2.imread('./datasets/example/n01440764/n01440764_1135.JPEG')
midas, transform = load_depth_model(model_type="DPT_Large", model_repo_or_path='intel-isl/MiDaS',
                                    source='github', device='cpu')
z = cv2_to_depth(x, midas, transform, 'cpu')
z = z.detach().cpu().numpy()
cv2.imwrite('sample2_depth.png', z)
