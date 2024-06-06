import cv2
import numpy as np

im1 = cv2.imread('../sample1.png')
im2 = cv2.imread('../sample2.png')
print(np.sum(im1!=im2))
