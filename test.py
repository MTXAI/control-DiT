import numpy as np
import torch

N = 2
H = 32
W = 32
patch_size = 2
C = 5
x = torch.zeros((N, H * W, patch_size**2 * C))
print(x.shape)
assert H * W == x.shape[1]

x = x.reshape(shape=(x.shape[0], H, W, patch_size, patch_size, C))
print(x.shape)

x = torch.einsum('nhwpqc->nchpwq', x)
print(x.shape)

imgs = x.reshape(shape=(x.shape[0], C, H * patch_size, H * patch_size))
print(imgs.shape, imgs)

imgs = imgs[:, :4]
print(imgs.shape, imgs)
