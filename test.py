import numpy as np
import torch

x = torch.zeros((2, 4, 32, 32))
z = torch.zeros((1, 1, 32, 32))

if x.shape[0]==2:
    x1 = torch.cat([x[:1], z], dim=1)
    x2 = torch.cat([x[1:], z], dim=1)
    x = torch.cat([x1, x2], dim=0)
else:
    x = torch.cat([x, z], dim=1)

print(x.shape, z.shape)

if x.shape[0]==2:
    x1 = x[:, :5 - z.shape[1]]
    x2 = x[:, 5:5 * 2 - z.shape[1]]
    x = torch.cat([x1, x2], dim=1)
else:
    x = x[:, :5 - z.shape[1]]
print(x.shape, z.shape)

# N = 2
# H = 32
# W = 32
# patch_size = 2
# C = 5
# x = torch.zeros((N, H * W, patch_size**2 * C))
# print(x.shape)
# assert H * W == x.shape[1]
#
# x = x.reshape(shape=(x.shape[0], H, W, patch_size, patch_size, C))
# print(x.shape)
#
# x = torch.einsum('nhwpqc->nchpwq', x)
# print(x.shape)
#
# imgs = x.reshape(shape=(x.shape[0], C, H * patch_size, H * patch_size))
# print(imgs.shape, imgs)
#
# imgs = imgs[:, :4]
# print(imgs.shape, imgs)
