import numpy as np
import torch

x = torch.zeros((2, 4, 32, 32))
z = torch.zeros((2, 1, 32, 32))
if x.shape[0] == 1:
    x = torch.cat([x, x], dim=0)
print(x.shape, z.shape)

if x.shape[0] == 2 and x.shape[0] != z.shape[0]:
    half = x[: len(x) // 2]
    x = torch.cat([half, half], dim=0)
    print(x.shape, z.shape)
    x1 = torch.cat([x[0], z[0]], dim=0).unsqueeze(0)
    x2 = torch.cat([x[1], z[0]], dim=0).unsqueeze(0)
    print(x1.shape, x2.shape)

    x = torch.cat([x1, x2], dim=0)
print(x.shape, z.shape)

if x.shape[0] == z.shape[0]:
    x = torch.cat([x, z], dim=1)

print(x.shape, z.shape)

if x.shape[0] == z.shape[0]:
    x1 = x[:, :5 - z.shape[1]]
    x2 = x[:, 5:5 * 2 - z.shape[1]]
    x = torch.cat([x1, x2], dim=1)
else:
    x = x[:, :5 - z.shape[1]]
print(x.shape, z.shape)

if x.shape[0] == 2 and x.shape[0] != z.shape[0]:
    eps, rest = x[:, :3], x[:, 3:]
    cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
    half_eps = uncond_eps + 4.0 * (cond_eps - uncond_eps)
    eps = torch.cat([half_eps, half_eps], dim=0)
    sample = torch.cat([eps, rest], dim=1)

    print(sample.shape)

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
