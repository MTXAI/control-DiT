import torch

z = torch.rand(32, 1, 32, 32)
print(z.shape)

z = z.repeat(1, 4, 1, 1)
print(z.shape)
print((z[0][2]==z[0][3]))
