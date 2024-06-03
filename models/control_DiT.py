# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

from models.DiT import modulate, DiT, get_2d_sincos_pos_embed


def operator_add(x, z):
    return x + z


#################################################################################
#                                 Core ControlDiT Model                                #
#################################################################################

class ControlDiTBlock(nn.Module):
    """
    A ControlDiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of ControlDiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class ControlDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
            self,
            dit: DiT = None,
            input_size=32,
            patch_size=2,
            in_channels=4,
            hidden_size=1152,
            depth=28,
            num_heads=16,
            mlp_ratio=4.0,
            class_dropout_prob=0.1,
            num_classes=1000,
            learn_sigma=True,
    ):
        super().__init__()
        self.dit = dit
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.z_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        num_patches = self.z_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            ControlDiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.dit.eval()
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.z_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.z_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.z_embedder.proj.bias, 0)

        # Zero-out adaLN modulation layers in ControlDiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.z_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def ckpt_wrapper(self, module):
        def ckpt_forward(*inputs):
            outputs = module(*inputs)
            return outputs

        return ckpt_forward

    def dit_forward(self, x, t, y):
        x = self.dit.x_embedder(x) + self.dit.pos_embed
        t = self.dit.t_embedder(t)  # (N, D)
        y = self.dit.y_embedder(y, self.training)  # (N, D)
        c = t + y  # (N, D)
        return x, c

    def forward(self, x, t, y, z):
        """
        Forward pass of ControlDiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x, c = self.dit_forward(x, t, y)
        z = self.z_embedder(z) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        for i in range(len(self.blocks)):
            dit_block = self.dit.blocks[i]
            block = self.blocks[i]
            x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(dit_block), x, c)  # (N, T, D)
            z = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), operator_add(x, z), c)  # (N, T, D)
        z = self.final_layer(z, c)  # (N, T, patch_size ** 2 * out_channels)
        z = self.unpatchify(z)  # (N, out_channels, H, W)
        return z

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of ControlDiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                                   ControlDiT Configs                                  #
#################################################################################

def ControlDiT_XL_2(**kwargs):
    return ControlDiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def ControlDiT_XL_4(**kwargs):
    return ControlDiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)


def ControlDiT_XL_8(**kwargs):
    return ControlDiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)


def ControlDiT_L_2(**kwargs):
    return ControlDiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def ControlDiT_L_4(**kwargs):
    return ControlDiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)


def ControlDiT_L_8(**kwargs):
    return ControlDiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)


def ControlDiT_B_2(**kwargs):
    return ControlDiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


def ControlDiT_B_4(**kwargs):
    return ControlDiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)


def ControlDiT_B_8(**kwargs):
    return ControlDiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)


def ControlDiT_S_2(**kwargs):
    return ControlDiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


def ControlDiT_S_4(**kwargs):
    return ControlDiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)


def ControlDiT_S_8(**kwargs):
    return ControlDiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


ControlDiT_models = {
    'DiT-XL/2': ControlDiT_XL_2, 'DiT-XL/4': ControlDiT_XL_4, 'DiT-XL/8': ControlDiT_XL_8,
    'DiT-L/2': ControlDiT_L_2, 'DiT-L/4': ControlDiT_L_4, 'DiT-L/8': ControlDiT_L_8,
    'DiT-B/2': ControlDiT_B_2, 'DiT-B/4': ControlDiT_B_4, 'DiT-B/8': ControlDiT_B_8,
    'DiT-S/2': ControlDiT_S_2, 'DiT-S/4': ControlDiT_S_4, 'DiT-S/8': ControlDiT_S_8,
}
