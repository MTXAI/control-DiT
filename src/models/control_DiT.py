# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
from copy import deepcopy

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
from torch import Tensor
from torch.nn import Module, init, Linear

from src.models.DiT import DiT, DiTBlock
from src.utils.model import auto_grad_checkpoint, requires_grad


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


#################################################################################
#                                 Core ControlDiT Model                                #
#################################################################################

class ControlDiTBlock(nn.Module):
    """
    A ControlDiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, block_index, base_dit_block: DiTBlock):
        super().__init__()
        self.block_index = block_index
        self.copied_block = deepcopy(base_dit_block)
        self.hidden_size = hidden_size = base_dit_block.hidden_size

        requires_grad(self.copied_block, True)
        self.copied_block.load_state_dict(base_dit_block.state_dict())
        self.copied_block.train()

        if self.block_index == 0:
            self.before_control = Linear(hidden_size, hidden_size)
            init.zeros_(self.before_control.weight)
            init.zeros_(self.before_control.bias)
        self.after_control = Linear(hidden_size, hidden_size)
        init.zeros_(self.after_control.weight)
        init.zeros_(self.after_control.bias)

    def forward(self, x, c, z):
        if self.block_index == 0:
            # the first block
            z = self.before_control(z)
            z = self.copied_block(x + z, c)
        else:
            z = self.copied_block(z, c)

        z_skip = self.after_control(z)

        return z, z_skip


class ControlDiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
            self,
            base_dit_model: DiT,
            copied_blocks_num=14,
            depth=28,
    ):
        super().__init__()

        self.base_dit_model = base_dit_model
        self.copied_blocks_num = copied_blocks_num
        self.total_blocks_num = depth

        # lock base dit model
        requires_grad(self.base_dit_model, False)

        self.control_dit_blocks = nn.ModuleList([
            ControlDiTBlock(idx, base_dit_model.blocks[idx]) for idx in range(copied_blocks_num)
        ])
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Zero-out adaLN modulation layers in ControlDiT blocks:
        for block in self.copied_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

    def __getattr__(self, name: str) -> Tensor or Module:
        # for override
        if name in ['forward', 'forward_with_cfg']:
            return self.__dict__[name]
        # for children
        elif name in ['base_dit_model', 'control_dit_blocks']:
            return super().__getattr__(name)
        # read parent
        else:
            return getattr(self.base_dit_model, name)

    def z_embedder(self, z):
        return self.x_embedder(z) + self.pos_embed

    def forward(self, x, t, y, z):
        """
        Forward pass of ControlDiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)  # (N, D)
        y = self.y_embedder(y, self.training)  # (N, D)
        c = t + y  # (N, D)
        z = self.z_embedder(z)

        # first dit block
        x = auto_grad_checkpoint(self.base_dit_model.blocks[0], x, c)
        forwarded_blocks_num = 1
        # control
        if z is not None:
            for idx in range(1, self.copied_blocks_num+1):
                z, z_skipped = auto_grad_checkpoint(self.control_dit_blocks.blocks[idx-1], x, c, z, None)
                x = auto_grad_checkpoint(self.base_dit_model.blocks[idx], x+z_skipped, c)
            forwarded_blocks_num += self.copied_blocks_num

        for idx in range(forwarded_blocks_num, self.total_blocks_num):
            x = auto_grad_checkpoint(self.base_dit_model.blocks[idx], x, c)

        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, z, cfg_scale):
        """
        Forward pass of ControlDiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half_x = x[: len(x) // 2]
        combined_x = torch.cat([half_x, half_x], dim=0)
        half_z = z[: len(z) // 2]
        combined_z = torch.cat([half_z, half_z], dim=0)
        model_out = self.forward(combined_x, t, y, combined_z)
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
    return ControlDiT(depth=28, **kwargs)


def ControlDiT_XL_4(**kwargs):
    return ControlDiT(depth=28, **kwargs)


def ControlDiT_XL_8(**kwargs):
    return ControlDiT(depth=28, **kwargs)


def ControlDiT_L_2(**kwargs):
    return ControlDiT(depth=24, **kwargs)


def ControlDiT_L_4(**kwargs):
    return ControlDiT(depth=24, **kwargs)


def ControlDiT_L_8(**kwargs):
    return ControlDiT(depth=24, **kwargs)


def ControlDiT_B_2(**kwargs):
    return ControlDiT(depth=12, **kwargs)


def ControlDiT_B_4(**kwargs):
    return ControlDiT(depth=12, **kwargs)


def ControlDiT_B_8(**kwargs):
    return ControlDiT(depth=12, **kwargs)


def ControlDiT_S_2(**kwargs):
    return ControlDiT(depth=12, **kwargs)


def ControlDiT_S_4(**kwargs):
    return ControlDiT(depth=12, **kwargs)


def ControlDiT_S_8(**kwargs):
    return ControlDiT(depth=12, **kwargs)


ControlDiT_models = {
    'DiT-XL/2': ControlDiT_XL_2, 'DiT-XL/4': ControlDiT_XL_4, 'DiT-XL/8': ControlDiT_XL_8,
    'DiT-L/2': ControlDiT_L_2, 'DiT-L/4': ControlDiT_L_4, 'DiT-L/8': ControlDiT_L_8,
    'DiT-B/2': ControlDiT_B_2, 'DiT-B/4': ControlDiT_B_4, 'DiT-B/8': ControlDiT_B_8,
    'DiT-S/2': ControlDiT_S_2, 'DiT-S/4': ControlDiT_S_4, 'DiT-S/8': ControlDiT_S_8,
}
