import torch
from torch import nn
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp

from models.DiT import get_2d_sincos_pos_embed, FinalLayer, DiTBlock, TimestepEmbedder, LabelEmbedder, DiT


class ControlDiT(nn.Module):
    def __init__(
            self,
            dit_model: DiT = None,
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
        self.dit_model = dit_model
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.z_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        num_patches = self.z_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
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

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.z_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.z_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)
        nn.init.constant_(self.z_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
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

    def forward(self, x, z, t, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        z: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        z = x+z
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        z = self.z_embedder(z) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)  # (N, D)
        y = self.y_embedder(y, self.training)  # (N, D)
        c = t + y  # (N, D)
        for i in range(len(self.blocks)):
            dit_block = self.dit_model.blocks[i]
            block = self.blocks[i]
            x = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(dit_block), x, c)  # (N, T, D)
            z = torch.utils.checkpoint.checkpoint(self.ckpt_wrapper(block), z, c)  # (N, T, D)
            z = x+z
        z = self.final_layer(z, c)  # (N, T, patch_size ** 2 * out_channels)
        z = self.unpatchify(z)  # (N, out_channels, H, W)
        return z


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def control_DiT_XL_2(**kwargs):
    return ControlDiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def control_DiT_XL_4(**kwargs):
    return ControlDiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)


def control_DiT_XL_8(**kwargs):
    return ControlDiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)


def control_DiT_L_2(**kwargs):
    return ControlDiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def control_DiT_L_4(**kwargs):
    return ControlDiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)


def control_DiT_L_8(**kwargs):
    return ControlDiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)


def control_DiT_B_2(**kwargs):
    return ControlDiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


def control_DiT_B_4(**kwargs):
    return ControlDiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)


def control_DiT_B_8(**kwargs):
    return ControlDiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)


def control_DiT_S_2(**kwargs):
    return ControlDiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


def control_DiT_S_4(**kwargs):
    return ControlDiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)


def control_DiT_S_8(**kwargs):
    return ControlDiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


control_DiT_models = {
    'DiT-XL/2': control_DiT_XL_2, 'DiT-XL/4': control_DiT_XL_4, 'DiT-XL/8': control_DiT_XL_8,
    'DiT-L/2': control_DiT_L_2, 'DiT-L/4': control_DiT_L_4, 'DiT-L/8': control_DiT_L_8,
    'DiT-B/2': control_DiT_B_2, 'DiT-B/4': control_DiT_B_4, 'DiT-B/8': control_DiT_B_8,
    'DiT-S/2': control_DiT_S_2, 'DiT-S/4': control_DiT_S_4, 'DiT-S/8': control_DiT_S_8,
}
