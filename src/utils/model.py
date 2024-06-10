import os
from collections import OrderedDict

from src.utils import *


def load_depth_model(model_type="DPT_Large", model_repo_or_path="intel-isl/MiDaS", source="github", device="cuda"):
    # model_type = "DPT_Large"  # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
    # model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
    # model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
    midas = torch.hub.load(repo_or_dir=model_repo_or_path, source=source, model=model_type)
    midas.to(device)
    midas.eval()

    # https://github.com/isl-org/MiDaS#Accuracy
    # https://pytorch.org/hub/intelisl_midas_v2/
    midas_transforms = torch.hub.load(repo_or_dir=model_repo_or_path, source=source, model="transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    return midas, transform


def load_pretrained_dit_model(model_ckpt):
    assert os.path.isfile(model_ckpt), f'Could not find DiT checkpoint at {model_ckpt}'
    checkpoint = torch.load(model_ckpt, map_location=lambda storage, loc: storage)
    if "ema" in checkpoint:  # supports checkpoints from train.py
        checkpoint = checkpoint["ema"]
    return checkpoint


none_model = 'None'


def load_model(model_ckpt=none_model):
    if model_ckpt is None or model_ckpt == "" or model_ckpt == none_model:
        return None
    assert os.path.isfile(model_ckpt), f'Could not find Model checkpoint at {model_ckpt}'
    checkpoint = torch.load(model_ckpt, map_location=lambda storage, loc: storage)
    return checkpoint


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        name = name.replace("module.", "")
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag
