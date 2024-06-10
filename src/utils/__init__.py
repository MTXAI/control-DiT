import torch
from accelerate import Accelerator

cuda = torch.cuda.is_available()


def setup_accelerator(force_cpu=False):
    accelerator = Accelerator(cpu=force_cpu)
    device = accelerator.device
    return accelerator, device
