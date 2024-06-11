from src.models.DiT import DiT_models
from src.models.control_DiT_baseline_v1 import ControlDiT_models_baseline_v1
from src.models.control_DiT_baseline_v2 import ControlDiT_models_baseline_v2
# from src.models.control_DiT_with_depth import DiT_models


def create_dit_model(model_type):
    return DiT_models[model_type]


def create_control_dit_model_baseline_v1(model_type):
    return ControlDiT_models_baseline_v1[model_type]


def create_control_dit_model_baseline_v2(model_type):
    return ControlDiT_models_baseline_v2[model_type]


def create_control_dit_model(model_type):
    pass
