from src.models.DiT import DiT_models
# from src.models.control_DiT_with_depth import DiT_models


def create_dit_model(model_type):
    return DiT_models[model_type]

def create_ControlDiT_model(model_type):
    pass
