from .hf_model import HF_MODEL_URLS, HFModel
from .openai_model import OPENAI_MODELS, OpenAIModel
from src.tools import get_default_device

def get_model(model_name, gpu_id=0):
    """Load / intialise the model and return it"""
    if model_name in OPENAI_MODELS.keys():
        model = OpenAIModel(model_name)
    elif model_name in HF_MODEL_URLS.keys():
        model = HFModel(
            device=get_default_device(gpu_id),
            model_name=model_name,
        )
    else:
        raise ValueError(
            f"Unknown model name {model_name}"
            f"Expected model names: {OPENAI_MODELS.keys()} or {HF_MODEL_URLS.keys()}"
        )
    return model