import timm
import torch


def load_model(model_path: str):
    model = timm.create_model('resnet50d.ra4_e3600_r224_in1k', pretrained=False)
    
    model.load_state_dict(torch.load(model_path))
    
    model.eval()

    return model
