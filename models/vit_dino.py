import torch

__all__ = [
    "dinov2_vits14",
    "dinov2_vitb14",
]

def dinov2_vits14(pretrained=False, **kwargs):
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=pretrained, **kwargs)
    return model

def dinov2_vitb14(pretrained=False, **kwargs):
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14', pretrained=pretrained, **kwargs)
    return model