import torch.nn.functional as F
import torch.nn as nn
import timm
# import clip

class CLIPModel(nn.Module):

    def __init__(self, backbone, pretrained, **kwargs):
        super().__init__()
        # self.model, _ = open_clip.create_model_from_pretrained(backbone, pretrained=pretrained, jit=False)
        # self.model = self.model.visual

        if backbone == "RN50":
            self.model, _ = clip.load("RN50", device="cuda", jit=False)
            self.model = self.model.visual
            self.model = self.model.float()
        else:
            self.model = timm.create_model(backbone, pretrained=pretrained, num_classes=0, no_jit=True)
    
    def forward(self, input):
        # import pdb; pdb.set_trace()
        features = self.model(input)
        ## normalize features
        # features = F.normalize(features, dim=-1)
        return features
    
def vitb16_clip(pretrained=False, **kwargs):
    # return CLIPModel('ViT-B-16', pretrained="laion2b_s34b_b88k")
    return CLIPModel('vit_base_patch16_clip_224.laion2b', pretrained=True)

def vitl16_clip(pretrained=False, **kwargs):
    # return CLIPModel('ViT-B-16', pretrained="laion2b_s34b_b88k")
    return CLIPModel('vit_large_patch14_clip_224.laion2b', pretrained=True)

def rn50_clip(pretrained=False, **kwargs):
    return CLIPModel('RN50', pretrained=True)

def vitb16_siglip(pretrained=False, **kwargs):
    # return CLIPModel('ViT-B-16-SigLIP', pretrained="webli")
    return CLIPModel('vit_base_patch16_siglip_224.webli', pretrained=True)

