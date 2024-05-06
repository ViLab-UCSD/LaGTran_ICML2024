import timm

model_names = [
    "vit_small_patch16_224",
    "vit_base_patch16_224",
    "vit_large_patch16_224"
]

def create_model(model_name, pretrained=True):
    
    model = timm.create_model(model_name, pretrained=pretrained)
    params = sum([p.numel() for p in model.parameters() if p.requires_grad==True])
    params /= 1e6
    # print("Number of parameters in {}: {:.02f}M".format(model_name, params))
    return model, params


def vit_b_16(pretrained=False, **kwargs):
    model, _ = create_model("vit_base_patch16_224", pretrained=pretrained)
    return model

def vit_s_16(pretrained=False, **kwargs):
    model, _ = create_model("vit_small_patch16_224", pretrained=pretrained)
    return model

def vit_l_16(pretrained=False, **kwargs):
    model, _ = create_model("vit_large_patch16_224", pretrained=pretrained)
    return model

def swin_b(pretrained=False, **kwargs):
    model, _ = create_model("swin_base_patch4_window7_224", pretrained=pretrained)
    return model

def deit_b(pretrained=False, **kwargs):
    model = timm.create_model("deit3_base_patch16_224.fb_in22k_ft_in1k", pretrained=pretrained, num_classes=0)
    return model