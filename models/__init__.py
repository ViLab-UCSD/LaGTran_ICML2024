### __init__.py
# Get model instance with designated parameters.
# Author: Gina Wu @ 01/22
###

import copy
import torch.nn as nn
import logging

## timm models
from .models_timm import timm_convnext, timm_deit, timm_resmlp, timm_resnet50, timm_swin

from .resnet import resnet10, resnet101, resnet18, resnet50
from .vit_timm import vit_b_16, vit_s_16, vit_l_16, swin_b, deit_b
from .vit_swag import ViTB16_swag, ViTL16_swag
from .vit_dino import dinov2_vits14, dinov2_vitb14
from .open_clip import vitb16_clip, rn50_clip, vitb16_siglip, vitl16_clip
from .linearcls import linearcls
from .lenet import lenet
from .mlpcls import mlpcls
from .fscls import fscls
from .advnet import advnet
from .mlpmdd import mddcls
from .mlpmcd import mcdcls
from .mlphda import hdacls
from .randomlyr import randomlayer
from .utils import grl_hook
from .memory_bank import MemoryModule

## Videos
from .TimeSformer import TimeSformer_Base


logger = logging.getLogger('mylogger')


def get_model(model_dict, verbose=False):

    name = model_dict['arch']
    model = _get_model_instance(name)
    param_dict = copy.deepcopy(model_dict)
    param_dict.pop('arch')

    if "timesformer" in name:
        model = model(**param_dict)
        model.head = nn.Identity()
    elif "timm" in name:
        if "resnet" in name:
            model = model(**param_dict)
            model.fc = nn.Identity() 
        elif "deit" in name or "swin" in name or "resmlp" in name:
            model = model(**param_dict)
            model.head = nn.Identity() 
        elif "next" in name:
            model = model(**param_dict)
            model.head.fc = nn.Identity() 
    elif "clip" in name:
        model = model(**param_dict)
    elif "swin" in name:
        model = model(**param_dict)
        model.head.fc = nn.Identity()
    else:
        if 'resnet' in name:
            model = model(**param_dict)
            model.fc = nn.Identity()
        elif 'vit' in name:
            model = model(**param_dict)
            model.head = nn.Identity()
        else:
            model = model(**param_dict)

    if verbose:
        logger.info(model)

    return model

def _get_model_instance(name):
    try:
        return {
            'resnet10': resnet10,
            'resnet18': resnet18,
            'resnet50': resnet50,
            'resnet101': resnet101,
            'vitb16'   : vit_b_16,
            'vits16'   : vit_s_16,
            "vitl16"   : vit_l_16,
            "vitb16_swag" : ViTB16_swag,
            "vitl16_swag" : ViTL16_swag,
            'linearcls': linearcls,
            'mlpcls': mlpcls,
            'advnet': advnet,
            'randomlyr': randomlayer,
            'lenet': lenet,
            'fscls': fscls,
            'mddcls': mddcls,
            'mcdcls': mcdcls,
            'hdacls'   : hdacls,
            "timm_resnet50" : timm_resnet50,
            "timm_swin"     : timm_swin,
            "timm_convnext" : timm_convnext,
            "timm_resmlp"   : timm_resmlp,
            "timm_deit"     : timm_deit,
            "vits14_dino" : dinov2_vits14,
            "vitb16_dino" : dinov2_vitb14,
            "vitb16_clip" : vitb16_clip,
            "vitl16_clip" : vitl16_clip,
            "resnet50_clip" : rn50_clip,
            "vitb16_siglip" : vitb16_siglip,
            "swinb16"     : swin_b,
            "deitb16"     : deit_b,
            "timesformerb_8f" : TimeSformer_Base,
        }[name]
    except:
        raise BaseException('Model {} not available'.format(name))


