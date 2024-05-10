import copy
import torch.nn as nn
import logging

from .vit_timm import vit_b_16, vit_s_16, vit_l_16
from .open_clip import vitb16_clip, rn50_clip, vitb16_siglip, vitl16_clip
from .linearcls import linearcls
from .mlpcls import mlpcls

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
    elif "clip" in name:
        model = model(**param_dict)
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
            'vitb16'   : vit_b_16,
            'vits16'   : vit_s_16,
            "vitl16"   : vit_l_16,
            'linearcls': linearcls,
            'mlpcls': mlpcls,
            "vitb16_clip" : vitb16_clip,
            "vitl16_clip" : vitl16_clip,
            "resnet50_clip" : rn50_clip,
            "vitb16_siglip" : vitb16_siglip,
            "timesformerb_8f" : TimeSformer_Base,
        }[name]
    except:
        raise BaseException('Model {} not available'.format(name))


