from .plain import train_plain
from .feat_plain import train_plain_feat
from .feat_lagtran import train_lagtran_feat
from .lagtran import train_lagtran

def get_trainer(cfg):
    trainer = _get_trainer_instance(cfg['trainer'])
    return trainer

def _get_trainer_instance(name):
    try:
        return {
            'plain' : train_plain,
            'lagtran'     : train_lagtran,
            'feat_plain' : train_plain_feat,
            'feat_lagtran' : train_lagtran_feat,
        }[name]
    except:
        raise BaseException('Trainer type {} not available'.format(name))


