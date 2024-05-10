import argparse
import os
import yaml
import torch
from torch import nn

from loader import get_dataloader
from models import get_model
from utils import get_logger, cvt2normal_state
from trainer import eval

def main(cfg, logger):
    if not torch.cuda.is_available():
        raise SystemExit('GPU is needed')

    # if "timesformer" in cfg['model']['feature_extractor']['arch']:
    #     cfg['data']['target']['loader'] = "VideoLoader"
    #     # cfg['data']['target']['dataset'] = "ego_exoDA"
    data_loader_tgt = get_dataloader(cfg['data']['target'], ["test"], cfg['testing']['batch_size'])

    n_classes = cfg["model"]["classifier"]["n_class"]
    write_file = cfg['testing']['write_file']

    # setup model (feature extractor + classifier + discriminator)
    n_gpu = torch.cuda.device_count()
    model_fe = get_model(cfg['model']['feature_extractor'], verbose=False).cuda()
    model_cls = get_model(cfg['model']['classifier'], verbose=False).cuda()

    if cfg['testing']['resume'].get('model', None):
        resume = cfg['testing']['resume']
        resume_model = resume['model']
        if os.path.isfile(resume_model):
            logger.info('Loading model from checkpoint {}'.format(resume_model))

            checkpoint = torch.load(resume_model)
            try:
                model_fe.load_state_dict((checkpoint['model_fe_state']))
                model_cls.load_state_dict((checkpoint['model_cls_state']))
            except:
                model_fe.load_state_dict(cvt2normal_state(checkpoint['model_fe_state']))
                model_cls.load_state_dict(cvt2normal_state(checkpoint['model_cls_state']))
                
            logger.info('Loading feature extractor and classifier')
            logger.info("Loaded checkpoint '{}'".format(resume_model))
        else:
            logger.info("No checkpoint found at '{}'".format(resume_model))
            # sys.exit(0)

    if n_gpu>1:
        model_fe = nn.DataParallel(model_fe, device_ids=range(n_gpu))
        model_cls = nn.DataParallel(model_cls, device_ids=range(n_gpu))

    eval(data_loader_tgt['test'], model_fe, model_cls, n_classes, write_file, cfg, logger)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='config')
    parser.add_argument(
        '--config',
        nargs='?',
        type=str,
        default='configs/default.yml',
        help='Configuration file to use',
    )

    # parser.add_argument("--source" , help="Source domain")
    parser.add_argument("--target" , help="Target domain")
    parser.add_argument("--dataset" , help="Dataset Name")
    parser.add_argument("--norm", type=int, default=0, help="Normalize features [0/1]")
    parser.add_argument("--data_root", type=str, help="Data root")
    parser.add_argument("--saved_model", help="Resume training from checkpoint")
    parser.add_argument("--write_file", help="write classwise accuracy to a file.")
    parser.add_argument("--backbone", help="backbone network", default="vitb16")
    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=yaml.SafeLoader)

    dataset = args.dataset
    if dataset == "GeoImnet":
        n_class = 600
    elif dataset == "GeoPlaces":
        n_class = 205
    elif dataset == "DomainNet":
        n_class = 345
    elif dataset == "Ego2Exo":
        n_class = 24
    else:
        raise ValueError("Unknown dataset.")

    ## overwrite config parameters
    cfg["model"]["classifier"]["n_class"] = n_class
    if args.norm:
        cfg["model"]["classifier"]["norm"] = args.norm
    # cfg["data"]["target"]["n_class"] = n_class

    # cfg["data"]["target"]["val"] = args.target

    cfg["data"]["target"]["domain"] = args.target
    cfg["data"]["target"]["json_file"] = "metadata/{}.json".format(dataset.lower())
    cfg["data"]["target"]["data_root"] = args.data_root

    cfg["testing"]["resume"]["model"] = args.saved_model
    cfg['testing']['write_file'] = args.write_file

    if args.backbone:
        cfg["model"]["feature_extractor"]["arch"] = args.backbone

    logdir = "test/"

    print('RUNDIR: {}'.format(logdir))

    logger = get_logger("runs/")
    logger.info('Start logging')

    logger.info(args)

    cfg['config'] = args.config

    main(cfg, logger)


