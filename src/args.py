import argparse
import os
import yaml
import shutil

from utils import get_logger

from torch.utils.tensorboard import SummaryWriter

def get_args():
    parser = argparse.ArgumentParser(description='config')
    parser.add_argument(
        '--config',
        nargs='?',
        type=str,
        default='configs/default.yml',
        help='Configuration file to use',
    )

    parser.add_argument("--seed", type=int, default=1234, help="Fix random seed")
    parser.add_argument("--source" , help="Source domain")
    parser.add_argument("--target" , help="Target domain")
    parser.add_argument("--dataset" , help="Dataset for adaptation")
    parser.add_argument("--lr_rate" , help="Learning Rate", default=0.003, type=float)
    parser.add_argument("--data_root", type=str, help="Data root")
    parser.add_argument("--cbs_source", type=int, default=1, choices=[0,1], help="Class balanced sampling in source")
    parser.add_argument("--trainer", required=True, type=str.lower, help="Adaptation method.")
    parser.add_argument("--num_iter", type=int, default=65001, help="Total number of iterations")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--resume", help="Resume training from checkpoint")
    parser.add_argument("--exp_name", help="experiment name")
    parser.add_argument("--backbone", help="backbone network", default="vitb16")
    parser.add_argument("--linear", help="Linear Probing of SSL Models", type=int, default=0)
    parser.add_argument("--val_freq", help="Validation Frequency", type=int, default=5000)
    parser.add_argument("--use_target", help="Use target labels for training", type=int, default=0)
    parser.add_argument("--timesformer_pt", help="Pretrained path for timesformer", default="pretrained_model/TimeSformer_divST_8x32_224_K600.pyth")

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=yaml.SafeLoader)

    cfg["training"]["iteration"] = args.num_iter
    cfg["exp"] = args.exp_name

    if args.backbone == "timesformerb_8f":
        cfg["model"]["feature_extractor"]["pretrained_model"] = args.timesformer_pt

    logdir = os.path.join('runs', os.path.basename(args.config)[:-4], cfg['exp'])
    if not os.path.exists(logdir):
        os.makedirs(logdir, exist_ok=True)
    writer = SummaryWriter(log_dir=logdir)

    print('RUNDIR: {}'.format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info('Start logging')

    ## Seed
    cfg["seed"] = args.seed

    dataset = args.dataset
    json_file = "metadata/{}.json".format(dataset.lower())

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

    cfg["training"]["val_interval"] = args.val_freq
    if "lagtran" in args.trainer:
        cfg["training"]["pseudo_filename"] = os.path.join("pseudo_labels", "{}_{}_{}.txt".format(dataset.lower(), args.source, args.target))

    if args.resume:
        cfg["training"]["resume"]["model"] = args.resume
    else:
        cfg["training"]["resume"]["model"] = None

    cfg["training"]["use_target"] = args.use_target ## for computing the target-supervised upper-bound

    cfg["training"]["trainer"] = args.trainer

    if args.lr_rate:
        cfg['training']['scheduler']['init_lr'] = args.lr_rate

    cfg["training"]["freeze_encoder"] = args.linear

    cfg["data"]["source"]["domain"] = args.source
    cfg["data"]["target"]["domain"] = args.target
    cfg["data"]["target"]["json_file"] = cfg["data"]["source"]["json_file"] = json_file
    cfg["data"]["target"]["data_root"] = cfg["data"]["source"]["data_root"] = args.data_root

    if args.cbs_source:
        cfg["data"]["source"]["sampler"] = {"name" : "class_balanced"}

    cfg['training']['batch_size'] = args.batch_size
    cfg["model"]["feature_extractor"]["arch"] = args.backbone

    return cfg, writer, logger, logdir

