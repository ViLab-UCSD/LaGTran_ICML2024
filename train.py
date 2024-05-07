import argparse
import os
import yaml
import random
import shutil
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from loader import get_dataloader
from models import get_model
from optimizers import get_optimizer, get_scheduler
from UDA_trainer import get_trainer, val
from losses import get_loss
from utils import cvt2normal_state, get_logger, loop_iterable, get_pseudo_labels

from torch.utils.tensorboard import SummaryWriter

def main():

    if not torch.cuda.is_available():
        raise SystemExit('GPU is needed')

    # setup random seeds
    seed=cfg.get('seed', 1234)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # setup data loader
    splits = ['train', 'test']
    data_loader_src = get_dataloader(cfg['data']['source'], splits, cfg['training']['batch_size'])
    if cfg['training']['trainer'] == "adamatch":
        cfg['training']['batch_size'] *= 3 #cfg["training"]["target_batch_multipler"]
    data_loader_tgt = get_dataloader(cfg['data']['target'], splits, cfg['training']['batch_size'])
    batch_iterator = zip(loop_iterable(data_loader_src['train']), loop_iterable(data_loader_tgt['train']))

    n_classes = cfg["data"]["target"]["n_class"]

    # setup model (feature extractor(s) + classifier(s) + discriminator)
    n_gpu = torch.cuda.device_count()
    model_fe = get_model(cfg['model']['feature_extractor']).cuda()
    # logger.info(model_fe)
    params = [{'params': model_fe.parameters(), 'lr': 1}]
    fe_list = [model_fe]

    if cfg['model'].get('feature_extractor_2', None):
        param_dict = cfg['model']['feature_extractor_2']
        trainable = param_dict['trainable']
        param_dict.pop("trainable")
        model_fe_2 = get_model(param_dict).cuda()
        if trainable:
            params += [{'params': model_fe_2.parameters(), 'lr': 1}]
        fe_list += [model_fe_2]

    model_cls = get_model(cfg['model']['classifier']).cuda()
    params += [{'params': model_cls.parameters(), 'lr': 10}]
    cls_list = [model_cls]

    total_n_params = sum([p.numel() for p in model_fe.parameters()]) + \
                            sum([p.numel() for p in model_cls.parameters()])

    if cfg['model'].get('classifier_2', None):
        model_cls_2 = get_model(cfg['model']['classifier_2']).cuda()
        if cfg['model']['classifier_2']['trainable']:
            params += [{'params': model_cls_2.parameters(), 'lr': 10}]
        cls_list += [model_cls_2]


    d_list = []
    if cfg['model'].get('discriminator', None):
        model_d = get_model(cfg['model']['discriminator']).cuda()
        params += [{'params': model_d.parameters(), 'lr': 10}]
        d_list = [model_d]
    
    # setup loss criterion. Order and names should match in the trainer file and config file.
    loss_dict = cfg['training']['losses']
    criterion_list = []
    for loss_name, loss_params in loss_dict.items():
        criterion_list.append(get_loss(loss_params))

    # setup optimizer
    opt_main_cls, opt_main_params = get_optimizer(cfg['training']['optimizer'])
    opt = opt_main_cls(params, **opt_main_params)

    # setup scheduler
    scheduler = get_scheduler(opt, cfg['training']['scheduler'])
    trainer = get_trainer(cfg["training"])
    
    # if checkpoint already present, resume from checkpoint.
    resume_from_ckpt = False
    if os.path.exists(os.path.join(logdir, 'checkpoint.pkl')):
        cfg['training']['resume']['model'] = os.path.join(logdir, 'checkpoint.pkl')
        cfg['training']['resume']['param_only'] = False
        cfg['training']['resume']['load_cls'] = True
        resume_from_ckpt = True

    # load checkpoint
    start_it = 0
    best_acc_tgt = best_acc_src = 0
    best_acc_tgt_top5 = best_acc_src_top5 = 0
    
    if cfg['training']['resume'].get('model', None):
        resume = cfg['training']['resume']
        resume_model = resume['model']
        if os.path.isfile(resume_model):

            checkpoint = torch.load(resume_model)

            if any(substring in resume_model for substring in ["mae", "simclr", "moco", "swav", "sup"]):
                if "mae" in resume_model:
                    load_dict = checkpoint["model"]
                    load_dict = {k:v for k,v in load_dict.items() if not k.startswith("decoder")}
                if "swav" in resume_model or "sup" in resume_model:
                    load_dict = checkpoint["state_dict"]
                    load_dict = {k.partition("module.")[-1]:v for k,v in load_dict.items()}
                if "moco" in resume_model:
                    load_dict = checkpoint["state_dict"]
                    load_dict = {k.partition("base_encoder.")[-1]:v for k,v in load_dict.items()}
                if "simclr" in resume_model:
                    load_dict = checkpoint["model"]
                    load_dict = {k.partition("backbone.")[-1]:v for k,v in load_dict.items()}
                if "mask_token" in load_dict:
                    load_dict.pop("mask_token")
            else:
                load_dict = checkpoint['model_fe_state']
            try:
                ks = model_fe.load_state_dict(load_dict, strict=True)
            except:
                ks = model_fe.load_state_dict(cvt2normal_state(load_dict), strict=True)
            print(ks)
            # import pdb; pdb.set_trace()
            logger.info('Loading model from checkpoint {}'.format(resume_model))
            ## TODO: add loading additional feature extractors and classifiers
            if resume.get('load_cls', True):
                try:
                    model_cls.load_state_dict((checkpoint['model_cls_state']))
                    logger.info('Loading classifier')
                except:
                    model_cls.load_state_dict(cvt2normal_state(checkpoint['model_cls_state']))
                    logger.info('Loading classifier')
            
            if checkpoint.get('model_d_state', None):
                model_d.load_state_dict((checkpoint['model_d_state']))

            if resume['param_only'] is False:
                start_it = checkpoint['iteration']
                best_acc_tgt = checkpoint.get('best_acc_tgt', 0)
                best_acc_src = checkpoint.get('best_acc_src', 0)
                opt.load_state_dict(checkpoint['opt_main_state'])
                scheduler.load_state_dict(checkpoint['scheduler_state'])
                logger.info('Resuming training state ... ')

            logger.info("Loaded checkpoint '{}'".format(resume_model))
        else:
            logger.info("No checkpoint found at '{}'".format(resume_model))

    logger.info('Start training from iteration {}'.format(start_it))

    if n_gpu > 1:
        logger.info("Using multiple GPUs")
        # fe_list = [nn.DataParallel(mfe, device_ids=range(n_gpu)) for mfe in fe_list]
        # cls_list = [nn.DataParallel(mcls, device_ids=range(n_gpu)) for mcls in cls_list]
        model_fe = nn.DataParallel(model_fe, device_ids=range(n_gpu))
        model_cls = nn.DataParallel(model_cls, device_ids=range(n_gpu))
        if len(d_list) > 0:
            d_list = [nn.DataParallel(md, device_ids=range(n_gpu)) for md in d_list]

    if cfg['training']['pseudo_filename']:
        fname = cfg['training']['pseudo_filename']
        logger.info("Loading pseudo labels from {}".format(fname))
        pseudo = get_pseudo_labels(fname)
    else:
        pseudo = None

    for it in range(start_it, cfg['training']['iteration']):

        scheduler.step()

        ## Trainer can take multiple feature extractors (eg: FixMatch), and multiple classifiers (eg: MCD)
        if pseudo:
            trainer(batch_iterator, model_fe, model_cls, *d_list, opt, it, *criterion_list,
                    cfg, logger, writer, pseudo)
        else:
            trainer(batch_iterator, model_fe, model_cls, *d_list, opt, it, *criterion_list,
                    cfg, logger, writer)

        # trainer(
        #     batch_iterator,
        #     model_fe,
        #     None,
        #     model_cls,
        #     None,
        #     model_d,
        #     None,
        #     opt_main,
        #     it,
        #     criterion_cls,
        #     criterion_d,
        #     cfg, logger, writer
        # )

        if (it + 1) % cfg['training']['val_interval'] == 0:
                
            with torch.no_grad():
                if "feat" in cfg['training']['trainer']:
                    acc_src, acc_src_top5 = val(data_loader_src['test'], None, model_cls, it, n_classes, "source", logger, writer)
                    acc_tgt, acc_tgt_top5 = val(data_loader_tgt['test'], None, model_cls, it, n_classes, "target", logger, writer)
                else:
                    acc_src, acc_src_top5 = val(data_loader_src['test'], model_fe, model_cls, it, n_classes, "source", logger, writer)
                    acc_tgt, acc_tgt_top5 = val(data_loader_tgt['test'], model_fe, model_cls, it, n_classes, "target", logger, writer)
                is_best = False
                if acc_tgt > best_acc_tgt:
                    is_best = True
                    best_acc_tgt = acc_tgt
                    best_acc_src = acc_src
                    best_acc_tgt_top5 = acc_tgt_top5
                    best_acc_src_top5 = acc_src_top5
                    with open(os.path.join(logdir, 'best_acc.txt'), "a") as fh:
                        write_str = "Source Top 1\t{src_top1:.3f}\tSource Top 5\t{src_top5:.3f}\tTarget Top 1\t{tgt_top1:.3f}\tTarget Top 5\t{tgt_top5:.3f}\n".format(src_top1=best_acc_src, src_top5=best_acc_src_top5, tgt_top1=best_acc_tgt, tgt_top5=best_acc_tgt_top5)
                        fh.write(write_str)
                # if acc_src > best_acc_src:
                #     best_acc_src = acc_src
                print_str = '[Val] Iteration {it}\tBest Acc source. {acc_src:.3f}\tBest Acc target. {acc_tgt:.3f}'.format(it=it+1, acc_src=best_acc_src, acc_tgt=best_acc_tgt)
                logger.info(print_str)

        # if (it + 1) % cfg['training']['save_interval'] == 0:
            ## TODO: add saving additional feature extractors and classifiers
            state = {
                'iteration': it + 1,
                'model_fe_state': model_fe.state_dict(),
                'model_cls_state': model_cls.state_dict(),
                'opt_main_state': opt.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_acc_tgt' : best_acc_tgt,
                'best_acc_src' : best_acc_src
            }

            if len(d_list):
                state['model_d_state'] = model_d.state_dict()
            
            ckpt_path = os.path.join(logdir, 'checkpoint.pkl')
            save_path = ckpt_path#.format(it=it+1)
#             last_path = ckpt_path.format(it=it+1-cfg['training']['save_interval'])
            torch.save(state, save_path)
#             if os.path.isfile(last_path):
#                 os.remove(last_path)

            if is_best:
                best_path = os.path.join(logdir, 'best_model.pkl')
                torch.save(state, best_path)
            logger.info('[Checkpoint]: {} saved'.format(save_path))



if __name__ == '__main__':
    global cfg, args, writer, logger, logdir
    valid_trainers = ["plain", "wsda", "feat_plain", "feat_wsda"]
    backbone_choices = ["resnet50", "vits16", "vitb16", "vitl16", "timm_convnext", "swinb16", "timesformerb_8f", "deitb16"]
    backbone_choices += ["dinov2_vits", "dinov2_vitb"]

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
    parser.add_argument("--json_file" , help="Path for annotation json file")
    parser.add_argument("--lr_rate" , help="Learning Rate", default=0.003, type=float)
    parser.add_argument("--num_class", type=int, help="Number of classes")
    parser.add_argument("--data_root", type=str, help="Data root")
    parser.add_argument("--cbs_source", type=int, default=0, choices=[0,1], help="Class balancing in source")
    parser.add_argument("--cbs_target", type=int, default=0, choices=[0,1], help="Class balancing in target")
    parser.add_argument("--trainer", required=True, type=str.lower, choices=valid_trainers, help="Adaptation method.")
    parser.add_argument("--num_iter", type=int, default=100004, help="Total number of iterations")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--resume", help="Resume training from checkpoint")
    parser.add_argument("--exp_name", help="experiment name")
    parser.add_argument("--backbone", help="backbone network", choices=backbone_choices, default="resnet50")
    parser.add_argument("--linear", help="Linear Probing of SSL Models", type=int, default=0)
    parser.add_argument("--val_freq", help="Validation Frequency", type=int, default=5000)
    parser.add_argument("--pl_source", help="Source for the pseudolabels", type=str)
    parser.add_argument("--hard_labels", type=int, default=1, choices=[0,1], help="Use hard labels for target training")
    parser.add_argument("--use_target", help="Use target labels for training", type=int, default=0)
    parser.add_argument("--frac_text", help="Fraction of text available.", type=float, default=1.)
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

    ## overwrite config parameters
    n_class = args.num_class
    cfg["model"]["classifier"]["n_class"] = n_class
    cfg["data"]["source"]["n_class"] = n_class
    cfg["data"]["target"]["n_class"] = n_class

    cfg["training"]["val_interval"] = args.val_freq
    if args.pl_source:
        if "wsda" not in args.trainer:
            logger.info("PL Training not allowed with trainer {}. Use --trainer=wsda".format(args.trainer))
        dataset = args.json_file.split("/")[-1].split(".")[0]
        if args.frac_text is not None:
            cfg["training"]["pseudo_filename"] = os.path.join("hard_labels", "{}_{}_{}_{}PL.txt".format(dataset, args.source, args.target, args.pl_source, str(args.frac_text).replace(".","-")))
        else:
            cfg["training"]["pseudo_filename"] = os.path.join("hard_labels", "{}_{}_{}_{}PL.txt".format(dataset, args.source, args.target, args.pl_source))
        cfg["training"]["losses"]["loss_tgt"]["top_pred"] = args.hard_labels
    else:
        cfg["training"]["pseudo_filename"] = None
        # cfg["training"]["losses"]["loss_tgt"] = None #["top_pred"] = False

    if args.trainer == "adamatch":
        cfg["data"]["source"]["dual_aug"] = True
        cfg["data"]["target"]["dual_aug"] = True
    else:
        cfg["data"]["source"]["dual_aug"] = False
        cfg["data"]["target"]["dual_aug"] = False

    cfg['pt_model'] = args.pt_model
    if args.resume:
        cfg["training"]["resume"]["model"] = args.resume
    else:
        if args.pt_model is not None and args.pt_dataset is not None:
            pt_path = os.path.join("pt_models", args.pt_model + "_" + args.backbone + "_" + args.pt_dataset) + ".pth"
            if os.path.exists(pt_path):
                cfg["training"]["resume"]["model"] = pt_path
        else:
            cfg["training"]["resume"]["model"] = None
    
    cfg["training"]["use_target"] = args.use_target

    # cfg['pt_model'] = args.pt_model
    # if args.resume:
    #     cfg["training"]["resume"]["model"] = args.resume
    # else:
    #     if cfg["pt_model"] == "moco":
    #         if args.backbone == "resnet50":
    #             cfg["training"]["resume"]["model"] = "SSL_models/r-50-1000ep.pth.tar"
    #         if args.backbone == "vits16":
    #             cfg["training"]["resume"]["model"] = "SSL_models/vit-s-300ep.pth.tar"
    #         if args.backbone == "vitb16":
    #             cfg["training"]["resume"]["model"] = "SSL_models/vit-b-300ep.pth.tar"
    #     elif cfg["pt_model"] == "swav":
    #         cfg["training"]["resume"]["model"] = "SSL_models/swav_800ep_pretrain.pth.tar"
    #     elif cfg["pt_model"] == "mae":
    #         if args.backbone == "vitb16":
    #             cfg["training"]["resume"]["model"] = "SSL_models/mae_pretrain_vit_base.pth"
    #         elif args.backbone == "vitl16":
    #             cfg["training"]["resume"]["model"] = "SSL_models/mae_pretrain_vit_large.pth"
    #     elif cfg["pt_model"] == "dino":
    #         if args.backbone == "resnet50":
    #             cfg["training"]["resume"]["model"] = "SSL_models/dino_resnet50_pretrain.pth"
    #         elif args.backbone == "vits16":
    #             cfg["training"]["resume"]["model"] = "SSL_models/dino_deitsmall16_pretrain.pth"
    #         elif args.backbone == "vitl16":
    #             cfg["training"]["resume"]["model"] = "SSL_models/dino_vitlarge16_pretrain.pth"
    #         elif args.backbone == "vitb16":
    #             cfg["training"]["resume"]["model"] = "SSL_models/dino_vitbase16_pretrain.pth"
    #     elif cfg["pt_model"] == "swag":
    #         if args.backbone == "vitb16":
    #             args.backbone = "vitb16_swag"
    #             cfg["training"]["resume"]["model"] = "SSL_models/swag_vit_b16.torch"
    #         elif args.backbone == "vitl16":
    #             args.backbone = "vitl16_swag"
    #             cfg["training"]["resume"]["model"] = "SSL_models/swag_vit_l16.torch"
    #     elif cfg["pt_model"] == "deit":
    #         if args.backbone == "vits16":
    #             cfg["training"]["resume"]["model"] = "SSL_models/deit_small_patch16_224-cd65a155.pth"
    #         if args.backbone == "vitb16":
    #             cfg["training"]["resume"]["model"] = "SSL_models/deit_base_patch16_224-b5f2ef4d.pth"
    #     else:
    #         cfg["training"]["resume"]["model"] = None

    cfg["training"]["trainer"] = args.trainer

    if args.lr_rate:
        cfg['training']['scheduler']['init_lr'] = args.lr_rate

    if args.trainer in ["cdan", "mcc", "memsac"]:
        cfg["model"]["discriminator"]["in_feature"] *= n_class ## for cdan
    elif args.trainer in ["hda", "toalign"]:
        cfg["model"]["discriminator"]["in_feature"] = n_class ## for hdan
    cfg["training"]["freeze_encoder"] = args.linear
        

    cfg["data"]["source"]["domain"] = args.source
    cfg["data"]["target"]["domain"] = args.target
    cfg["data"]["target"]["json_file"] = cfg["data"]["source"]["json_file"] = args.json_file
    cfg["data"]["target"]["data_root"] = cfg["data"]["source"]["data_root"] = args.data_root
    # cfg["data"]["source"]["train"] = cfg["data"]["source"]["val"] = args.source
    # cfg["data"]["target"]["train"] = cfg["data"]["target"]["val"] = args.target

    if args.pt_model:
        if "clip" in args.pt_model:
            args.backbone += "_clip"
        if "siglip" in args.pt_model:
            args.backbone += "_siglip"
        if "_dino" in args.pt_model:
            args.backbone += "_dino"

    if args.cbs_source:
        cfg["data"]["source"]["sampler"] = {"name" : "class_balanced"}
    else:
        cfg["data"]["source"]["sampler"] = {"name" : "random"}

    if args.cbs_target:
        cfg["data"]["target"]["sampler"] = {"name" : "class_balanced"}
    else:
        cfg["data"]["target"]["sampler"] = {"name" : "random"}

    cfg['training']['batch_size'] = args.batch_size
    cfg["model"]["feature_extractor"]["arch"] = args.backbone

    if args.backbone in ["timm_swin", "swinb16"]: ## different crop sizes only for swin
        cfg["data"]["source"]["crop_size"] = cfg["data"]["target"]["crop_size"] = 224
    else:
        cfg["data"]["source"]["crop_size"] = cfg["data"]["target"]["crop_size"] = 224

    # if args.trainer == "mdd":
    #     cfg["model"]["classifier"]["feat_size"] = [2048, 2048, 2048]
    # else:
    if args.backbone == "resnet50_clip":
        cfg["model"]["classifier"]["feat_size"] = [1024,256]
    # elif args.backbone == "vitb16_clip":
    #     cfg["model"]["classifier"]["feat_size"] = [512,256]
    elif args.backbone == "vits16":
        cfg["model"]["classifier"]["feat_size"] = [384,256]
    elif "resnet50" in args.backbone:
        cfg["model"]["classifier"]["feat_size"] = [2048,256]
    # elif args.backbone.startswith(("vitb16")):
    #     cfg["model"]["classifier"]["feat_size"] = [768,256]
    elif args.backbone.startswith(("vitl16")):
        cfg["model"]["classifier"]["feat_size"] = [1024,256]
    elif args.backbone in ["timm_swin", "timm_convnext"]:
        cfg["model"]["classifier"]["feat_size"] = [768,256]
    elif args.backbone in ["timm_deit", "timm_resmlp"]:
        cfg["model"]["classifier"]["feat_size"] = [384,256]
    elif args.backbone.startswith("dinov2_vits"):
        cfg["model"]["classifier"]["feat_size"] = [384,256]
    elif "vitb" in args.backbone or "deitb" in args.backbone:
        cfg["model"]["classifier"]["feat_size"] = [768,256]
    elif "swinb16" in args.backbone:
        cfg["model"]["classifier"]["feat_size"] = [1024,256]

    if args.trainer == "mdd":
        d1, d2 = cfg["model"]["classifier"]["feat_size"]
        cfg["model"]["classifier"]["feat_size"] = [d1,256,256]
    elif "feat" in args.trainer:
        cfg["model"]["classifier"]["feat_size"] = [1536,256,256,256]

    ## Method specific parameters
    if cfg["training"]["trainer"] == "memsac":
        cfg["training"]["losses"]["loss_msc"]["dim"] = cfg["model"]["classifier"]["feat_size"][-1]

    if cfg['training']['trainer'] == "llr_wsda":
        # assert args.resume is not None, "Need to resume from checkpoint for LLR-WSDA"
        cfg['training']['resume']['load_cls'] = True

    logger.info(args)

    main()

