import os
import random

import numpy as np
import torch
from torch import nn

from loader import get_dataloader
from models import get_model
from optimizers import get_optimizer, get_scheduler
from trainer import get_trainer, val
from losses import get_loss
from utils import cvt2normal_state, loop_iterable, get_pseudo_labels
from args import get_args

def main(cfg, writer, logger, logdir):

    if not torch.cuda.is_available():
        raise SystemExit('GPU is needed')

    # setup data loader
    splits = ['train', 'test']
    data_loader_src = get_dataloader(cfg['data']['source'], splits, cfg['training']['batch_size'])
    data_loader_tgt = get_dataloader(cfg['data']['target'], splits, cfg['training']['batch_size'])
    batch_iterator = zip(loop_iterable(data_loader_src['train']), loop_iterable(data_loader_tgt['train']))

    n_classes = cfg["model"]["classifier"]["n_class"]

    # setup model (feature extractor(s) + classifier(s))
    n_gpu = torch.cuda.device_count()
    model_fe = get_model(cfg['model']['feature_extractor']).cuda()
    params = [{'params': model_fe.parameters(), 'lr': 1}]

    model_cls = get_model(cfg['model']['classifier']).cuda()
    params += [{'params': model_cls.parameters(), 'lr': 10}]
    
    # setup loss criterion. Order and names should match in the trainer file and config file.
    loss_dict = cfg['training']['losses']
    loss_criterion = get_loss(loss_dict['loss_cls'])

    # setup optimizer
    opt_main_cls, opt_main_params = get_optimizer(cfg['training']['optimizer'])
    opt = opt_main_cls(params, **opt_main_params)

    # setup scheduler
    scheduler = get_scheduler(opt, cfg['training']['scheduler'])
    trainer = get_trainer(cfg["training"])
    
    # if checkpoint already present, resume from checkpoint.
    if os.path.exists(os.path.join(logdir, 'checkpoint.pkl')):
        cfg['training']['resume']['model'] = os.path.join(logdir, 'checkpoint.pkl')
        cfg['training']['resume']['param_only'] = False
        cfg['training']['resume']['load_cls'] = True

    # load checkpoint
    start_it = 0
    best_acc_tgt = best_acc_src = 0
    best_acc_tgt_top5 = best_acc_src_top5 = 0
    
    if cfg['training']['resume'].get('model', None):
        resume = cfg['training']['resume']
        resume_model = resume['model']
        if os.path.isfile(resume_model):

            checkpoint = torch.load(resume_model)

            load_dict = checkpoint['model_fe_state']
            try:
                ks = model_fe.load_state_dict(load_dict, strict=True)
            except:
                ks = model_fe.load_state_dict(cvt2normal_state(load_dict), strict=True)
            
            logger.info('Loading model from checkpoint {}'.format(resume_model))
            
            if resume.get('load_cls', True):
                try:
                    model_cls.load_state_dict((checkpoint['model_cls_state']))
                    logger.info('Loading classifier')
                except:
                    model_cls.load_state_dict(cvt2normal_state(checkpoint['model_cls_state']))
                    logger.info('Loading classifier')

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
        model_fe = nn.DataParallel(model_fe, device_ids=range(n_gpu))
        model_cls = nn.DataParallel(model_cls, device_ids=range(n_gpu))

    if cfg['training']['pseudo_filename']:
        fname = cfg['training']['pseudo_filename']
        logger.info("Loading pseudo labels from {}".format(fname))
        pseudo = get_pseudo_labels(fname)
    else:
        pseudo = None

    for it in range(start_it, cfg['training']['iteration']):

        scheduler.step()

        trainer(batch_iterator, model_fe, model_cls, opt, it, loss_criterion,
                cfg, logger, writer, pseudo=pseudo)

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
                print_str = '[Val] Iteration {it}\tBest Acc source. {acc_src:.3f}\tBest Acc target. {acc_tgt:.3f}'.format(it=it+1, acc_src=best_acc_src, acc_tgt=best_acc_tgt)
                logger.info(print_str)

            state = {
                'iteration': it + 1,
                'model_fe_state': model_fe.state_dict(),
                'model_cls_state': model_cls.state_dict(),
                'opt_main_state': opt.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_acc_tgt' : best_acc_tgt,
                'best_acc_src' : best_acc_src
            }
            
            ckpt_path = os.path.join(logdir, 'checkpoint.pkl')
            save_path = ckpt_path
            torch.save(state, save_path)

            if is_best:
                best_path = os.path.join(logdir, 'best_model.pkl')
                torch.save(state, best_path)
            logger.info('[Checkpoint]: {} saved'.format(save_path))

if __name__ == '__main__':

    cfg, writer, logger, logdir = get_args()

    # setup random seeds
    seed=cfg.get('seed', 2344)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    main(cfg, writer, logger, logdir)