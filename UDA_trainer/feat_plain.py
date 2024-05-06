import torch

def train_plain_feat(batch_iterator, model_fe, model_cls, opt, it, criterion_cls,
            cfg, logger, writer):

    # setting training mode
    model_cls = model_cls.train()
    opt.zero_grad()

    # get data
    (_, feat_src, lbl_src), (_, feat_tgt, lbl_tgt) = next(batch_iterator)
    feat_src = feat_src.cuda()
    lbl_src = lbl_src.cuda()
    feat_tgt = feat_tgt.cuda()
    lbl_tgt = lbl_tgt.cuda()

    # forward
    output_src = model_cls(feat_src, feat=False)
    if cfg["training"]["use_target"]: 
        output_tgt = model_cls(feat_tgt, feat=False)

    loss = torch.mean(criterion_cls(output_src, lbl_src).squeeze())
    if cfg["training"]["use_target"]: 
        loss += torch.mean(criterion_cls(output_tgt, lbl_tgt).squeeze())

    if loss.item() > 1e5 or torch.isnan(loss):
        logger.info('Loss explosion: {}'.format(loss.item()))
        import pdb; pdb.set_trace()
        return

    # back propagation
    loss.backward()
    opt.step()

    curr_lr = opt.param_groups[0]['lr']
    if (it + 1) % cfg['training']['print_interval'] == 0:
        print_str = 'Iteration: [{0}/{1}]\t' \
            'LR: [{curr_lr:.4g}]\t' \
            'CLoss {closs:.4f}\t'.format(
                it + 1, cfg['training']['iteration'], curr_lr=curr_lr,
                closs=loss.item()
            )

        logger.info(print_str)

    writer.add_scalar('train/lr', curr_lr, it + 1)
    writer.add_scalar('train/c_loss', loss.item(), it + 1)
