import torch

def train_daln(batch_iterator, model_fe, model_cls, opt, it, criterion_cls, criterion_d, 
                    cfg, logger, writer):

    # setting training mode
    model_fe.train()
    model_cls.train()
    opt.zero_grad()

    # get data
    (_, img_src, lbl_src), (_, img_tgt, _) = next(batch_iterator)
    img_src, img_tgt, lbl_src = img_src.cuda(), img_tgt.cuda(), lbl_src.cuda()

    # forward
    bs_size = img_src.size(0)
    all_images = torch.cat([img_src, img_tgt], dim=0)
    output, feature = model_cls(model_fe(all_images), feat=True)
    output_src, _ = output.split(bs_size)

    ## classification loss
    closs = torch.mean(criterion_cls(output_src, lbl_src).squeeze())

    ## adversarial loss
    daloss = criterion_d(feature, model_cls.out)

    # subtract the adversarial loss: https://github.com/xiaoachen98/DALN/tree/master
    loss = closs + daloss

    # back propagation
    loss.backward()
    opt.step()

    curr_lr = opt.param_groups[0]['lr']
    if (it + 1) % cfg['training']['print_interval'] == 0:
        print_str = 'Iteration: [{0}/{1}]\t' \
            'LR: [{curr_lr:.4g}]\t' \
            'CLoss {closs:.4f}\t' \
            'DALoss {daloss:.4f}'.format(
                it + 1, cfg['training']['iteration'], curr_lr=curr_lr,
                closs=closs.item(), daloss=daloss.item()
            )

        logger.info(print_str)

    writer.add_scalar('train/lr', curr_lr, it + 1)
    writer.add_scalar('train/c_loss', closs.item(), it + 1)
    writer.add_scalar('train/da_loss', daloss.item(), it + 1)