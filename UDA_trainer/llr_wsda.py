import torch
import numpy as np

def train_llr_wsda(batch_iterator, model_fe, model_cls, opt, it, criterion_cls, criterion_kl,
                    cfg, logger, writer, pseudo):

    # setting training mode
    model_fe.train()
    model_cls.train()
    opt.zero_grad()

    # get data
    (_, _, _), (fid_tgt, img_tgt, _) = next(batch_iterator)
    fid_tgt = [str(fid) for fid in fid_tgt.tolist()]
    # import pdb; pdb.set_trace()
    lbl_tgt = torch.tensor([int(pseudo.get(fid, -1)) for fid in fid_tgt], dtype=torch.long)
    valid_tgt = lbl_tgt != -1
    img_tgt = img_tgt.cuda()
    lbl_tgt = lbl_tgt.cuda()

    # forward
    # with torch.no_grad():
    #     output_tgt = model_fe(img_tgt).detach()
    output_tgt = model_fe(img_tgt)
    output_tgt = model_cls(output_tgt)
        
    ## classification loss
    if sum(valid_tgt) > 0:
        output_tgt = output_tgt[valid_tgt]
        lbl_tgt = lbl_tgt[valid_tgt]
        tgt_loss = torch.mean(criterion_kl(output_tgt, lbl_tgt))
    else:
        assert 1==2
        tgt_loss = torch.tensor(0).cuda()
    loss = tgt_loss

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
            'LR: [{curr_lr:.4g}]\t'\
            'TgtLoss {tgt_loss:.4f}'.format(
                it + 1, cfg['training']['iteration'], curr_lr=curr_lr,
                tgt_loss=tgt_loss.item()
            )

        logger.info(print_str)

    writer.add_scalar('train/lr', curr_lr, it + 1)
    # writer.add_scalar('train/src_loss', src_loss.item(), it + 1)
    writer.add_scalar('train/tgt_loss', tgt_loss.item(), it + 1)