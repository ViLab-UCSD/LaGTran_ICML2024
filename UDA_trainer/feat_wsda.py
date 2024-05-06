import torch
import numpy as np

def train_wsda_feat(batch_iterator, model_fe, model_cls, opt, it, criterion_cls, criterion_kl,
                    cfg, logger, writer, pseudo):

    # setting training mode
    model_cls.train()
    opt.zero_grad()

    # get data
    (_, img_src, lbl_src), (fid_tgt, img_tgt, _) = next(batch_iterator)
    fid_tgt = [str(fid) for fid in fid_tgt.tolist()]
    # import pdb; pdb.set_trace()
    lbl_tgt = torch.tensor([int(pseudo.get(fid,-1)) for fid in fid_tgt], dtype=torch.long)
    # valid_tgt = lbl_tgt != -1
    img_src, img_tgt, lbl_src = img_src.cuda(), img_tgt.cuda(), lbl_src.cuda()
    lbl_tgt = lbl_tgt.cuda()

    # forward
    bs_size = img_src.size(0)
    all_images = torch.cat([img_src, img_tgt], dim=0)
    output = model_cls(all_images, feat=False)
        
    output_src, output_tgt = output.split(bs_size)

    ## classification loss
    src_loss = torch.mean(criterion_cls(output_src, lbl_src).squeeze())

    ## target soft KL Loss
    # output_tgt = output_tgt[valid_tgt]
    # lbl_tgt = lbl_tgt[valid_tgt]
    tgt_loss = torch.mean(criterion_kl(output_tgt, lbl_tgt))

    loss = src_loss + tgt_loss

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
            'SrcLoss {src_loss:.4f}\t' \
            'TgtLoss {tgt_loss:.4f}'.format(
                it + 1, cfg['training']['iteration'], curr_lr=curr_lr,
                src_loss=src_loss.item(), tgt_loss=tgt_loss.item()
            )

        logger.info(print_str)

    writer.add_scalar('train/lr', curr_lr, it + 1)
    writer.add_scalar('train/src_loss', src_loss.item(), it + 1)
    writer.add_scalar('train/tgt_loss', tgt_loss.item(), it + 1)