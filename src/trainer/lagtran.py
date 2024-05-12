import torch

def train_lagtran(batch_iterator, model_fe, model_cls, opt, it, criterion_cls,
                    cfg, logger, writer, pseudo):

    # setting training mode
    if cfg["training"]["freeze_encoder"]:
        model_fe = model_fe.eval()
    else:
        model_fe = model_fe.train()
    model_cls.train()
    opt.zero_grad()

    # get data
    (_, img_src, lbl_src), (image_id, img_tgt, _) = next(batch_iterator)
    
    img_src, img_tgt, lbl_src = img_src.cuda(), img_tgt.cuda(), lbl_src.cuda()

    # forward
    bs_size = img_src.size(0)
    all_images = torch.cat([img_src, img_tgt], dim=0)
    if cfg["training"]["freeze_encoder"]:
        with torch.no_grad():
            output = model_fe(all_images).detach()
        output = model_cls(output)
    else:
        output = model_cls(model_fe(all_images), feat=False)
        
    output_src, output_tgt = output.split(bs_size)

    ## classification loss
    src_loss = torch.mean(criterion_cls(output_src, lbl_src).squeeze())

    image_id = [str(fid) for fid in image_id.tolist()]
    lbl_tgt = torch.tensor([int(pseudo[fid]) for fid in image_id], dtype=torch.long)
    lbl_tgt = lbl_tgt.cuda()
    tgt_loss = torch.mean(criterion_cls(output_tgt, lbl_tgt).squeeze())

    loss = src_loss + tgt_loss

    if loss.item() > 1e5 or torch.isnan(loss):
        raise Exception('Loss explosion: {}'.format(loss.item()))

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