import torch
from metrics import averageMeter, accuracy, percls_accuracy

def val(data_loader, model_fe, model_cls, it, n_classes, domain, logger, writer):

    # setup average meters
    top1 = averageMeter()
    top5 = averageMeter()

    # setting training mode
    if model_fe:
        model_fe.eval()
    model_cls.eval()

    all_preds = []
    all_labels = []
    for (step, value) in enumerate(data_loader):

        image = value[1]
        target = value[2].cuda(non_blocking=True)

        if isinstance(image, torch.Tensor):
            image = image.cuda()
            # forward
            with torch.no_grad():
                if isinstance(model_cls, (list, tuple)):
                    features = model_fe(image)
                    output = torch.sum(torch.stack([cls(features) for cls in model_cls]), dim=0)
                else:
                    if model_fe:
                        output = model_cls(model_fe(image), feat=False)
                    else:
                        output = model_cls(image, feat=False)
        else: ## Video Evaluation
            logits_all_clips = []
            for clip in image:
                clip = clip.cuda()
                with torch.no_grad():
                    output = model_cls(model_fe(clip), feat=False)
                logits_all_clips.append(output)
            logits_all_clips = torch.stack(logits_all_clips)
            output = logits_all_clips.max(dim=0).values

        # measure accuracy
        prec1, prec5 = accuracy(output, target, topk=(1, 5))
        top1.update(prec1, output.size(0))
        top5.update(prec5, output.size(0))

        # per class accuracy metrics
        all_preds.extend(output.argmax(1).cpu().numpy().tolist())
        all_labels.extend(target.cpu().numpy().tolist())

    classwise_accuracy = percls_accuracy(all_preds, all_labels)

    logger.info('[Val] Iteration {it}\tTop 1 Acc {top1.avg:.3f}\tClasswise Acc. {cls_acc:.3f}'.format(it=it+1, top1=top1, cls_acc=classwise_accuracy.mean()))

    return top1.avg, top5.avg