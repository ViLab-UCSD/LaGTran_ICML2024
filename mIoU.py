## compute mIoU given the label and prediction tensors.

import torch

def mIoU(label: torch.Tensor, pred: torch.Tensor) -> float:
    """
    Return the mean intersection over union of the given label and prediction tensors.

    Args:
        label (torch.Tensor), BxHxW: The label tensor.
        pred (torch.Tensor), BxHXW: The prediction tensor.
    Returns:
        float: The mean intersection over union.
    """

    ## convert the label to one hot
    nClass = label.max().item() + 1
    one_hot_label = torch.eye(nClass)[label].permute(0, 3, 1, 2).float()
    one_hot_pred = torch.eye(nClass)[pred].permute(0, 3, 1, 2).float()

    intersection = torch.sum(one_hot_label * one_hot_pred, dim=(0, 2, 3))
    union = torch.sum(one_hot_label + one_hot_pred, dim=(0, 2, 3)) - intersection
    iou = intersection / (union + 1e-6)

    return iou.mean().item()