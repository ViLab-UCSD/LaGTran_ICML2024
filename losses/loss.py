import torch.nn as nn
import torch.nn.functional as F

# class KLDivLoss(nn.Module):

#     def __init__(self, reduction="batchmean", top_pred=False):
#         super().__init__()
#         self.reduction = reduction
#         self.top_pred = top_pred
#         if top_pred:
#             self.loss = torch.nn.CrossEntropyLoss(reduction="none")
#         else:
#             self.loss = torch.nn.KLDivLoss(reduction=reduction)
    
#     def forward(self, input, target):
#         if self.top_pred:
#             # target = target.argmax(dim=-1)
#             return self.loss(input, target)
#         else:
#             input = F.log_softmax(input, dim=-1)
#             return self.loss(input, target)


class CrossEntropyLoss(nn.CrossEntropyLoss):

    def __init__(self, reduction="none"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target, weight=None):
        if weight is None:
            weight = self.weight
        return F.cross_entropy(input, target, weight=weight, reduction=self.reduction)