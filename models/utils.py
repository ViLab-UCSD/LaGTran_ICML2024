### utils.py
# Utility functions.
###

from utils import calc_coeff
import torch
import numpy as np
import torch.nn as nn
from typing import Optional, Tuple, Any
from torch.autograd import Function


def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1


# class GradientReverseLayer(torch.autograd.Function):
#     def __init__(self, iter_num=0, alpha=1.0, low_value=0.0, high_value=0.1, max_iter=1000.0):
#         self.iter_num = iter_num
#         self.alpha = alpha
#         self.low_value = low_value
#         self.high_value = high_value
#         self.max_iter = max_iter

#     @staticmethod
#     def forward(ctx, input):
#         ctx.iter_num += 1
#         output = input * 1.0
#         return output

#     @staticmethod
#     def backward(self, grad_output):
#         self.coeff = calc_coeff(self.iter_num, self.high_value, self.low_value, self.alpha, self.max_iter)
#         return -self.coeff * grad_output

class GradientReverseLayer(torch.autograd.Function):
    iter_num = 0
    max_iter = 1000
    @staticmethod
    def forward(ctx, input):
        GradientReverseLayer.iter_num += 1
        return input * 1.0

    @staticmethod
    def backward(ctx, gradOutput):
        alpha = 1
        low = 0.0
        high = 0.1
        iter_num, max_iter = GradientReverseLayer.iter_num, GradientReverseLayer.max_iter 
        coeff = calc_coeff(iter_num, high, low, alpha, max_iter)
        return -coeff * gradOutput


class GradientReverseFunction(Function):

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


# class GradientReverseLayer(nn.Module):
#     def __init__(self):
#         super(GradientReverseLayer, self).__init__()

#     def forward(self, *input):
#         return GradientReverseFunction.apply(*input)


class WarmStartGradientReverseLayer(nn.Module):
    """Gradient Reverse Layer :math:`\mathcal{R}(x)` with warm start

        The forward and backward behaviours are:

        .. math::
            \mathcal{R}(x) = x,

            \dfrac{ d\mathcal{R}} {dx} = - \lambda I.

        :math:`\lambda` is initiated at :math:`lo` and is gradually changed to :math:`hi` using the following schedule:

        .. math::
            \lambda = \dfrac{2(hi-lo)}{1+\exp(- α \dfrac{i}{N})} - (hi-lo) + lo

        where :math:`i` is the iteration step.

        Args:
            alpha (float, optional): :math:`α`. Default: 1.0
            lo (float, optional): Initial value of :math:`\lambda`. Default: 0.0
            hi (float, optional): Final value of :math:`\lambda`. Default: 1.0
            max_iters (int, optional): :math:`N`. Default: 1000
            auto_step (bool, optional): If True, increase :math:`i` each time `forward` is called.
              Otherwise use function `step` to increase :math:`i`. Default: False
        """

    def __init__(self, alpha: Optional[float] = 1.0, lo: Optional[float] = 0.0, hi: Optional[float] = 1.,
                 max_iters: Optional[int] = 1000., auto_step: Optional[bool] = False):
        super(WarmStartGradientReverseLayer, self).__init__()
        self.alpha = alpha
        self.lo = lo
        self.hi = hi
        self.iter_num = 0
        self.max_iters = max_iters
        self.auto_step = auto_step

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """"""
        coeff = float(
            2.0 * (self.hi - self.lo) / (1.0 + np.exp(-self.alpha * self.iter_num / self.max_iters))
            - (self.hi - self.lo) + self.lo
        )
        if self.auto_step:
            self.step()
        return GradientReverseFunction.apply(input, coeff)

    def step(self):
        """Increase iteration number :math:`i` by 1"""
        self.iter_num += 1
