from .functional import revgrad
from torch.nn import Module
from torch import tensor

# 包装了 functional.py 中定义的 RevGrad 函数，使其可以作为一个层（layer）在神经网络中使用
class RevGrad(Module):
    def __init__(self, alpha=2., *args, **kwargs):
        """
        A gradient reversal layer.

        This layer has no parameters, and simply reverses the gradient
        in the backward pass.
        """
        super().__init__(*args, **kwargs)

        self._alpha = tensor(alpha, requires_grad=False)
# 接受一个参数 alpha，用于指定在反向传播时梯度的反转系数
    def forward(self, input_):
        return revgrad(input_, self._alpha)
