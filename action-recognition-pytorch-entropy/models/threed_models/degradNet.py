import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
#from .quantization import *
import numpy as np

import numbers
from matplotlib import cm
from PIL import Image, ImageOps
import imageio
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm


######################################################################################

class GaussianSmoothing(nn.Module):

    def __init__(self, channels, kernel_size, sigma, dim=3):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [1,kernel_size,kernel_size]
        if isinstance(sigma, numbers.Number):
            sigma = [1,sigma,sigma]

        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ],
            indexing='ij'
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.groups = channels
        self.k_size = kernel_size[-1]

        self.conv = nn.Conv3d(3, 3, groups= self.groups, kernel_size= kernel_size, bias=False, padding=(0,(self.k_size-1)//2,(self.k_size-1)//2))
        self.conv.weight = torch.nn.Parameter(torch.FloatTensor(kernel))
        self.conv.weight.requires_grad = False


    def forward(self, input):
        input = self.conv(input)
        self.conv.weight.data = torch.clamp(self.conv.weight.data, min=0)
        return input



# class GaussianSmoothing(nn.Module):
#     def __init__(self, channels, kernel_size, sigma_init, dim=3):
#         super(GaussianSmoothing, self).__init__()
#         self.channels = channels
#         self.dim = dim

#         # 处理 kernel_size 格式
#         if isinstance(kernel_size, numbers.Number):
#             kernel_size = [kernel_size] * dim  # 扩展到所有维度
#         self.kernel_size = kernel_size

#         # 初始化可学习的 sigma（每个空间维度独立）
#         if isinstance(sigma_init, numbers.Number):
#             sigma_init = [sigma_init] * (dim - 1)  # 假设第一个维度不需要模糊（如通道维度）
#         self.sigma = nn.Parameter(torch.tensor(sigma_init, dtype=torch.float32))

#     def forward(self, x):
#         # 动态生成高斯核
#         kernel = 1
#         meshgrids = torch.meshgrid(
#             [
#                 torch.arange(size, dtype=torch.float32, device=x.device)
#                 for size in self.kernel_size
#             ],
#             indexing='ij'
#         )

#         # 遍历每个维度生成高斯分布
#         for i, (size, mgrid) in enumerate(zip(self.kernel_size, meshgrids)):
#             if i == 0:
#                 # 跳过第一个维度（通常为通道维度，kernel_size=1）
#                 continue  
#             std = self.sigma[i-1] if i <= len(self.sigma) else 1.0  # 取对应的 sigma
#             mean = (size - 1) / 2
#             kernel *= (1 / (std * math.sqrt(2 * math.pi))) * \
#                       torch.exp(-((mgrid - mean) / (std + 1e-6)) ** 2 / 2)

#         # 归一化并调整形状
#         kernel = kernel / torch.sum(kernel)
#         kernel = kernel.view(1, 1, *kernel.size())
#         kernel = kernel.repeat(self.channels, *[1] * (kernel.dim() - 1))

#         # 计算 padding
#         padding = [(k - 1) // 2 for k in self.kernel_size]

#         # 根据维度选择卷积函数
#         if self.dim == 3:
#             return F.conv3d(x, kernel, groups=self.channels, padding=padding)
#         elif self.dim == 2:
#             return F.conv2d(x, kernel, groups=self.channels, padding=padding)
#         else:
#             raise ValueError("Unsupported dimension: {}".format(self.dim))


class ResNet(nn.Module):
    def __init__(self, sig_scale=5, quantize_bits=4, quantize=True, avg=False,):
        super().__init__()
        self.gauss = GaussianSmoothing(3,5,3)
        self.bits = quantize_bits
        self.sig_scale = sig_scale
        self.bias = nn.Parameter(torch.from_numpy(np.linspace(0, 2 ** self.bits - 1, 2 ** self.bits, dtype='float32')[:-1] + 0.5))
        self.levels = np.linspace(0, 2 ** self.bits - 1, 2 ** self.bits, dtype='float32')[:-1]
 
    # BDQ编码器
    def forward(self, x):
        # 高斯模糊
        x = self.gauss(x)        
        # 差分
        x_roll = torch.roll(x, 1, dims= 2)
        x = x-x_roll
        x = x[:,:,1:,:,:]
        # 量化
        qmin = 0.
        qmax = 2. ** self.bits - 1.
        min_value = x.min()
        max_value = x.max()
        scale_value = (max_value - min_value) / (qmax - qmin)
        scale_value = max(scale_value, 1e-4)
        x = ((x - min_value) / ((max_value - min_value) + 1e-4)) * (qmax - qmin)
        y = torch.zeros(x.shape, device=x.device)
        self.bias.data = self.bias.data.clamp(0, (2 ** self.bits - 1))
        self.bias.data = self.bias.data.sort(0).values

        for i in range(self.levels.shape[0]):
            y = y + torch.sigmoid(self.sig_scale * ((x) - self.bias[i]))

        y = y.mul(scale_value).add(min_value)
        
        return y, self.bias

def resnet_degrad():
    """
    Constructs a ResNet-10 model.
    """
    model = ResNet()
    return model


