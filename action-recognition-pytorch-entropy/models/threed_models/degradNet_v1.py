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
    def __init__(self, sig_scale=5, quantize_bits=3, time_steps = [1,2],quantize=True, avg=False,):
        super().__init__()
        self.gauss = GaussianSmoothing(3,5,3)
        # 差分模块参数
        self.time_steps = time_steps  # 新增：多时间步长参数，如[1,2,3]
        weights = torch.exp(torch.linspace(2, 0, len(time_steps)))  # 指数衰减
        self.alpha = nn.Parameter(weights/weights.sum())
        # self.alpha = nn.Parameter(torch.ones(len(time_steps)) / len(time_steps))
        # 量化模块参数
        self.sig_scale = sig_scale
        self.bits = quantize_bits
        self.bias = nn.Parameter(torch.from_numpy(np.linspace(0, 2 ** self.bits - 1, 2 ** self.bits, dtype='float32')[:-1] + 0.5))
        self.levels = np.linspace(0, 2 ** self.bits - 1, 2 ** self.bits, dtype='float32')[:-1]
        # 存储中间输出的变量
        self.blur_output = None
        self.diff_output = None


    def compute_multi_scale_diff(self, x):
        """
        计算多尺度帧间差异
        输入：x [B,C,T,H,W]
        输出：加权后的差异帧 [B,C,T',H,W]
        """
        B, C, T, H, W = x.shape
        all_diffs = []
        
        # 遍历每个时间步长k
        for k in self.time_steps:
            # 获取t-k帧（超出范围用零填充）
            x_shift = torch.zeros_like(x)
            if k < T:
                x_shift[:, :, k:] = x[:, :, :-k]
            
            # 计算绝对差异 |B_t - B_{t-k}|
            diff = torch.abs(x - x_shift)
            all_diffs.append(diff)
        
        # 拼接不同尺度的差异 [K, B,C,T,H,W]
        all_diffs = torch.stack(all_diffs, dim=0)  # [K,B,C,T,H,W]
        
        # 计算注意力权重（softmax归一化）
        weights = torch.softmax(self.alpha, dim=0)  # [K]
        
        # 加权融合多尺度差异
        weighted_diff = torch.einsum('k,kbcthw->bcthw', weights, all_diffs)
        
        # 裁剪有效时间范围（去除前max(time_steps)帧）
        max_k = max(self.time_steps)
        return weighted_diff[:, :, max_k:, :, :]  # [B,C,T-max_k,H,W]

    # BDQ编码器
    def forward(self, x):
        # 高斯模糊
        x = self.gauss(x)      
        self.blur_output = x[:,:,len(self.time_steps):,:,:]
        # 差分
        # x_roll = torch.roll(x, 1, dims= 2)
        # x = x-x_roll
        # x = x[:,:,1:,:,:]
        x = self.compute_multi_scale_diff(x)  # 输出 [B,C,T',H,W]
        self.diff_output = x
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
    model = ResNet()
    return model


