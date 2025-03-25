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
from torchvision.transforms.functional import gaussian_blur

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


class GhostSuppressor(nn.Module):
    def __init__(self, alpha=0.5, kernel_size=3):
        super().__init__()
        self.alpha = alpha
        self.kernel_size = kernel_size
        
        # 预定义形态学操作核
        self.register_buffer('morph_kernel', 
            torch.ones(1, 1, kernel_size, kernel_size))

    def adaptive_threshold(self, x):
        """
        输入: x [B,C,T,H,W]
        输出: 阈值化后的张量 [B,C,T,H,W]
        """
        B, C, T, H, W = x.shape
        
        # 合并B和T维度以便并行处理
        ori_x = x.permute(0,2,1,3,4).contiguous().reshape(B*T, C, H, W)
        # 复制原始x
        x_merged = torch.abs(ori_x)
        # 计算局部均值 (3x3窗口)
        local_mean = F.avg_pool2d(x_merged, kernel_size=5, stride=1, padding=2)
        
        # 计算全局标准差 (按帧独立计算)
        global_std = torch.std(x_merged.view(B*T, C, -1), dim=2)      # [B*T,C]
        global_std = global_std.view(B*T, C, 1, 1)                   # [B*T,C,1,1]
        
        # 动态阈值
        threshold = local_mean + self.alpha * global_std
        # print(local_mean.shape)
        # # print(global_std)
        # print(x_merged[11].max())
        # print(local_mean[11].max())
        # print(global_std[11].max())

        # 二值化并恢复原始维度
        mask = (x_merged > threshold).float()
        return (mask * ori_x).reshape(B, T, C, H, W).permute(0,2,1,3,4)

    def morphological_closing(self, x):
        """
        输入: x [B,C,T,H,W]
        输出: 闭运算结果 [B,C,T,H,W]
        """
        B, C, T, H, W = x.shape
        
        # 合并维度: [B*C*T,1,H,W]
        x_merged = x.reshape(-1, 1, H, W)
        
        # 膨胀操作
        dilated = F.max_pool2d(
            x_merged, 
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size//2
        )
        
        # 腐蚀操作 (通过负片最大池化实现)
        eroded = -F.max_pool2d(
            -dilated, 
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.kernel_size//2
        )
        
        # 恢复原始维度
        return eroded.reshape(B, C, T, H, W)

    def forward(self, diff_frames):
        """
        输入: diff_frames [B,C,T,H,W]
        输出: 残影抑制后的差分帧 [B,C,T,H,W]
        """
        B, C, T, H, W = diff_frames.shape
        # 步骤1: 自适应阈值
        thresholded = self.adaptive_threshold(diff_frames)
        
        # 步骤2: 形态学闭运算
        # closed = self.morphological_closing(thresholded)
        
        # 步骤3: 高斯平滑 (可选)
        # smoothed = gaussian_blur(
        #     thresholded.contiguous().reshape(-1, 1, H, W), 
        #     kernel_size=3, 
        #     sigma=0.5
        # ).reshape(B, C, T, H, W)

        return thresholded

def rgb_to_grayscale(input_tensor):
    """
    将RGB视频张量转换为灰度图像
    输入: [batch, 3, frames, H, W] (取值范围建议[0,1]或[0,255])
    输出: [batch, 1, frames, H, W]
    """
    if input_tensor.size(1) != 3:
        raise ValueError("输入张量的通道数必须为3 (RGB)")
    
    # 定义RGB转灰度的权重 (依据ITU-R BT.601标准)
    weights = torch.tensor([0.2989, 0.5870, 0.1140], 
                          device=input_tensor.device,
                          dtype=input_tensor.dtype)  # [3]
    
    # 扩展维度以匹配输入张量: [3] → [1, 3, 1, 1, 1]
    weights = weights.view(1, 3, 1, 1, 1)
    
    # 加权求和: 按通道维度求和
    gray = (input_tensor * weights).sum(dim=1, keepdim=True)  # [B,1,T,H,W]
    
    return gray
class ResNet(nn.Module):
    def __init__(self, sig_scale=5, quantize_bits=4, quantize=True, avg=False,):
        super().__init__()
        self.gauss = GaussianSmoothing(3,5,3)
        self.bits = quantize_bits
        self.sig_scale = sig_scale
        self.bias = nn.Parameter(torch.from_numpy(np.linspace(0, 2 ** self.bits - 1, 2 ** self.bits, dtype='float32')[:-1] + 0.5))
        self.levels = np.linspace(0, 2 ** self.bits - 1, 2 ** self.bits, dtype='float32')[:-1]
        # 存储中间输出的变量
        self.blur_output = None
        self.diff_output = None
 
    # BDQ编码器
    def forward(self, x):
        # 高斯模糊
        x = self.gauss(x)      
        self.blur_output = x[:,:,1:,:,:]
        # 差分
        # x = rgb_to_grayscale(x)
        x_roll = torch.roll(x, 1, dims= 2)
        x = x-x_roll
        x = x[:,:,1:,:,:]
        ghost_suppressor = GhostSuppressor(alpha=0, kernel_size=3)
        x = ghost_suppressor(x)
        self.diff_output = torch.abs(x)
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


