import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import NoNorm


from .utilityNet import I3Du
from .budgetNet import I3Db
from .i3d_resnet import i3d_resnet




# 考虑换数据集的问题，则num_classes需更改，但是目前的类别都直接藏在模型结构里面
def i3d(num_classes_target, num_classes_budget, dropout, without_t_stride, pooling_method,bdq_v,abla, **kwargs):
# def i3d(num_classes, dropout, without_t_stride, pooling_method, **kwargs):

    # print(f"num_classes: {num_classes}")
    if bdq_v == 'v0':
        from .degradNet_v0 import resnet_degrad
        model_degrad =  resnet_degrad()
    elif bdq_v == 'v3':
        from .degradNet_v3 import resnet_degrad
        model_degrad =  resnet_degrad(abla) # 只有v3搞了消融实验
    # model_utility = I3Du()
    model_utility = i3d_resnet(num_classes = num_classes_target, dropout = dropout, without_t_stride = without_t_stride)
    model_budget = I3Db(num_classes = num_classes_budget)





    return [model_degrad, model_utility, model_budget] 







