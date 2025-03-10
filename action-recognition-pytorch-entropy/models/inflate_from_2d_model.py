import torch
from collections import OrderedDict


def inflate_from_2d_model(state_dict_2d, state_dict_3d, skipped_keys=None, inflated_dim=2):

    if skipped_keys is None:
        skipped_keys = []

    missed_keys = []
    new_keys = []
    # 遍历 state_dict_2d 和 state_dict_3d，找出两者中不匹配的层名称
    for old_key in state_dict_2d.keys():
        if old_key not in state_dict_3d.keys():
            missed_keys.append(old_key)
    for new_key in state_dict_3d.keys():
        if new_key not in state_dict_2d.keys():
            new_keys.append(new_key)
    # print("Missed tensors: {}".format(missed_keys))
    # print("New tensors: {}".format(new_keys))
    # print("Following layers will be skipped: {}".format(skipped_keys))

    state_d = OrderedDict()
    unused_layers = [k for k in state_dict_2d.keys()]
    uninitialized_layers = [k for k in state_dict_3d.keys()]
    initialized_layers = []
    for key, value in state_dict_2d.items():
        # 如果某些层的名称包含 skipped_keys 中的关键字，这些层会被忽略
        skipped = False
        for skipped_key in skipped_keys:
            if skipped_key in key:
                skipped = True
                break
        if skipped:
            continue
        new_value = value
        # only inflated conv's weights
        if key in state_dict_3d:
            if value.ndimension() == 4 and 'weight' in key:
                value = torch.unsqueeze(value, inflated_dim) # 为 2D 权重增加一个维度
                repeated_dim = torch.ones(state_dict_3d[key].ndimension(), dtype=torch.int)
                repeated_dim[inflated_dim] = state_dict_3d[key].size(inflated_dim)
                new_value = value.repeat(repeated_dim.tolist()) # 根据 3D 权重的目标尺寸，使用 repeat 函数扩展权重到目标大小
            state_d[key] = new_value
            initialized_layers.append(key)
            uninitialized_layers.remove(key)
            unused_layers.remove(key)

    # print("Initialized layers: {}".format(initialized_layers)) # 成功初始化的 3D 层
    # print("Uninitialized layers: {}".format(uninitialized_layers)) # 3D 模型中未被初始化的层
    # print("Unused layers: {}".format(unused_layers)) # 2D 模型中未使用的层

    return state_d
