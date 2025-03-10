# from . import s3d, i3d, s3d_resnet, i3d_resnet, resnet, inception_v1
# 替换为如下
from . import i3d, i3d_resnet, resnet, inception_v1

MODEL_TABLE = {
    'i3d': i3d,
    'i3d_resnet': i3d_resnet,
    'resnet': resnet,
    'inception_v1': inception_v1
}

# 构建和命名不同的深度学习模型
def build_model(args, test_mode=False):
    """
    Args:
        args: all options defined in opts.py and num_classes
        test_mode:
    Returns:
        network model
        architecture name
    """
    model = MODEL_TABLE[args.backbone_net](**vars(args))
    network_name = model.network_name if hasattr(model, 'network_name') else args.backbone_net
    arch_name = "{dataset}".format(
        dataset=args.dataset)

    # add setting info only in training
    if not test_mode:
        arch_name += "-bs{}-e{}".format(args.batch_size, args.epochs)
    return model, arch_name
