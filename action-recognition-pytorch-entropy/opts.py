import argparse
from models.model_builder import MODEL_TABLE
from utils.dataset_config import DATASET_CONFIG

def arg_parser():
    parser = argparse.ArgumentParser(description='PyTorch Action recognition Training')

    # model definition
    parser.add_argument('--backbone_net', default='i3d', type=str, help='backbone network',
                        choices=list(MODEL_TABLE.keys()))
    parser.add_argument('-d', '--depth', default=18, type=int, metavar='N',
                        help='depth of resnet (default: 18)', choices=[18, 34, 50, 101, 152]) #暂无用
    parser.add_argument('--dropout', default=0.5, type=float,
                        help='dropout ratio before the final layer')  #暂无用
    parser.add_argument('--groups', default=16, type=int, help='number of frames')
    parser.add_argument('--frames_per_group', default=1, type=int,
                        help='[uniform sampling] number of frames per group; '
                             '[dense sampling]: sampling frequency') # 每个group采样多少帧
    # dense_sampling: frames_per_group 为间隔（如每隔1帧）采样 groups 个连续或邻近帧
    # not dense_sampling: 将视频均分为 groups 个时间段, 在每个时间段内随机采样 frames_per_group 帧
    parser.add_argument('--without_t_stride', dest='without_t_stride', action='store_true',
                        help='skip the temporal pooling in the model') # 暂无用
    parser.add_argument('--pooling_method', default='max',
                        choices=['avg', 'max'], help='method for temporal pooling method') # 暂无用
    parser.add_argument('--dw_t_conv', dest='dw_t_conv', action='store_true',
                        help='[S3D model] only enable depth-wise conv for temporal modeling') # 暂无用
    # model definition: temporal model for 2D models 暂无用
    parser.add_argument('--temporal_module_name', default=None, type=str,
                        help='[2D model] which temporal aggregation module to use. None == TSN',
                        choices=[None, 'TSN', 'TAM'])
    parser.add_argument('--blending_frames', default=3, type=int, help='For TAM only.')
    parser.add_argument('--blending_method', default='sum',
                        choices=['sum', 'max'], help='method for blending channels in TAM')
    parser.add_argument('--no_dw_conv', dest='dw_conv', action='store_false',
                        help='[2D model] disable depth-wise conv for TAM')

    # training setting
    parser.add_argument('--gpu', type=int, nargs='+', default=None,
                        help='Specify a list of GPU IDs to use (e.g., --gpu 0 1).')
    parser.add_argument('--disable_cudnn_benchmark', dest='cudnn_benchmark', action='store_false',
                        help='Disable cudnn to search the best mode (avoid OOM)')
    parser.add_argument('-b', '--batch-size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--lr_scheduler', default='multisteps', type=str,
                        help='learning rate scheduler',
                        choices=['step', 'multisteps', 'cosine', 'plateau']) # 暂无用
    parser.add_argument('--lr_steps', default=[50, 100, 150], type=float, nargs="+",
                        metavar='LRSteps', help='[step]: use a single value: the periodto decay '
                                                'learning rate by 10. '
                                                '[multisteps] epochs to decay learning rate by 10') # 暂无用
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--nesterov', action='store_true', help='enable nesterov momentum optimizer')
    parser.add_argument('--weight_decay', '--wd', default=1e-5, type=float,
                        metavar='W', help='weight decay (default: 1e-5)')
    parser.add_argument('--weight',default = 2, type = int, help = 'adv weight (default: 2)')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', dest='pretrained', type=str, metavar='PATH',
                        help='use pre-trained model')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--clip_gradient', '--cg', default=None, type=float,
                        help='clip the total norm of gradient before update parameter') # 防止梯度爆炸
#     parser.add_argument('--no_imagenet_pretrained', dest='imagenet_pretrained',
#                         action='store_false',
#                         help='disable to load imagenet model')

    # data-related
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--datadir', metavar='DIR', help='path to dataset file list')
    parser.add_argument('--dataset', default='SBU',
                        choices=list(DATASET_CONFIG.keys()), help='which dataset.')
    parser.add_argument('--threed_data', action='store_false',
                        help='format data to 5D for 3D onv.')
    parser.add_argument('--input_size', default=224, type=int, metavar='N', help='spatial size')
    parser.add_argument('--disable_scaleup', action='store_true',
                        help='do not scale up and then crop a small region, '
                             'directly crop the input_size from center.')
    parser.add_argument('--random_sampling', action='store_true',
                        help='[Uniform sampling only] perform non-deterministic frame sampling '
                             'for data loader during the evaluation.')
    parser.add_argument('--dense_sampling', action='store_true',
                        help='perform dense sampling for data loader')
    parser.add_argument('--augmentor_ver', default='v1', type=str, choices=['v1', 'v2'],
                        help='[v1] TSN data augmentation, [v2] resize the shorter side to `scale_range`')
                        # 保留V1用于复现旧实验结果，1确保覆盖特定尺寸特征 2适合对尺度敏感的任务（如动作识别）
                        # V2作为改进后的新方案供选择
    parser.add_argument('--scale_range', default=[240, 250], type=int, nargs="+",
                        metavar='scale_range', help='scale range for augmentor v2')
    parser.add_argument('--modality', default='rgb', type=str, help='rgb or flow',
                        choices=['rgb', 'flow'])
#     parser.add_argument('--mean', type=float, nargs="+", metavar='MEAN',
#                         help='[Data normalization] mean, dimension should be 3 for RGB, 1 for flow')
#     parser.add_argument('--std', type=float, nargs="+", metavar='STD',
#                         help='[Data normalization] std, dimension should be 3 for RGB, 1 for flow')
    # logging
    parser.add_argument('--logdir', default='train_log', type=str, help='log path')
#     parser.add_argument('--print-freq', default=100, type=int,
#                         help='frequency to print the log during the training')
#     parser.add_argument('--show_model', action='store_true',
#                         help='show model and then exit intermediately')

    # for testing and validation
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--num_crops', default=1, type=int, choices=[1, 3, 5, 10],
                        help='[Test.py only] number of crops.')
    parser.add_argument('--num_clips', default=1, type=int,
                        help='[Test.py only] number of clips.')
    parser.add_argument('--model_type',default='target',type=str,choices=['target','budget'],
                        help='[Test.py only] type of evaluate model')
                

    # for distributed learning, not supported yet
    parser.add_argument('--sync-bn', action='store_true',
                        help='sync BN across GPUs')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--bdq_v',default='v0',type=str,choices=['v0','v3'],
                        help='choose BDQ model to train or test')
    parser.add_argument('--abla', type=str, nargs='+', default=['B','D','S','Q'],
                        help='Specify a list of BDQ modules to use (e.g., --abla B D).')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')

    return parser
