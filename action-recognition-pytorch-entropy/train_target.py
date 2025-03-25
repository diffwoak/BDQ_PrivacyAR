import os
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ["CUDA_VISIBLE_DEVICES"]= "0"
import shutil
import time
import sys
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.multiprocessing as mp
from torch.optim import lr_scheduler

import tensorboard_logger
import torch.nn.functional as F
import csv

from models._internally_replaced_utils import load_state_dict_from_url

from models import build_model
from utils.utils import (train, validate, train1, validate1, build_dataflow, get_augmentor,
                         save_checkpoint, train_target, validate_target, train_budget, validate_budget)
from utils.video_dataset import VideoDataSet
from utils.dataset_config import get_dataset_config
from opts import arg_parser


import torch
from  torch.nn.modules.loss import _Loss
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Argument 'interpolation' of type int.*")

from models.threed_models.utilityNet import I3Du
from models.threed_models.budgetNet import I3Db
from models.threed_models.degradNet import resnet_degrad
from models.threed_models.i3d_resnet import i3d_resnet
import random
import numpy as np

def set_seed(seed = 3407):
    torch.manual_seed(seed)
    # 如果你使用 CUDA，还需要设置 CUDA 的随机种子
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多 GPU
    # 设置 Python 的随机种子
    random.seed(seed)

    # 设置 NumPy 的随机种子
    np.random.seed(seed)

# 日志类工具
class Logger(object):

    def __init__(self, path, header):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')


        self.logger.writerow(header)
        self.header = header

    def __del(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])

        self.logger.writerow(write_values)
        self.log_file.flush()

# 计算输入的概率分布的熵
class EntropyLoss(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(EntropyLoss, self).__init__(size_average, reduce, reduction)

    # input is probability distribution of output classes
    def forward(self, input):
        if (input < 0).any() or (input > 1).any():
            raise Exception('Entropy Loss takes probabilities 0<=input<=1')

        input = input + 1e-16  # for numerical stability while taking log
        H = -torch.mean(torch.sum(input * torch.log(input), dim=1)) #  -Σ(p * log(p))

        return H


def main():
    global args
    parser = arg_parser()
    args = parser.parse_args()
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpu))
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count() 
    set_seed()
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size 
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    cudnn.benchmark = args.cudnn_benchmark

    num_classes_target, num_classes_budget, train_list_name, val_list_name, test_list_name, filename_seperator, image_tmpl, filter_video, label_file = get_dataset_config(args.dataset)
    args.num_classes_target = num_classes_target
    args.num_classes_budget = num_classes_budget


    if gpu is not None:
        print("Use GPU: {} for training".format(gpu))
    args.dist_url = 'tcp://127.0.0.1:23458'
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    if args.modality == 'rgb':
        args.input_channels = 3
    elif args.modality == 'flow':
        args.input_channels = 2 * 5

    model, arch_name = build_model(args)

    arch_name = f'{args.dataset}_train_target'

    model_degrad= model[0]
    model_target= model[1]
    model_budget= model[2]

    if args.rank == 0:
        print("=> creating model '{}'".format(arch_name))

    # gpu
    model_degrad = model_degrad.cuda(gpu)
    model_target = model_target.cuda(gpu)
    model_budget = model_budget.cuda(gpu)

    checkpoint_degrad = torch.load(f'results/{args.dataset}/adv/model_degrad.ckpt', map_location="cpu")
    model_degrad.load_state_dict(checkpoint_degrad)

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / args.world_size)
        args.workers = int(args.workers / ngpus_per_node)
        if args.sync_bn:# 同步批量归一化（SyncBatchNorm）
            process_group = torch.distributed.new_group(list(range(args.world_size)))
            model_degrad = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_degrad, process_group)
            model_target = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_target, process_group)
            model_budget = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_budget, process_group)
        # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
        model_degrad = torch.nn.parallel.DistributedDataParallel(model_degrad, device_ids=[gpu])
        model_target = torch.nn.parallel.DistributedDataParallel(model_target, device_ids=[gpu])
        model_budget = torch.nn.parallel.DistributedDataParallel(model_budget, device_ids=[gpu])
    elif gpu is not None:
        torch.cuda.set_device(gpu)
    else:
        # 使用DataParallel（仅在单机多卡非分布式时）
        model_degrad = torch.nn.DataParallel(model_degrad).cuda()
        model_target = torch.nn.DataParallel(model_target).cuda()
        model_budget = torch.nn.DataParallel(model_budget).cuda()
        args.rank = 0

    # Loss function
    train_criterion = nn.CrossEntropyLoss().cuda(gpu)
    val_criterion = nn.CrossEntropyLoss().cuda(gpu)
    train_entropy_criterion = EntropyLoss().cuda(gpu)
    val_entropy_criterion = EntropyLoss().cuda(gpu)
    reconst_criterion = nn.MSELoss().cuda(gpu)

    # Data loading code
    val_list = os.path.join(args.datadir, val_list_name)

    norm_value= 255
    args.threed_data = True

    val_augmentor = get_augmentor(False, args.input_size, scale_range=args.scale_range, mean=[110.63666788 / norm_value, 103.16065604 / norm_value, 96.29023126 / norm_value],
                                    std=[38.7568578 / norm_value, 37.88248729 / norm_value, 40.02898126 / norm_value], disable_scaleup=args.disable_scaleup,
                                    threed_data=args.threed_data,
                                    is_flow=True if args.modality == 'flow' else False,
                                    version=args.augmentor_ver)

    val_dataset = VideoDataSet(args.datadir, val_list, args.groups, args.frames_per_group,
                                num_clips=args.num_clips,
                                modality=args.modality, image_tmpl=image_tmpl,
                                dense_sampling=args.dense_sampling,
                                transform=val_augmentor, is_train=False, test_mode=False,
                                seperator=filename_seperator, filter_video=filter_video)

    val_loader = build_dataflow(val_dataset, is_train=False, batch_size=args.batch_size,
                                workers=args.workers,
                                is_distributed=args.distributed)

    log_folder = os.path.join(args.logdir, arch_name)
    if args.rank == 0:
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)

    # BDQ编码器
    # checkpoint_degrad = torch.load(f'/root/project/BDQ_PrivacyAR/action-recognition-pytorch-entropy/results/{args.dataset}/adv/model_degrad.ckpt', map_location="cpu")
    
        # 单卡加载
    # state_dict = {k.replace("module.", ""): v for k, v in checkpoint_degrad.items()}
    # model_degrad.load_state_dict(state_dict)
    del checkpoint_degrad

    train_list = os.path.join(args.datadir, train_list_name)

    train_augmentor = get_augmentor(True, args.input_size, scale_range=args.scale_range, mean=[110.63666788 / norm_value, 103.16065604 / norm_value, 96.29023126 / norm_value],
                                    std=[38.7568578 / norm_value, 37.88248729 / norm_value, 40.02898126 / norm_value],
                                    disable_scaleup=args.disable_scaleup,
                                    threed_data=args.threed_data,
                                    is_flow=True if args.modality == 'flow' else False,
                                    version=args.augmentor_ver)

    train_dataset = VideoDataSet(args.datadir, train_list, args.groups, args.frames_per_group,
                                    num_clips=args.num_clips,
                                    modality=args.modality, image_tmpl=image_tmpl,
                                    dense_sampling=args.dense_sampling,
                                    transform=train_augmentor, is_train=True, test_mode=False,
                                    seperator=filename_seperator, filter_video=filter_video)

    train_loader = build_dataflow(train_dataset, is_train=True, batch_size=args.batch_size,
                                    workers=args.workers, is_distributed=args.distributed)

    total_epochs= args.epochs

    save_dest = f'results/{args.dataset}'
    if not os.path.isdir(save_dest):
        os.mkdir(save_dest)

    #----------------- START OF TARGET MODEL TRAINING------------------#

    train_logger = Logger(save_dest+f'/target/'+'model_target_train'+'.log',['epoch','prec1_T', 'prec5_T'])
    val_logger = Logger(save_dest+f'/target/'+'model_target_val'+'.log',['epoch','prec1_T', 'prec5_T'])

    params = model_target.parameters()
    optimizer = torch.optim.SGD(params, args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max= total_epochs, eta_min=1e-7, verbose=True)
    best_val_top1 = float(80.0)
    for epoch in range(args.start_epoch, total_epochs):
        
        trainT_top1, trainT_top5, _, _, train_losses= train1(train_loader, model_degrad, model_target, model_budget, optimizer, train_criterion, train_entropy_criterion,  epoch + 1, gpu_id= gpu, rank=args.rank, step='target')
        if args.rank == 0:
            train_logger.log({'epoch': epoch,'prec1_T': trainT_top1.item(), 'prec5_T': trainT_top5.item()})
        
        if args.distributed:
            dist.barrier()

        valT_top1, valT_top5,  _, _, val_losses = validate1(val_loader, model_degrad, model_target, model_budget, val_criterion, step='target', gpu_id= gpu)

        if args.rank == 0:
            val_logger.log({'epoch': epoch,'prec1_T': valT_top1.item(), 'prec5_T': valT_top5.item()})

        scheduler.step()

        if args.distributed:
            dist.barrier()

        print('Train: [{:03d}/{:03d}]\tLoss: {:4.4f}\tTopT@1: {:.4f}\tTopT@5: {:.4f}\t'.format(
                    epoch + 1, total_epochs, train_losses, trainT_top1, trainT_top5), flush=True)
        print('Val  : [{:03d}/{:03d}]\tLoss: {:4.4f}\tTopT@1: {:.4f}\tTopT@5: {:.4f}\t'.format(
                    epoch + 1, total_epochs, val_losses, valT_top1,valT_top5),flush=True)

        if args.rank == 0:
            model_dict = model_target.module.state_dict() if isinstance(model_target, torch.nn.parallel.DistributedDataParallel) else model_target.state_dict()
            # 保存最佳模型（当验证指标提升时）
            if valT_top1 > best_val_top1:
                best_val_top1 = valT_top1.item()
                # 提取模型参数（兼容DDP模式）
                # 基础保存名称
                save_name = f"model_target_epoch{epoch}_topT{valT_top1.item():.2f}.ckpt"
                print(f"New best model saved: {save_name}")
                # 始终保存当前 epoch 的模型
                torch.save(model_dict, f"{save_dest}/target/{save_name}")
            torch.save(model_dict, save_dest+'/target/'+ 'model_target' + '.ckpt')

        # if args.rank == 0:
        #     if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        #         save_dict = model_target.module.state_dict()  # 提取内部模型
        #     else:
        #         save_dict = model_target.state_dict()

        #     torch.save(save_dict, save_dest+'/target/'+ 'model_target'+'.ckpt')        
        
        #----------------- END OF TARGET MODEL TRAINING------------------#
        
if __name__ == '__main__':
    main()
