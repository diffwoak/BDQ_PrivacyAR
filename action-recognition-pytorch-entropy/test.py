import os
import time

import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tqdm import tqdm

from models import build_model
from utils.utils import build_dataflow, AverageMeter, accuracy
from utils.video_transforms import *
from utils.video_dataset import VideoDataSet
from utils.dataset_config import get_dataset_config
from opts import arg_parser
import pickle
import torch.distributed as dist
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="Argument 'interpolation' of type int.*")
from visual import visual_a_batch

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
def eval_a_batch(data, model_degrad, model,model_type, num_clips=1, num_crops=1, threed_data=False):
    with torch.no_grad():
        data, bias = model_degrad(data)
        batch_size = data.shape[0]
        if threed_data:
            tmp = torch.chunk(data, num_clips * num_crops, dim=2)
            data = torch.cat(tmp, dim=0)
        else:
            data = data.view((batch_size * num_crops * num_clips, -1) + data.size()[2:])
        # output_budget = []
        # for r in range(data.size(2)):
        #     output_budget.append(model(data[:,:,r,:,:])) 
        # print(len(output_budget))
        if model_type == 'target':
            result = model(data)
        else:
            r = random.randint(0,data.size(2)-1)
            result = model(data[:,:,r,:,:])

        if threed_data:
            tmp = torch.chunk(result, num_clips * num_crops, dim=0)
            result = None
            for i in range(len(tmp)):
                result = result + tmp[i] if result is not None else tmp[i]
            result /= (num_clips * num_crops)
        else:
            result = result.reshape(batch_size, num_crops * num_clips, -1).mean(dim=1)

    return result


def main():
    global args
    parser = arg_parser()
    args = parser.parse_args()
    cudnn.benchmark = True
    set_seed()
    dist.init_process_group(
        backend='nccl',  # 或 'gloo'（如果无 GPU）
        init_method='tcp://localhost:12345',  # 任意可用端口
        rank=0,
        world_size=1
    )

    num_classes_target, num_classes_budget, train_list_name, val_list_name, test_list_name, filename_seperator, image_tmpl, filter_video, label_file = get_dataset_config(args.dataset)

    data_list_name = val_list_name if args.evaluate else test_list_name

    args.num_classes_target = num_classes_target
    args.num_classes_budget = num_classes_budget
    gpu = args.gpu
    num_classes = num_classes_target if args.model_type == 'target' else num_classes_budget

    if args.modality == 'rgb':
        args.input_channels = 3
    elif args.modality == 'flow':
        args.input_channels = 2 * 5

    

    models, arch_name = build_model(args, test_mode=True)

    
    model_degrad= models[0]
    if args.model_type == 'target':
        model = models[1]
    else:
        model = models[2]

    # arch_name= 'kinetics400-rgb-i3d-f64-multisteps-bs32-e160'
    arch_name = 'test_arch'

    norm_value= 255
    mean = [110.63666788 / norm_value, 103.16065604 / norm_value, 96.29023126 / norm_value]
    std = [38.7568578 / norm_value, 37.88248729 / norm_value, 40.02898126 / norm_value]

    gpu = args.gpu[0]
    torch.cuda.set_device(gpu)
    model_degrad = model_degrad.cuda()
    model_degrad.eval()
    model = model.cuda()
    model.eval()
    
    print("=> using pre-trained model '{}'".format(arch_name))
    checkpoint = torch.load(f'results/{args.dataset}_{args.bdq_v}/adv/model_degrad_new.ckpt', map_location='cpu')

    state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    model_degrad.load_state_dict(state_dict,strict = False)
    del checkpoint
    if args.model_type == 'target':
        checkpoint = torch.load(f'results/{args.dataset}_{args.bdq_v}/target/d152_model_target_epoch25_topT82.43.ckpt', map_location='cpu')
        # checkpoint = torch.load(f'results/{args.dataset}_{args.bdq_v}/alone/d50_model_target_epoch38_topT91.89.ckpt', map_location='cpu')
        # model.load_state_dict(checkpoint['state_dict'])
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        model.load_state_dict(state_dict) 
        del checkpoint
    else:
        checkpoint = torch.load(f'results/{args.dataset}_{args.bdq_v}/budget/d152_model_budget.ckpt', map_location='cpu')
        # checkpoint = torch.load(f'results/{args.dataset}_{args.bdq_v}/alone/d50_model_budget_epoch26_topT98.99.ckpt', map_location='cpu')
        # model.load_state_dict(checkpoint['state_dict'])
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        model.load_state_dict(state_dict)
        del checkpoint

    # augmentor
    if args.disable_scaleup:
        scale_size = args.input_size
    else:
        scale_size = int(args.input_size / 0.875 + 0.5)
    augments = []
    if args.num_crops == 1:
        augments += [
            GroupScale(scale_size),
            GroupCenterCrop(args.input_size)
        ]
    else:
        flip = True if args.num_crops == 10 else False
        augments += [
            GroupOverSample(args.input_size, scale_size, num_crops=args.num_crops, flip=flip),
        ]
    augments += [
        Stack(threed_data=args.threed_data),
        ToTorchFormatTensor(num_clips_crops=args.num_clips * args.num_crops),
        GroupNormalize(mean=mean, std=std, threed_data=args.threed_data)
    ]
    augmentor = transforms.Compose(augments)

    # Data loading code
    data_list = os.path.join(args.datadir, data_list_name)
    sample_offsets = list(range(-args.num_clips // 2 + 1, args.num_clips // 2 + 1))
    print("Image is scaled to {} and crop {}".format(scale_size, args.input_size))
    print("Number of crops: {}".format(args.num_crops))
    print("Number of clips: {}".format(args.num_clips))

    val_dataset = VideoDataSet(args.datadir, data_list, args.groups, args.frames_per_group,
                                 num_clips=args.num_clips, modality=args.modality,
                                 image_tmpl=image_tmpl, dense_sampling=args.dense_sampling,
                                 fixed_offset=not args.random_sampling,
                                 transform=augmentor, is_train=False, test_mode=False,
                                 seperator=filename_seperator, filter_video=filter_video)

    data_loader = build_dataflow(val_dataset, is_train=False, batch_size=args.batch_size,
                                 workers=args.workers)

    log_folder = os.path.join(args.logdir, arch_name)

    if not os.path.exists(log_folder):
        os.makedirs(log_folder)

    batch_time = AverageMeter()
    logfile = open(os.path.join(log_folder, 'evaluate_log.log'), 'a')
    top1 = AverageMeter()
    top5 = AverageMeter()

    total_outputs = 0
    outputs = np.zeros((len(data_loader) * args.batch_size, num_classes))
    # switch to evaluate mode

    total_batches = len(data_loader)
    count = 0
    with torch.no_grad(), tqdm(total=total_batches) as t_bar:
        end = time.time()
        # for i, (video, label) in enumerate(data_loader):
        for i, (video, target_actor, target_action) in enumerate(data_loader):
            label = target_action if args.model_type == 'target' else target_actor
            video = video.cuda(non_blocking=True)
            output = eval_a_batch(video, model_degrad, model, args.model_type, num_clips=args.num_clips, num_crops=args.num_crops, threed_data=args.threed_data)

            label = label.cuda(non_blocking=True)
            # measure accuracy
            prec1, prec5 = accuracy(output, label, topk=(1, 5))
            # if prec1[0].item() == 0:
            #     count += 1
            #     if count == 2:
            #         visual_a_batch(video,model_degrad)
            #         return
            top1.update(prec1[0], video.size(0))
            top5.update(prec5[0], video.size(0))
            output = output.data.cpu().numpy().copy()
            batch_size = output.shape[0]
            outputs[total_outputs:total_outputs + batch_size, :] = output
            
            total_outputs += video.shape[0]
            batch_time.update(time.time() - end)
            end = time.time()
            t_bar.update(1)

        outputs = outputs[:total_outputs]
        print("Predict {} videos.".format(total_outputs), flush=True)
        np.save(os.path.join(log_folder, '{}_{}crops_{}clips_{}_details.npy'.format(
            "val" if args.evaluate else "test", args.num_crops,
            args.num_clips, args.input_size)), outputs)


    print('Val@{}({}) (# crops = {}, # clips = {}): \tTop@1: {:.4f}\tTop@5: {:.4f}'.format(
        args.input_size, scale_size, args.num_crops, args.num_clips, top1.avg, top5.avg),
        flush=True)
    print('Val@{}({}) (# crops = {}, # clips = {}): \tTop@1: {:.4f}\tTop@5: {:.4f}'.format(
        args.input_size, scale_size, args.num_crops, args.num_clips, top1.avg, top5.avg),
        flush=True, file=logfile)

    logfile.close()


if __name__ == '__main__':
    main()
