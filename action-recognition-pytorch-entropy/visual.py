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
from torchvision import transforms
 


def eval_a_batch(data, model, num_clips=1, num_crops=1, threed_data=False):
    with torch.no_grad():
        batch_size = data.shape[0]
        if threed_data:
            tmp = torch.chunk(data, num_clips * num_crops, dim=2)
            data = torch.cat(tmp, dim=0)
        else:
            data = data.view((batch_size * num_crops * num_clips, -1) + data.size()[2:])
        result = model(data)

        if threed_data:
            tmp = torch.chunk(result, num_clips * num_crops, dim=0)
            result = None
            for i in range(len(tmp)):
                result = result + tmp[i] if result is not None else tmp[i]
            result /= (num_clips * num_crops)
        else:
            result = result.reshape(batch_size, num_crops * num_clips, -1).mean(dim=1)
    return result

# def visual_a_batch(data, model):

#     # data : torch.Size([1, 48, 480, 640])
#     # output_dir = 'visual91notrain'
#     output_dir = 'visual0'

#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     with torch.no_grad():
#         B, _, C_T, H, W= data.shape
#         num_images= C_T//3
#         result, bias = model(data)   

#         # result.shape : torch.Size([1, 3, 15, 480, 640])
#         toPIL = transforms.ToPILImage() #这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
#         for i in range(num_images):
#             image = data[:,i*3:(i+1)*3,:,:]
#             image = torch.squeeze(image)
#             pic = toPIL(image)
#             pic.save(f'{output_dir}/origin_{i}.png')

#         for i in range(result.size(2)):
#             image = result[:,:,i,:,:]
#             image = torch.squeeze(image)
#             pic = toPIL(image)
#             pic.save(f'{output_dir}/anony_{i}.png')

# def visual_a_batch(data, model):
#     output_dir = 'visual0'
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     with torch.no_grad():
#         # 正确分解五维数据的维度
#         B, C, T, H, W = data.shape  # B=batch, C=channel, T=image_nums
#         # num_images_per_time = C // 3  # 每个时间步拆分为多少个3通道的图像
        
#         result, bias = model(data)
        
#         toPIL = transforms.ToPILImage()  # 确保正确导入

#         # 处理输入数据
#         for t in range(T):
#             image = data[:, :, t, :, :]
#             # 去除 batch 维度（假设 B=1）
#             image = image.squeeze(0)
#             # 保存为文件名包含时间步和通道索引
#             pic = toPIL(image)
#             pic.save(f"{output_dir}/origin_t{t}.png")

#         # 处理模型输出（假设 result 的形状为 [B, 3, T, H, W]）
#         for t in range(result.size(2)):
#             image = result[:, :, t, :, :]
#             image = image.squeeze(0)
#             pic = toPIL(image)
#             pic.save(f"{output_dir}/anony_t{t}.png")

def visual_a_batch(data, model):
    output_dir = 'visual0'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with torch.no_grad():
        # 获取数据维度
        B, C, T_data, H, W = data.shape
        _, _, T_result, _, _ = model(data)[0].shape  # 假设 result 的维度是 [B, C, T_result, H, W]
        result, bias = model(data)
        blur_frames = model.blur_output    # 模糊后的帧 [B,C,T-1,H,W]
        diff_frames = model.diff_output    # 差分帧 [B,C,T-1,H,W]
        steps = len(model.time_steps)

        toPIL = transforms.ToPILImage()

        for batch_idx in range(B):
            # --- 收集 data 的后15张图像（假设 data 的时间步 T_data=16）---
            data_images = []
            for t in range(steps, 16):  # 时间步 1~15（共15张）
                # 提取单个图像：形状 [C, H, W]
                image = data[batch_idx, :, t, :, :]
                # 转换为 PIL 图像
                pil_image = toPIL(image)
                data_images.append(pil_image)

            # --- 收集 result 的15张图像（假设 result 的时间步 T_result=15）---
            result_images = []
            blur_images = []
            diff_images = []
            for t in range(T_result):  # 时间步 0~14（共15张）
                image_b = blur_frames[batch_idx, :, t, :, :]
                image_d = diff_frames[batch_idx, :, t, :, :]
                image = result[batch_idx, :, t, :, :]
                pil_image_b = toPIL(image_b)
                pil_image_d = toPIL(image_d)
                pil_image = toPIL(image)
                blur_images.append(pil_image_b)
                diff_images.append(pil_image_d)
                result_images.append(pil_image)

            # --- 合并图像 ---
            # 计算合并后的图像尺寸：宽度为 15*W，高度为 2*H
            total_width = (16-steps) * W
            total_height = 4 * H

            # 创建空白画布
            combined = Image.new('RGB', (total_width, total_height))

            # 粘贴 data 的图像到第一行
            x_offset = 0
            for img in data_images:
                combined.paste(img, (x_offset, 0))
                x_offset += img.width
            
            # 粘贴 blur_frames 的图像到第二行
            x_offset = 0
            for img in blur_images:
                combined.paste(img, (x_offset, H))
                x_offset += img.width
            
            # 粘贴 diff_frames 的图像到第三行
            x_offset = 0
            for img in diff_images:
                combined.paste(img, (x_offset, 2*H))
                x_offset += img.width

            # 粘贴 result 的图像到第四行
            x_offset = 0
            for img in result_images:
                combined.paste(img, (x_offset, 3*H))
                x_offset += img.width

            # 保存合并后的图像
            combined.save(os.path.join(output_dir, f"combined_batch{batch_idx}.png"))

def main():
    global args
    parser = arg_parser()
    args = parser.parse_args()
    cudnn.benchmark = True

    # num_classes, train_list_name, val_list_name, test_list_name, filename_seperator, image_tmpl, filter_video, label_file = get_dataset_config(args.dataset)
    num_classes_target, num_classes_budget, train_list_name, val_list_name, test_list_name, filename_seperator, image_tmpl, filter_video, label_file = get_dataset_config(args.dataset)
    args.num_classes_target = num_classes_target
    args.num_classes_budget = num_classes_budget
    data_list_name = val_list_name

    # args.num_classes = num_classes
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu[0])
    if args.modality == 'rgb':
        args.input_channels = 3
    elif args.modality == 'flow':
        args.input_channels = 2 * 5
 
    # args.backbone_net = 'i3dBDQ'

    model, arch_name = build_model(args, test_mode=True)
    # print(model[0])
    # from pdb import set_trace
    # set_trace()
    model_degrad = model[0]

    # 加载权重

    # anonymizer_path = 'results/Origin_1e_3_BDQ_alpha8__pretrain/model_degrad_0.ckpt'
    # state_dict = torch.load(anonymizer_path)
    # model_degrad.load_state_dict(state_dict['state_dict'])
    # print(torch.load(anonymizer_path)['state_dict'])
    checkpoint = torch.load(f'results/{args.dataset}/adv/model_degrad.ckpt', map_location='cpu')
    state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
    model_degrad.load_state_dict(state_dict)
    del checkpoint

    model_degrad = model_degrad.cuda()
    
    norm_value= 255
    mean = mean=[110.63666788 / norm_value, 103.16065604 / norm_value, 96.29023126 / norm_value]
    std=[38.7568578 / norm_value, 37.88248729 / norm_value, 40.02898126 / norm_value]


    # model.to(torch.device('cuda:0'))
    # model = torch.nn.DataParallel(model).cuda()
    model_degrad.eval()

    # augmentor
    if args.disable_scaleup:
        scale_size = args.input_size
    else:
        scale_size = int(args.input_size / 0.875 + 0.5)

    augments = []
    if args.num_crops == 1:
        augments += [
            # GroupScale(scale_size),
            # GroupCenterCrop(args.input_size)
        ]
    else:
        flip = True if args.num_crops == 10 else False
        augments += [
            GroupOverSample(args.input_size, scale_size, num_crops=args.num_crops, flip=flip),
        ]
    augments += [
        Stack(threed_data=args.threed_data),
        ToTorchFormatTensor(num_clips_crops=args.num_clips * args.num_crops),
        # GroupNormalize(mean=mean, std=std, threed_data=args.threed_data)
    ]
    augmentor = transforms.Compose(augments)
    # print(augmentor)

    # Data loading code
    data_list = os.path.join(args.datadir, data_list_name)
    # sample_offsets = list(range(-args.num_clips // 2 + 1, args.num_clips // 2 + 1))

    val_dataset = VideoDataSet(args.datadir, data_list, args.groups, args.frames_per_group,
                                 num_clips=args.num_clips, modality=args.modality,
                                 image_tmpl=image_tmpl, dense_sampling=args.dense_sampling,
                                #  fixed_offset=not args.random_sampling,
                                 transform=augmentor, is_train=False, test_mode=False,
                                 seperator=filename_seperator, filter_video=filter_video)

    data_loader = build_dataflow(val_dataset, is_train=False, batch_size=args.batch_size,
                                 workers=args.workers)

    # outputs = np.zeros((len(data_loader) * args.batch_size, num_classes))
    # switch to evaluate mode

    with torch.no_grad():
        data_list = list(data_loader)
        # 直接通过下标访问
        video, _, _ = data_list[0]  # index 是你想访问的下标
        video = video.cuda(non_blocking=True)
        visual_a_batch(video, model_degrad)


if __name__ == '__main__':
    main()
