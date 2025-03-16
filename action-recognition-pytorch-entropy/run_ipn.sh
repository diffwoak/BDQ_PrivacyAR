#!/bin/bash

# IPN数据集
python train.py --multiprocessing-distributed --datadir /root/autodl-tmp/IPN_splits --dataset IPN --groups 20 --dense_sampling --weight 8 --without_t_stride
# 固定BDQ 训练动作识别网络
python train_target.py --multiprocessing-distributed --datadir /root/autodl-tmp/IPN_splits --dataset IPN --groups 20 --dense_sampling --weight 8 --without_t_stride
# 固定BDQ 训练隐私识别网络
python train_budget.py --multiprocessing-distributed --datadir /root/autodl-tmp/IPN_splits --dataset IPN --groups 20 --dense_sampling --weight 8 --without_t_stride
## 测试动作识别
python test.py --datadir /root/autodl-tmp/IPN_splits --dataset IPN --groups 20 --dense_sampling --disable_scaleup --model_type target --gpu 0 -b 1
## 测试隐私识别
python test.py --datadir /root/autodl-tmp/IPN_splits --dataset IPN --groups 20 --dense_sampling --disable_scaleup --model_type budget --gpu 0 -b 1