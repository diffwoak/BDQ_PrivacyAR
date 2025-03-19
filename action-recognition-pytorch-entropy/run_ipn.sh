#!/bin/bash

# IPN数据集
python train.py --multiprocessing-distributed --datadir /root/autodl-tmp/IPN_splits --dataset IPN --groups 36 --dense_sampling --weight 32 --without_t_stride --lr 0.006 --epochs 8
# 固定BDQ 训练动作识别网络
python train_target.py --multiprocessing-distributed --datadir /root/autodl-tmp/IPN_splits --dataset IPN --groups 36 --dense_sampling --weight 32 --without_t_stride --lr 0.006 --epochs 8
# 固定BDQ 训练隐私识别网络
python train_budget.py --multiprocessing-distributed --datadir /root/autodl-tmp/IPN_splits --dataset IPN --groups 36 --dense_sampling --weight 32 --without_t_stride --lr 0.006 --epochs 8
## 测试动作识别
python test.py --datadir /root/autodl-tmp/IPN_splits --dataset IPN --groups 36 --dense_sampling --disable_scaleup --model_type target --gpu 0 -b 1
## 测试隐私识别
python test.py --datadir /root/autodl-tmp/IPN_splits --dataset IPN --groups 36 --dense_sampling --disable_scaleup --model_type budget --gpu 0 -b 1