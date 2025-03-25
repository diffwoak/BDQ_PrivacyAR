#!/bin/bash

# 分布式训练环境(默认SBU数据集的配置)
python train.py --multiprocessing-distributed --datadir /root/autodl-tmp/SBU_splits --dense_sampling --without_t_stride --lr 0.003 --weight 6 -b 8
# 固定BDQ 训练动作识别网络
python train_target.py --multiprocessing-distributed --datadir /root/autodl-tmp/SBU_splits --dense_sampling --without_t_stride --lr 0.003 --weight 6 -b 8
# 固定BDQ 训练隐私识别网络
python train_budget.py --multiprocessing-distributed --datadir /root/autodl-tmp/SBU_splits --dense_sampling --without_t_stride --lr 0.003 --weight 6 -b 8
## 测试动作识别
python test.py --datadir /root/autodl-tmp/SBU_splits --dense_sampling --model_type target  --gpu 0 -b 8 --without_t_stride
## 测试隐私识别
python test.py --datadir /root/autodl-tmp/SBU_splits --dense_sampling --model_type budget  --gpu 0 -b 8 --without_t_stride