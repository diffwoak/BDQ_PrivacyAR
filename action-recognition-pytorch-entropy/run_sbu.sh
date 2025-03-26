#!/bin/bash

# 分布式训练环境(默认SBU数据集的配置)
# python train.py --multiprocessing-distributed --datadir /root/autodl-tmp/SBU_splits --dense_sampling --without_t_stride --lr 0.003 --weight 6 -b 8 --bdq_v v3
# 固定BDQ 训练动作识别网络
python train_target.py --multiprocessing-distributed --datadir /root/autodl-tmp/SBU_splits --dense_sampling --without_t_stride --lr 0.003 --weight 6 -b 8 --bdq_v v3 --resume 6_model_degrad_epoch42_topT83.78_topB35.77_D48.02
# # 固定BDQ 训练隐私识别网络
# ## 测试动作识别
# python test.py --datadir /root/autodl-tmp/SBU_splits --dense_sampling --model_type target  --gpu 0 -b 8 --without_t_stride
# ## 测试隐私识别
# python test.py --datadir /root/autodl-tmp/SBU_splits --dense_sampling --model_type budget  --gpu 0 -b 8 --without_t_stride

# 固定BDQ 训练动作识别网络
python train_target.py --multiprocessing-distributed --datadir /root/autodl-tmp/SBU_splits --dense_sampling --without_t_stride --lr 0.003 --weight 4 -b 8 --bdq_v v3 --resume 4_model_degrad_epoch48_topT86.49_topB26.67_D59.82

# 固定BDQ 训练动作识别网络
python train_target.py --multiprocessing-distributed --datadir /root/autodl-tmp/SBU_splits --dense_sampling --without_t_stride --lr 0.003 --weight 10 -b 8 --bdq_v v3 --resume 10_model_degrad
