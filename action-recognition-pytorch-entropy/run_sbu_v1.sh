#!/bin/bash

python train_budget.py --multiprocessing-distributed --datadir /root/autodl-tmp/SBU_splits --dense_sampling --without_t_stride --lr 0.003 --weight 6 -b 8 --bdq_v v3 --resume 6_model_degrad_epoch42_topT83.78_topB35.77_D48.02
python train_budget.py --multiprocessing-distributed --datadir /root/autodl-tmp/SBU_splits --dense_sampling --without_t_stride --lr 0.003 --weight 4 -b 8 --bdq_v v3 --resume 4_model_degrad_epoch48_topT86.49_topB26.67_D59.82
