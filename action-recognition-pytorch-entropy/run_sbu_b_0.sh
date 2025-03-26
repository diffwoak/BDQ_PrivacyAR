#!/bin/bash

python train_budget.py --multiprocessing-distributed --datadir /root/autodl-tmp/SBU_splits --dense_sampling --without_t_stride --lr 0.003 --weight 8 -b 8 --bdq_v v3 --abla B  --gpu 0 --dist-url tcp://127.0.0.1:23455
python train_budget.py --multiprocessing-distributed --datadir /root/autodl-tmp/SBU_splits --dense_sampling --without_t_stride --lr 0.003 --weight 8 -b 8 --bdq_v v3 --abla D --gpu 0 --dist-url tcp://127.0.0.1:23455
python train_budget.py --multiprocessing-distributed --datadir /root/autodl-tmp/SBU_splits --dense_sampling --without_t_stride --lr 0.003 --weight 8 -b 8 --bdq_v v3 --abla D S --gpu 0 --dist-url tcp://127.0.0.1:23455
python train_budget.py --multiprocessing-distributed --datadir /root/autodl-tmp/SBU_splits --dense_sampling --without_t_stride --lr 0.003 --weight 8 -b 8 --bdq_v v3 --abla Q --gpu 0 --dist-url tcp://127.0.0.1:23455
