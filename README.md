# BDQ_PrivacyAR

## 环境配置


```
conda create -n bdq python=3.9
conda env list
source activate bdq
```

```
# cuda:11.4没有对应的版本，但可以向下兼容，所以选择cuda=11.3的版本安装cudatoolkit、torch
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
pip install tensorboard_logger
pip install tensorflow==2.12.0
pip install tqdm
pip install scikit-video
pip install matplotlib
pip install opencv-python
pip install imageio==2.9.0
pip install numpy==1.23
```

## 查看GPU信息

```
nvidia-smi

echo "USER | PID | GPU_MEM(MiB) | GPU_ID | PROCESS | ELAPSED_TIME" && nvidia-smi --query-compute-apps=pid,used_gpu_memory,gpu_bus_id --format=csv,noheader | awk -F', ' '{print $1,$2,$3}' | while read -r pid mem gpu_bus; do ps -o user=,cmd=,etime= -p $pid | awk -v pid="$pid" -v mem="$mem" -v gpu_bus="$gpu_bus" '{print $1 " | " pid " | " mem " | " gpu_bus " | " $2 " | " $3}'; done

kill [PID]
```

```
conda env list
source activate bdq
```

## 快速使用

```
cd BDQ_PrivacyAR/action-recognition-pytorch-entropy
cd /root/project/BDQ_PrivacyAR/action-recognition-pytorch-entropy
# 分布式训练环境(默认SBU数据集的配置)
python train.py --multiprocessing-distributed --datadir /data/chenxinyu/data/SBU_splits --dense_sampling --without_t_stride
# 固定BDQ 训练动作识别网络
python train_target.py --multiprocessing-distributed --datadir /data/chenxinyu/data/SBU_splits --dense_sampling --without_t_stride
# 固定BDQ 训练隐私识别网络
python train_budget.py --multiprocessing-distributed --datadir /data/chenxinyu/data/SBU_splits --dense_sampling --without_t_stride
## 测试动作识别
python test.py --datadir /data/chenxinyu/data/SBU_splits --dense_sampling --model_type target  --gpu 0 -b 1 --without_t_stride
## 测试隐私识别
python test.py --datadir /data/chenxinyu/data/SBU_splits --dense_sampling --model_type budget  --gpu 0 -b 1 --without_t_stride

# KTH数据集
python train.py --multiprocessing-distributed --datadir /data/chenxinyu/data/KTH_splits --dataset KTH --groups 32 --dense_sampling --disable_scaleup --weight 1 --without_t_stride --augmentor_ver v2
# 固定BDQ 训练动作识别网络
python train_target.py --multiprocessing-distributed --datadir /data/chenxinyu/data/KTH_splits --dataset KTH --groups 32 --dense_sampling --disable_scaleup --weight 1 --without_t_stride --augmentor_ver v2
# 固定BDQ 训练隐私识别网络
python train_budget.py --multiprocessing-distributed --datadir /data/chenxinyu/data/KTH_splits --dataset KTH --groups 32 --dense_sampling --disable_scaleup --weight 1 --without_t_stride --augmentor_ver v2
## 测试动作识别
python test.py --datadir /data/chenxinyu/data/KTH_splits --dataset KTH --groups 32 --dense_sampling --disable_scaleup --model_type target --gpu 0 -b 1
## 测试隐私识别
python test.py --datadir /data/chenxinyu/data/KTH_splits --dataset KTH --groups 32 --dense_sampling --disable_scaleup --model_type budget --gpu 0 -b 1

# IPN数据集
python train.py --multiprocessing-distributed --datadir /data/chenxinyu/data/IPN_splits --dataset IPN --groups 32 --dense_sampling --weight 8 --without_t_stride
# 固定BDQ 训练动作识别网络
python train_target.py --multiprocessing-distributed --datadir /data/chenxinyu/data/IPN_splits --dataset IPN --groups 32 --dense_sampling --weight 8 --without_t_stride
# 固定BDQ 训练隐私识别网络
python train_budget.py --multiprocessing-distributed --datadir /data/chenxinyu/data/IPN_splits --dataset IPN --groups 32 --dense_sampling --weight 8 --without_t_stride
## 测试动作识别
python test.py --datadir /data/chenxinyu/data/IPN_splits --dataset IPN --groups 32 --dense_sampling --disable_scaleup --model_type target --gpu 0 -b 1
## 测试隐私识别
python test.py --datadir /data/chenxinyu/data/IPN_splits --dataset IPN --groups 32 --dense_sampling --disable_scaleup --model_type budget --gpu 0 -b 1

```


## 远程服务器版本

```
cd BDQ_PrivacyAR/action-recognition-pytorch-entropy

# 分布式训练环境(默认SBU数据集的配置)
python train.py --multiprocessing-distributed --datadir /root/autodl-tmp/SBU_splits --dense_sampling --without_t_stride --lr 0.003 --weight 8 -b 8
# 固定BDQ 训练动作识别网络
python train_target.py --multiprocessing-distributed --datadir /root/autodl-tmp/SBU_splits --dense_sampling --without_t_stride --lr 0.003 --weight 8 -b 8
# 固定BDQ 训练隐私识别网络
python train_budget.py --multiprocessing-distributed --datadir /root/autodl-tmp/SBU_splits --dense_sampling --without_t_stride --lr 0.003 --weight 8 -b 8
## 测试动作识别
python test.py --datadir /root/autodl-tmp/SBU_splits --dense_sampling --model_type target  --gpu 0 -b 8 --without_t_stride
## 测试隐私识别
python test.py --datadir /root/autodl-tmp/SBU_splits --dense_sampling --model_type budget  --gpu 0 -b 8 --without_t_stride

# KTH数据集
python train.py --multiprocessing-distributed --datadir /root/autodl-tmp/KTH_splits --dataset KTH --without_t_stride --groups 32 --dense_sampling --disable_scaleup --weight 2 --without_t_stride --lr 0.003 -b 8 --epochs 13
# 固定BDQ 训练动作识别网络
python train_target.py --multiprocessing-distributed --datadir /root/autodl-tmp/KTH_splits --dataset KTH --without_t_stride --groups 32 --dense_sampling --disable_scaleup --weight 2 --without_t_stride --lr 0.003 -b 8 --epochs 13
# 固定BDQ 训练隐私识别网络
python train_budget.py --multiprocessing-distributed --datadir /root/autodl-tmp/KTH_splits --dataset KTH --without_t_stride --groups 32 --dense_sampling --disable_scaleup --weight 2 --without_t_stride --lr 0.003 -b 8 --epochs 13
## 测试动作识别
python test.py --datadir /root/autodl-tmp/KTH_splits --dataset KTH --groups 32 --without_t_stride --dense_sampling --disable_scaleup --model_type target --gpu 0 -b 1
## 测试隐私识别
python test.py --datadir /root/autodl-tmp/KTH_splits --dataset KTH --groups 32 --without_t_stride --dense_sampling --disable_scaleup --model_type budget --gpu 0 -b 1

# IPN数据集
python train.py --multiprocessing-distributed --datadir /root/autodl-tmp/IPN_splits --dataset IPN --groups 32 --dense_sampling --weight 20 --without_t_stride -b 8 --lr 0.003
# 固定BDQ 训练动作识别网络
python train_target.py --multiprocessing-distributed --datadir /root/autodl-tmp/IPN_splits --dataset IPN --groups 32 --dense_sampling --weight 8 --without_t_stride -b 8 --lr 0.003
# 固定BDQ 训练隐私识别网络
python train_budget.py --multiprocessing-distributed --datadir /root/autodl-tmp/IPN_splits --dataset IPN --groups 32 --dense_sampling --weight 8 --without_t_stride -b 8 --lr 0.003
## 测试动作识别
python test.py --datadir /root/autodl-tmp/IPN_splits --dataset IPN --groups 32 --dense_sampling --without_t_stride --disable_scaleup --model_type target --gpu 0 -b 8
## 测试隐私识别
python test.py --datadir /root/autodl-tmp/IPN_splits --dataset IPN --groups 32 --dense_sampling --without_t_stride --disable_scaleup --model_type budget --gpu 0 -b 8

```

## 可视化

```
python visual.py --datadir /data/chenxinyu/data/SBU_splits --dense_sampling --without_t_stride --gpu 1 -b 4

python visual.py --datadir /root/autodl-tmp/SBU_splits --dense_sampling --without_t_stride --gpu 0 -b 4

python visual.py --datadir /root/autodl-tmp/IPN_splits --dataset IPN --groups 16 --dense_sampling --without_t_stride --gpu 0 -b 4
```

# 消融实验

```
# 固定BDQ 训练动作识别网络
python train_target.py --multiprocessing-distributed --datadir /root/autodl-tmp/SBU_splits --dense_sampling --without_t_stride --lr 0.003 --weight 8 -b 8 --bdq_v v3 --abla B D S Q
# 固定BDQ 训练隐私识别网络
python train_budget.py --multiprocessing-distributed --datadir /root/autodl-tmp/SBU_splits --dense_sampling --without_t_stride --lr 0.003 --weight 8 -b 8 --bdq_v v3 --abla B D S Q
## 测试动作识别
python test.py --datadir /root/autodl-tmp/SBU_splits --groups 16 --dense_sampling --disable_scaleup --model_type target --gpu 1 -b 8 --bdq_v v3 --abla B D S Q
## 测试隐私识别
python test.py --datadir /root/autodl-tmp/SBU_splits --groups 16 --dense_sampling --disable_scaleup --model_type budget --gpu 1 -b 8 --bdq_v v3 --abla B D S Q
```
