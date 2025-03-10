# BDQ_PrivacyAR

## 环境配置

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

### 我的
```
cd project/BDQ_PrivacyAR/action-recognition-pytorch-entropy
cd BDQ_PrivacyAR/action-recognition-pytorch-entropy
python train.py --datadir data/SBU_splits
python train.py --multiprocessing-distributed --datadir data/SBU_splits

# 密集采样
python train.py --multiprocessing-distributed --datadir data/SBU_splits --dense_sampling

# 使用特定gpu
python train.py --multiprocessing-distributed --gpu 0 1 3 --datadir data/SBU_splits --dense_sampling

# KTH数据集
python train.py --multiprocessing-distributed --datadir data/KTH_splits --dataset KTH --groups 32 --dense_sampling --disable_scaleup --weight 1
python test.py --datadir data/KTH_splits --dataset KTH --groups 16 --dense_sampling --disable_scaleup --model_type target  --gpu 1 -b 1
python test.py --datadir data/KTH_splits --dataset KTH --groups 32 --dense_sampling --disable_scaleup --model_type budget

# IPN数据集
python train.py --multiprocessing-distributed --datadir data/IPN_splits --dataset IPN --groups 32 --dense_sampling --weight 8


```

### 测试
```
python test.py --datadir data/SBU_splits --multiprocessing-distributed --gpu 2
```

