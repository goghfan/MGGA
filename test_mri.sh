#!/bin/sh 

#SBATCH -p gpu

#SBATCH -o test_Ct_out.log

#SBATCH -e test_ct_err.log

#SBATCH -J optimize

#SBATCH -N 1

#SBATCH --gpus=1

# 设置新的MASTER_PORT环境变量，确保端口没有被占用
export MASTER_PORT=29507

torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=29507 /data/home/scyb093/GF/MedSAM/Registration_MRI.py

