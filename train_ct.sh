#!/bin/sh 

#SBATCH -p gpu

#SBATCH -o test_Ct_out.log

#SBATCH -e test_Ct_err.log

#SBATCH -J optimize

#SBATCH -N 1

#SBATCH --gpus=1
 
python -m torch.distributed.launch --nproc_per_node=1 /data/home/scyb093/GF/MedSAM/CT_Train.py

