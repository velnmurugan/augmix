#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 6:00:00
#SBATCH -o /work/ws-tmp/g058978-augmix_ws/augmix/logs/cluster.%j.%x.log
#SBATCH -c 8
#SBATCH --gres gpu:1

# Make conda available:
eval "$(conda shell.bash hook)"
# Activate a conda environment:
conda activate myenv


#python cifar_original.py -m resnet18_npt -lr 0.0001 -e 2
python cifar.py -m resnet18_npt -lr 0.0001 -e 2
python cifar.py -m resnet18_pt -lr 0.0001 -e 2
python cifar.py -m convnext_npt -lr 0.0001 -e 2
python cifar.py -m convnext_pt -lr 0.0001 -e 2

