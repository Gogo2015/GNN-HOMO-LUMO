#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 4
#SBATCH -t 08:00:00
#SBATCH --gres=gpu:V100:1
#SBATCH --mem-per-gpu=32G
#SBATCH -J gnn-virtual-node
#SBATCH -o logs/virtual_node_%j.out

mkdir -p logs

module load anaconda3/2022.05.0.1
conda activate ~/scratch/dl-netrunners

cd ~/scratch/DL-NetRunners

python3 -m src.train --config configs/virtual_node.yml
