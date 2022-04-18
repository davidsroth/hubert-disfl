#!/bin/bash
#SBATCH --job-name=hubert_disfl
#SBATCH --open-mode=append
#SBATCH --output=/scratch/%u/projects/hubert-disfl/logs/%x_%j.out
#SBATCH --error=/scratch/%u/projects/hubert-disfl/logs/%x_%j.err
#SBATCH --export=ALL
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH -c 8

singularity exec --nv --bind /scratch --overlay /scratch/$(whoami)/images/overlay-25GB-500K.ext3:ro /scratch/$(whoami)/images/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "
source /ext3/env.sh
conda activate
cd /home/$(whoami)/scratch/projects/hubert-disfl
source setup.sh
sh distributed_run.sh
"