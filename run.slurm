#!/bin/bash
#SBATCH --job-name=hubert_disfl
#SBATCH --open-mode=append
#SBATCH --output=/scratch/dsr331/%j_%x.out
#SBATCH --error=/scratch/dsr331/%j_%x.err
#SBATCH --export=ALL
#SBATCH --account=ds_ga_1012_2022sp
#SBATCH --partition=n1s8-v100-1
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH -c 4
#SBATCH --requeue

## TODO: Run your hyperparameter search in a singularity container. Remember to 
## activate your Conda environment in the container before running your Python script.

singularity exec --nv --bind /scratch --overlay /scratch/dsr331/overlay-25GB-500K.ext3:ro /scratch/dsr331/cuda11.4.2-cudnn8.2.4-devel-ubuntu20.04.3.sif /bin/bash -c "
source /ext3/env.sh
conda activate
cd /home/dsr331/projects/hubert-disfl
source setup.sh
sh run.sh
"

