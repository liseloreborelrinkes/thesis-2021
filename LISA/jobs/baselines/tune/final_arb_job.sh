#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=100:00:00
#SBATCH --partition gpu
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=liseloreborelrinkes@hotmail.com

#Loading modules
module load 2020
module load Python
module load CUDA/11.0.2-GCC-9.3.0

#Execute a Python program located in $HOME.
python $HOME/OpenNMT-py/train.py -config $HOME/config/final_arb_train.yaml