#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=02:00:00
#SBATCH --partition gpu
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=liseloreborelrinkes@hotmail.com

#Loading modules
module load 2020
module load Python
module load CUDA/11.0.2-GCC-9.3.0

#Execute a Python program located in $HOME.
python $HOME/OpenNMT-py/train.py -train_from $HOME/saved_models/final_output/arb_model__step_200000.pt -config $HOME/config/final_arb_hy-ro_tune.yaml