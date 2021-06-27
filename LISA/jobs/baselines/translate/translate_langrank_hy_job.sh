#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:10:00
#SBATCH --partition gpu_short

#Loading modules
module load 2020
module load Python
module load CUDA/11.0.2-GCC-9.3.0

#Execute a Python program located in $HOME.
python $HOME/OpenNMT-py/translate.py -model $HOME/saved_models/final_output/hy-bh_hy_model_step_37000.pt -src $HOME/bpe_data/final_data/test_data/final_hy_test_bpe.txt -output $HOME/bpe_data/final_data/pred_data/pred_tune_bg_hy_test_37000.txt -verbose -replace_unk -gpu 0

sed -i "s/@@ //g"  $HOME/bpe_data/final_data/pred_data/pred_tune_bg_hy_test_37000.txt

perl  $HOME/OpenNMT-py/tools/multi-bleu.perl $HOME/bpe_data/final_data/test_data/hy_en_test.txt < $HOME/bpe_data/final_data/pred_data/pred_tune_bg_hy_test_37000.txt