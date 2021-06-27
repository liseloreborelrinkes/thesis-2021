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
python $HOME/OpenNMT-py/translate.py -model $HOME/saved_models/final_output/lt_model_step_9000.pt -src $HOME/bpe_data/final_data/regular_data/lt_src_test_bpe.txt -output $HOME/bpe_data/final_data/pred_data/pred_regular_lt_test_9000.txt -verbose -replace_unk -gpu 0

sed -i "s/@@ //g"  $HOME/bpe_data/final_data/pred_data/pred_regular_lt_test_9000.txt

perl  $HOME/OpenNMT-py/tools/multi-bleu.perl $HOME/bpe_data/final_data/regular_data/lt_tgt_test_bpe.txt < $HOME/bpe_data/final_data/pred_data/pred_regular_lt_test_9000.txt