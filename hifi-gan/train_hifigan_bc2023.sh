#!/bin/bash

input_training_file=finetune_train.txt
input_validation_file=finetune_val.txt
export CUDA_VISIBLE_DEVICES='1,2'
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

python train.py --config config_v4.json \
    --input_wavs_dir ./ \
    --input_training_file $input_training_file \
    --input_validation_file $input_validation_file \
    --fine_tuning true\
    --input_mels_dir /speechgroup/bc2023_hifigan_finetune_mel/bc2023_var_conv_finetune

    
