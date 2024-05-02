preprocess_config=config/bc2023_var_decoder/preprocess.yaml
model_config=config/bc2023_var_decoder_ft/model.yaml
train_config=config/bc2023_var_decoder_ft/train.yaml
export CUDA_VISIBLE_DEVICES=2

python synthesis_mel.py -p $preprocess_config \
                -m $model_config \
                -t $train_config \
                --restore_step 240000
