# text="L'aéronef fit un crochet à droite pour éviter les hautes tours de l'Observatoire et de la grande usine électrique du mont Valérien, puis d'un seul bond au-dessus du quartier industriel de Nanterre, elle arriva au tournant de la Seine."
source="preprocessed_data/bc2023/bc2023_test/testing.txt"
testing_dir="preprocessed_data/bc2023/bc2023_test"
export CUDA_VISIBLE_DEVICES=0


python synthesize.py \
        -p config/bc2023_var_decoder_ft/preprocess.yaml \
        -m config/bc2023_var_decoder_ft/model.yaml \
        -t config/bc2023_var_decoder_ft/train.yaml \
        --mode batch \
        --testing_dir $testing_dir \
        --source $source \
        --restore_step 440000


