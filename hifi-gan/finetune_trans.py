import os

mel_path = "/speechgroup/bc2023_hifigan_finetune_mel/bc2023_var_conv_finetune"

mels = os.listdir(mel_path)
wav_path = "/speechgroup/bc2023_hifigan_finetune_mel/wav/trim_wav/"


train_mels = mels[:-200]
val_mels = mels[-200:]

with open("finetune_train.txt",'w') as f:
    for mel in train_mels:
        mel = mel.split('.')[0]
        f.write(os.path.join(wav_path, mel) + '\n')

with open("finetune_val.txt", 'w') as f:
    for mel in val_mels:
        mel = mel.split('.')[0]
        f.write(os.path.join(wav_path, mel) + '\n')