from meldataset import mel_spectrogram
import numpy as np
import torch
import os


save_mel_dir = "speech_data_mel_orig"
os.makedirs(save_mel_dir, exist_ok=True)

with open("bc2023_training.txt") as f:
    files = f.readlines()
    pass