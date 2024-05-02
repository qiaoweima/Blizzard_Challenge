import os

import librosa
import numpy as np
from scipy.io import wavfile
from tqdm import tqdm
import glob

def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    for dataset in ["train"]:
        print("Processing {}ing set...".format(dataset))
        dataset_dir = os.path.join(in_dir,dataset)
        wavs = glob.glob(os.path.join(dataset_dir,"*.wav"))
        for wav_file in wavs:
            wav_name = os.path.basename(wav_file).split(".")[0]
            speaker = "01"
            text = ""
            with open(os.path.join(dataset_dir,wav_name+".txt")) as f:
                text = f.readline()
            os.makedirs(os.path.join(out_dir, speaker),exist_ok=True)
            wav, _ = librosa.load(wav_file,sampling_rate)
            wav = wav / max(abs(wav)) * max_wav_value
            wavfile.write(
                os.path.join(out_dir,speaker,wav_name + '.wav'),
                sampling_rate,
                wav.astype(np.int16)
            )
            with open(
                os.path.join(out_dir, speaker, "{}.lab".format(wav_name)),
                "w",
            ) as f1:
                f1.write(text)

