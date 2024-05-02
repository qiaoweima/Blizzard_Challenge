import librosa
from scipy.io import wavfile
import numpy
import os

corpus_dir = "speech_data/bc2023"
save_dir = "speech_data/bc2023_split"
transcript = "NEB_train.csv"

sampling_rate = 22050

os.makedirs(save_dir, exist_ok=True)

with open(os.path.join(corpus_dir, transcript)) as f:
    lines = f.readlines()
    file_name = ""
    audio = None
    count = 0
    prev_frame = -1
    for i, line in enumerate(lines):
        name, start, end = line.split('|')[0:3]
        start = int(float(start) /1000. * sampling_rate)
        end = int(float(end) / 1000. * sampling_rate)
        if file_name != name:
            count = 0
            file_name = name
            prev_frame = -1
            sr, audio = wavfile.read(os.path.join(corpus_dir, name + ".wav"))
            if not sr == sampling_rate:
                raise ValueError("Sampling Rate don't match!", sr)
        if start >= prev_frame:
            wavfile.write(os.path.join(save_dir, "%s_%d.wav" %(file_name, count)), 
                        sampling_rate, audio[start:end + 1])
            prev_frame = end

            count += 1            


