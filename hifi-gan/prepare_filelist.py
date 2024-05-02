import os
import glob
from pymediainfo import MediaInfo

def get_duration(file_path):
    info = MediaInfo.parse(file_path)
    track = info.tracks[0]  
    return track.duration / 1000 # ms -> s


data_dir = "speech_data"
datasets = [
    "bc2021", "LJSpeech/wavs", "VCTK-Corpus/22050","IEMOCAP_full_release/22050"
]

val_datasets =[
    "bc2023_split"
]


training_file = "bc2023_training.txt"
validation_file = "bc2023_validation.txt"


count = 0
for dataset in datasets:
    with open(training_file, "w") as f:
        pattern = os.path.join(data_dir, dataset, "*.wav")
        files = glob.glob(pattern, recursive=True)
        for file in files:
            f.write(file.split('.')[0])
            f.write("\n")
            count +=1

print("Training data size:", count)

count = 0
for dataset in val_datasets:
    with open(validation_file, "w") as f:
        pattern = os.path.join(data_dir, dataset, "*.wav")
        files = glob.glob(pattern, recursive=True)
        for file in files:
            f.write(file.split('.')[0])
            f.write("\n")
            count +=1
print("Validation data size:", count)