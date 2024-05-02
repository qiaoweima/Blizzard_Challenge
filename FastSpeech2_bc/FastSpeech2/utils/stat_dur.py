import os
import wave

def get_wave_duration(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)
    return duration, frames

def seconds_to_hours(seconds):
    hours = seconds / 3600
    return hours

def get_wave_files_info(directory):
    wave_files_info = {}
    
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            file_path = os.path.join(directory, filename)
            duration, frames = get_wave_duration(file_path)
            # duration_hours = seconds_to_hours(duration)
            # wave_files_duration[filename] = duration_hours
            wave_files_info[filename] = [duration, frames]
    return wave_files_info

# Example usage
duration_sum = 0.
frames_sum = 0
count = 0
directory_path = '/home/pc/bc2023/FastSpeech2_bc/FastSpeech2/output/result/bc2023_testing'
info_dict = get_wave_files_info(directory_path)
print(len(info_dict))
for filename, info in info_dict.items():
    duration_sum += info[0]
    frames_sum += info[1]


print(duration_sum, frames_sum)