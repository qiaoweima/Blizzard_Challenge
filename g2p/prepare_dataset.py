import os
import numpy as np
import random
dict_path = "fr_lexicon_cleaned.dict"


train_ratio, dev_ratio = 0.999, 0.001


with open(dict_path) as f:
    lines = f.readlines()
    random.shuffle(lines)    
    train_lines = lines[:int(train_ratio * len(lines))]
    dev_lines = lines[int(train_ratio * len(lines)):]
    with open("train_dict.txt",'w') as tf:
        tf.writelines(train_lines)
    with open("dev_dict.txt",'w') as df:
        df.writelines(dev_lines)


