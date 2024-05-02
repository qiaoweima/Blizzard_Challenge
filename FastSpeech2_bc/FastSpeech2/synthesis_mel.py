import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import to_device, log, synth_one_sample
from model import FastSpeech2Loss
from dataset import Dataset
import numpy as np

from evaluate import evaluate

# from sentence_transformers import SentenceTransformer

# import ptvsd

# ptvsd.enable_attach(("0.0.0.0",5698))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(args, configs):
    # print("Waiting for remote debuger to attach....")
    # ptvsd.wait_for_attach()
    # print("Prepare training ...")

    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = Dataset(
        "train.txt", preprocess_config, train_config, sort=True, drop_last=True
    )
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4  # Set this larger than 1 to enable sorting in Dataset
    assert batch_size * group_size < len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=0
    )

    # Prepare model
    model, optimizer = get_model(args, configs, device, train=True)
    model = nn.DataParallel(model)
    num_param = get_param_num(model)
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)
    print("Number of FastSpeech2 Parameters:", num_param)


    # Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    # Training
    step = args.restore_step + 1
    epoch = 1
    total_step = train_config["step"]["total_step"]

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()

    mel_path = train_config['path']['mel_path']
    
    with torch.no_grad():
        model.eval()
        while True:
            inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
            for batchs in loader:
                for batch in batchs:
                    batch = to_device(batch, device)
                    # embeddings = sentence_emb_model(batch[1])

                    # Forward
                    # output = model(*([embeddings, batch[2:]]))
                    output = model(*(batch[2:]))
                    mel_postnet = output[1]
                    mel_lens = output[9]

                    basenames = batch[0]
                    
                    # print(mel_postnet)
                    for i , base in enumerate(basenames):
                        mel_file = os.path.join(mel_path,base + '.npy')
                        if os.path.exists(mel_file):
                            continue
                        # print(mel_lens[i])
                        np.save(mel_file, mel_postnet[i,:mel_lens[i],:].detach().cpu().numpy())
                    

                    if step == total_step:
                        quit()
                    step += 1
                    outer_bar.update(1)

                inner_bar.update(1)
            break
            # epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    main(args, configs)
