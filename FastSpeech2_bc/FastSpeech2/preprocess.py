import argparse

import yaml
import os
from preprocessor.preprocessor import Preprocessor

# import ptvsd
# ptvsd.enable_attach(("0.0.0.0","5601"))

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

if __name__ == "__main__":
    # print("Waiting for debuger to Attach....")
    # ptvsd.wait_for_attach()
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    preprocessor = Preprocessor(config)
    preprocessor.build_from_path()
