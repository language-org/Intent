# Intent inference
# author: Steeve Laquitaine

import os
from time import time
import sys

# set environment variables
proj_path = os.getcwd()
os.environ["PROJ_PATH"] = proj_path
os.environ["NLTK_DATA"] = os.path.join(
    proj_path, "data/06_models/nltk_data"
)

from src.intent.pipelines import train_predict

if __name__ == "__main__":
    """Entry point
    usage:
        python main.py train_predict
    """
    if sys.argv[1] == "train_predict":
        train_predict.run()
    else:
        raise NotImplementedError(
            """This is not implented: Use $python main.py train_predict"""
        )
    # clean up caches
    # os.system("rm -f ~/.allenlp")
