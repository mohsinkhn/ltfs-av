import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GroupKFold
from sklearn.metrics import roc_auc_score

from config import UTILITY, SUBMISSIONS, LOGS
from utils import read_data, write_sub, get_cvlist, get_expanding_count
from TargetEncoder import TargetEncoderWithThresh

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

LOGGER_FILE=str(Path(LOGS) / "generate_base_feathers.log")
handler = logging.FileHandler(LOGGER_FILE)
handler.setLevel(logging.INFO)
# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def read_feats1(flag="train"):
    return pd.read_csv(str(Path(UTILITY) / "{}_feats1.csv".format(flag)))


if __name__=="__main__":
    logger.info("")
    logger.info("<===========STARTING SCRIPT==================>")
    logger.info("Reading data")
    train = read_data("train")
    test = read_data("test")
    logger.info(train.columns)
    logger.info("Reading feats1")
    train_feats1 = read_feats1("train")    
    test_feats1 = read_feats1("test")
    
    train = pd.concat([train, train_feats1], axis=1)
    test = pd.concat([test, test_feats1], axis=1)
    
    train = train.sort_values(by=["DisbursalDate"]).reset_index(drop=True)
    test = test.sort_values(by=["DisbursalDate"]).reset_index(drop=True)
    
    logger.info("Saving base features to feather file")
    train.to_feather(str(Path(UTILITY) / "train_basefeats.ftr"))
    test.to_feather(str(Path(UTILITY) / "test_basefeats.ftr"))

