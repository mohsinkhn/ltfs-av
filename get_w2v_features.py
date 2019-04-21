from gensim.models import Word2Vec
from itertools import chain
import pandas as pd
import numpy as np
from pathlib import Path

from config import UTILITY, LOGS
from utils import read_data

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

LOGGER_FILE=str(Path(LOGS) / "get_w2v_feats.log")
handler = logging.FileHandler(LOGGER_FILE)
handler.setLevel(logging.INFO)
# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


W2V_CONFIG = {
             "cols": ["branch_id", "manufacturer_id",
                      "supplier_id", "Employee_code_ID",
                      "Current_pincode_ID", "State_ID"],
             "vector_size": 150,
             "window_size":6,
             "epochs": 30,
             "min_count": 1,
             "sample": 1e-1
             }


def make_sentences(df):
    df["sentence"] = ""
    for col in W2V_CONFIG["cols"]:
        df["sentence"] += col[:3] + "_" + df[col].astype(str) + " "
    return df


if __name__=="__main__":
    train = read_data("train")
    test = read_data("test")

    train = make_sentences(train)
    test = make_sentences(test)

    all_sentences = list(np.hstack((train["sentence"].str.split(" ").values, test["sentence"].str.split(" ").values)))
    w2v_model = Word2Vec(min_count=W2V_CONFIG["min_count"],
                     window=W2V_CONFIG["window_size"],
                     size=W2V_CONFIG["vector_size"],
                     sample=W2V_CONFIG["sample"],
                     workers=15)
    w2v_model.build_vocab(all_sentences, progress_per=10000)
    w2v_model.train(all_sentences, total_examples=w2v_model.corpus_count, epochs=100, report_delay=1)
    w2v_model.wv.save(str(Path(UTILITY)/ "w2v_model.vectors"))
