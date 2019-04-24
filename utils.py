import pandas as pd
import numpy as np
import numba
from pathlib import Path
from collections import Counter
import itertools

from config import ROOT, UTILITY, SUBMISSIONS


def read_base_feats(flag="train"):
    return pd.read_feather(str(Path(UTILITY) / "{}_basefeats.ftr".format(flag)))


def correct_dob(x):
    x = x.split("-")
    d, m, y = x[0], x[1], x[2]
    if int(y) > 15:
        y = "19" + y
    else:
        y = "20" + y
    return "-".join([d, m, y])


def read_data(flag="train"):
    filename = "{}.csv".format(flag)
    df = pd.read_csv(str(Path(ROOT) / filename), parse_dates=["DisbursalDate"], dayfirst=True)
    df["Date.of.Birth"] = df["Date.of.Birth"].apply(correct_dob)
    df["Date.of.Birth"] = pd.to_datetime(df["Date.of.Birth"])
    return df


def get_cvlist(train):
    tr_idx1 = train.loc[train.DisbursalDate < pd.to_datetime("2018-10-01")].index
    val_idx1 = train.loc[train.DisbursalDate >= pd.to_datetime("2018-10-01")].index
    
    tr_idx2 = train.loc[train.DisbursalDate < pd.to_datetime("2018-10-15")].index
    val_idx2 = train.loc[train.DisbursalDate >= pd.to_datetime("2018-10-15")].index

    tr_idx3 = train.loc[train.DisbursalDate < pd.to_datetime("2018-10-21")].index
    val_idx3 = train.loc[train.DisbursalDate >= pd.to_datetime("2018-10-21")].index

    cvlist = [(tr_idx3, val_idx3), (tr_idx2, val_idx2), (tr_idx1, val_idx1)]
    return cvlist


def write_sub(y_test_preds, filename="sub.csv"):
    sub = pd.read_csv(str(Path(ROOT) / "test.csv"), usecols=["UniqueID"])
    sub["loan_default"] = y_test_preds
    sub.to_csv(str(Path(SUBMISSIONS) / filename), index=False)


@numba.jit
def get_splits(a):
    m = np.concatenate([[True], a[1:] != a[:-1], [True]])
    m = np.flatnonzero(m)
    return m


@numba.jit
def get_unq_reverse(arr):
    unq = np.unique(arr)
    #sort_idx = np.argsort(arr)
    #unsort_idx = np.argsort(sort_idx)
    #sorted_arr = arr[sort_idx]
    out_rev = np.zeros((len(arr), ))
    n = len(arr)
    for i in range(len(arr)):
        for j in range(len(unq)):
            if arr[i] == unq[j]:
                out_rev[i] = j
   
    return unq, out_rev


@numba.jit
def get_expanding_count(datecol, idcol, targetcol=None):
    """
    expanding counts
    """
    unq_ids, idcol_inv = np.unique(idcol, return_inverse=True)
    unq_dates, date_cnts = np.unique(datecol, return_counts=True)

    n = len(idcol)
    nunq = len(unq_ids)
    learned_dict = np.zeros((nunq, ))
    tmp_cnts = np.zeros((nunq, ))
    out = np.zeros((n, ))
    if targetcol is not None:
        m = get_splits(datecol)
        n = len(m) - 1
        for i in range(n):
            j = m[i]
            k = m[i+1]
            sub_data = idcol_inv[j:k][targetcol[j:k] == 1]
            vals, cnts = np.unique(sub_data, return_counts=True)
            for p in range(j, k):
                out[p] = learned_dict[idcol_inv[p]]
            for (val, cnt) in zip(vals, cnts):
                learned_dict[val] += cnt
                
    else:
        m = get_splits(datecol)
        n = len(m) - 1
        for i in range(n):
            j = m[i]
            k = m[i+1]
            sub_data = idcol_inv[j:k]
            vals, cnts = np.unique(sub_data, return_counts=True)
            for p in range(j, k):
                out[p] = learned_dict[idcol_inv[p]]
            
            for (val, cnt) in zip(vals, cnts):
                learned_dict[val] += cnt
                
                
    return out


class Tokenizer:
    def __init__(self, max_features=20000, max_len=6, tokenizer=str.split):
        self.max_features = max_features
        self.tokenizer = tokenizer
        self.doc_freq = None
        self.vocab = None
        self.vocab_idx = None
        self.max_len = max_len

    def fit_transform(self, texts):
        tokenized = []
        n = len(texts)

        tokenized = [self.tokenizer(text) for text in texts]
        self.doc_freq = Counter(itertools.chain.from_iterable(tokenized))

        vocab = [t[0] for t in self.doc_freq.most_common(self.max_features)]
        vocab_idx = {w: (i + 1) for (i, w) in enumerate(vocab)}
        # doc_freq = [doc_freq[t] for t in vocab]

        # self.doc_freq = doc_freq
        self.vocab = vocab
        self.vocab_idx = vocab_idx

        result_list = []
        #tokenized = [self.tokenizer(text) for text in texts]
        for text in tokenized:
            text = self.text_to_idx(text, self.max_len)
            result_list.append(text)

        result = np.zeros(shape=(n, self.max_len), dtype=np.int32)
        for i in range(n):
            text = result_list[i]
            result[i, :len(text)] = text

        return result

    def text_to_idx(self, tokenized, max_len):
        return [self.vocab_idx[t] for i, t in enumerate(tokenized) if (t in self.vocab_idx) and (i < max_len)]

    def transform(self, texts):
        n = len(texts)
        result = np.zeros(shape=(n, self.max_len), dtype=np.int32)

        for i in range(n):
            text = self.tokenizer(texts[i])
            text = self.text_to_idx(text, self.max_len)
            result[i, :len(text)] = text

        return result

    def vocabulary_size(self):
        return len(self.vocab) + 1


def normalize_disbursal(train, test):
    train["DWeek"] = train["DisbursalDate"].dt.week
    test["DWeek"] = test["DisbursalDate"].dt.week

    data = pd.concat([train[["DWeek", "disbursed_amount"]], test[["DWeek", "disbursed_amount"]]])
    mean_map = data.groupby("DWeek")["disbursed_amount"].mean()
    train["mean_disbursed_amount"] = train["DWeek"].map(mean_map)
    test["mean_disbursed_amount"] = test["DWeek"].map(mean_map)

    train["disbursed_amount"] = train["disbursed_amount"] - train["mean_disbursed_amount"]
    test["disbursed_amount"] = test["disbursed_amount"] - test["mean_disbursed_amount"]
    return train, test
