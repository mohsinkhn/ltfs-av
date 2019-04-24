import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, QuantileTransformer, MinMaxScaler, OneHotEncoder
from keras.layers import Input, Dense
from keras.models import Model, load_model, save_model
from keras.optimizers import Adam, SGD

from config import UTILITY, LOGS
from utils import read_data, read_base_feats

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


BUREAU_CONFIG = {
             "qnt_cols": ['PERFORM_CNS.SCORE',
                  'PRI.CURRENT.BALANCE',
                  'PRI.SANCTIONED.AMOUNT', 'PRI.DISBURSED.AMOUNT',
                  'SEC.CURRENT.BALANCE',
                  'SEC.SANCTIONED.AMOUNT', 'SEC.DISBURSED.AMOUNT',
                  'PRIMARY.INSTAL.AMT', 'SEC.INSTAL.AMT'],
             "cat_cols": ['cns_desc'],
             "minmax_cols": ['PRI.NO.OF.ACCTS', 'PRI.ACTIVE.ACCTS',
                  'PRI.OVERDUE.ACCTS', 'SEC.NO.OF.ACCTS', 'SEC.ACTIVE.ACCTS',
                  'SEC.OVERDUE.ACCTS', 'avg_acc_age', 'cr_hist_len',
                  'NEW.ACCTS.IN.LAST.SIX.MONTHS', 'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS',
                  'NO.OF_INQUIRIES'],
             "latent_dim": 256,
             "epochs": 500,
             }


def qnt_transform(train, test):
    cont_feats = BUREAU_CONFIG["qnt_cols"] 
    data = pd.concat([train[cont_feats],
                      test[cont_feats] ])
    scaler = QuantileTransformer(output_distribution="normal", n_quantiles=2000)
    scaler.fit(data)
    train_qnt = scaler.transform(train[cont_feats])
    test_qnt = scaler.transform(test[cont_feats])
    return train_qnt, test_qnt


def minmax_transform(train, test):
    feats = BUREAU_CONFIG["minmax_cols"] 
    data = pd.concat([train[feats],
                      test[feats] ])
    scaler = MinMaxScaler()
    scaler.fit(data)
    train_minmax = scaler.transform(train[feats])
    test_minmax = scaler.transform(test[feats])
    return train_minmax, test_minmax


def cat_transform(train, test):
    feats = BUREAU_CONFIG["cat_cols"] 
    data = pd.concat([train[feats],
                      test[feats] ])
    scaler = OneHotEncoder(sparse=False)
    scaler.fit(data)
    train_onehot = scaler.transform(train[feats])
    test_onehot = scaler.transform(test[feats])
    return train_onehot, test_onehot


class AENN():
    def __init__(self, hidden_dim=100, num_feats=None):
        self.hidden_dim = hidden_dim
        self.num_feats = num_feats
        self.model = self.build_model()
    
    def build_model(self):
        self.inp = Input(shape=(self.num_feats, ))
        self.h1 = Dense(int(self.hidden_dim * 3), activation='relu')(self.inp)
        self.d1 = Dense(self.hidden_dim,)(self.h1)
        self.h2 = Dense(int(self.hidden_dim * 3), activation='relu')(self.d1)
        self.out = Dense(self.num_feats)(self.h2)
        model = Model(inputs=[self.inp], outputs=[self.out])
        opt = Adam(decay=1e-3) #SGD(lr=0.01, decay=1e-3, momentum=0.9, nesterov=True)
        model.compile(loss='mean_squared_error', optimizer=opt)
        return model

    def encoder(self):
        model = Model(inputs=[self.inp], outputs=self.d1)
        return model


if __name__=="__main__":
    train = read_base_feats("train")
    test = read_base_feats("test")

    train_qnt, test_qnt = qnt_transform(train, test)
    train_minmax, test_minmax = minmax_transform(train, test)
    train_oh, test_oh = cat_transform(train, test)

    train_all = np.hstack((train_qnt, train_minmax, train_oh))
    test_all = np.hstack((test_qnt, test_minmax, test_oh))
    data_all = np.vstack((train_all, test_all))
    print(data_all.shape)

    ae = AENN(hidden_dim = BUREAU_CONFIG["latent_dim"],
              num_feats = data_all.shape[1])
    ae.model.fit(data_all, data_all, epochs=BUREAU_CONFIG["epochs"], batch_size=3000)
    enc = ae.encoder()
    data_ae = np.hstack((enc.predict(data_all), data_all))
    print(data_ae.shape, data_ae[:2])
    train_ae = data_ae[: len(train)]
    test_ae = data_ae[len(train):]
    np.save(str(Path(UTILITY) / "train_bureau_ae.npy"), train_ae)
    np.save(str(Path(UTILITY) / "test_bureau_ae.npy"), test_ae)
