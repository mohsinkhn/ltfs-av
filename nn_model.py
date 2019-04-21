import numpy as np
import pandas as pd
from pathlib import Path
from keras.layers import Input, Embedding, Flatten, concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D, Dense, Dropout, BatchNormalization
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
from keras.models import Model, load_model, save_model
from keras.utils import Sequence
from keras.callbacks import Callback, ModelCheckpoint, LearningRateScheduler
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import roc_auc_score
from scipy.stats import gmean
from gensim.models import KeyedVectors
np.random.seed(10001)
import tensorflow as tf
tf.random.set_random_seed(10001)

from config import UTILITY, ROOT, SUBMISSIONS
from utils import read_data, read_base_feats, Tokenizer
from get_w2v_features import make_sentences, W2V_CONFIG


def tokenize_sentence(train, test, w2v_model, max_len=6):
    data = pd.concat([train["sentence"],  test["sentence"]]).values
    tok = Tokenizer(max_features=15000, max_len=max_len)
    tokens = tok.fit_transform(data)
    #n = len(train)
    #train_tokens = tokens[:n]
    #test_tokens = tokens[n:]
    vocab_len = tok.vocabulary_size()
    idx_to_word = {v:k for k, v in tok.vocab_idx.items()}
    embedding_matrix = np.zeros((vocab_len+1, W2V_CONFIG["vector_size"]))
    for i in range(vocab_len):
        if i == 0:
            continue
        embedding_matrix[i] = w2v_model[idx_to_word[i]]
    return tok, embedding_matrix


def load_bureau_feats():
    train = np.load(str(Path(UTILITY) / "train_bureau_ae.npy"))[:, :300] 
    test = np.load(str(Path(UTILITY) / "test_bureau_ae.npy"))[:, :300]
    bfeats = ["bf_"+str(i) for i in range(train.shape[1])]
    train = pd.DataFrame(train, columns=bfeats) 
    test = pd.DataFrame(test, columns=bfeats)
    return train, test

def prep_base_feats(train, test):
    cont_feats = ["disbursed_amount", "ltv", "age", "disbur_to_sanction", "disbur_to_sanction2"]
    data = pd.concat([train[cont_feats],
                      test[cont_feats] ])
    scaler = QuantileTransformer(output_distribution="normal", n_quantiles=2000)
    scaler.fit(data)
    train[cont_feats] = scaler.transform(train[cont_feats])
    test[cont_feats] = scaler.transform(test[cont_feats])
    
    bin_feats = ["Aadhar_flag", "PAN_flag", "VoterID_flag", "Driving_flag", "etype",
                "Disbursalweek", "Disbursaldayofweek", "fake_dob"]
    #data = pd.concat([train[bin_feats].fillna(0),
    #                  test[bin_feats].fillna(0) ])
    #scaler = QuantileTransformer(output_distribution="normal", n_quantiles=2000)
    #scaler.fit(data)
    train[bin_feats] = train[bin_feats].fillna(-1)
    test[bin_feats] = test[bin_feats].fillna(-1)
    return train, test


class ROC_AUC(Callback):
    def __init__(self, validation_data):
        self.X_val, self.y_val = validation_data
        #self.y_val = np.vstack([y for x, y in validation_data])
    
    def on_epoch_end(self, epoch, logs={}):
        y_preds = self.model.predict(self.X_val, batch_size=1000).flatten()
        val_rocauc = roc_auc_score(self.y_val, y_preds)
        logs.update({"val_rocauc": val_rocauc})
        print("ROC AUC for this fold, is ", val_rocauc)


class NNv1():
    def __init__(self, weights=None, w2v_feats=6, bureau_feats=32, cont_feats=7, trainable=False):
        self.weights = weights
        self.max_vocab = self.weights.shape[0]
        self.emb_dim = self.weights.shape[1]
        self.trainable = trainable
        self.w2v_feats = w2v_feats
        self.bureau_feats = bureau_feats
        self.cont_feats = cont_feats
        self.model = self.build_model()

    def build_model(self):
        inp1 = Input(shape=(self.w2v_feats, ))
        inp2 = Input(shape=(self.bureau_feats, ))
        inp3 = Input(shape=(self.cont_feats,))

        emb1 = Embedding(self.max_vocab, self.emb_dim, weights=[self.weights], input_length=self.w2v_feats, trainable=self.trainable)(inp1)
        x1 = GlobalAveragePooling1D()(emb1)
        #x1 = Flatten()(emb1)
        #x2 = Dense(256)(inp2)
        x3 = Dense(256, activation="relu")(inp3)
        x = concatenate([x1, inp2, x3])
        x = BatchNormalization()(x)
        x = Dense(1024, activation="relu")(x)
        #x = BatchNormalization()(x)
        x = Dropout(0.4)(x)
        x = Dense(1024, activation="relu")(x)
        #x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        #x = Dense(1024, activation="relu")(x)
        #x = Dropout(0.5)(x)
        x = Dense(1024, activation="relu")(x)
        out = Dense(1, activation="sigmoid")(x)
        
        model = Model(inputs=[inp1, inp2, inp3], outputs=out)
        opt = Adam(lr=0.001) #
        #opt = SGD(lr=0.01, momentum=0.9, nesterov=True)
        model.compile(loss="binary_crossentropy", optimizer=opt)
        model.summary()
         
        return model



def prep_nn_inputs(tr, val, tok,  br_feats, base_feats):
    tr_tokens = tok.transform(tr["sentence"])
    val_tokens = tok.transform(val["sentence"])
    tr_br, val_br = tr[br_feats].values, val[br_feats].values
    tr_base, val_base = tr[base_feats].values, val[base_feats].values
    return [tr_tokens, tr_br, tr_base], [val_tokens, val_br, val_base]


if __name__=="__main__":
    train = read_base_feats("train")
    test = read_base_feats("test")

    train = make_sentences(train)
    test = make_sentences(test)
    train_br, test_br = load_bureau_feats()
    train = pd.concat([train, train_br], axis=1)
    test = pd.concat([test, test_br], axis=1)
    train["Disbursalweek"] = (train["DisbursalDate"].dt.day // 7)/4
    test["Disbursalweek"] = (test["DisbursalDate"].dt.day // 7)/4
    
    train["Disbursalday"] = (train["DisbursalDate"].dt.day )/31
    test["Disbursalday"] = (test["DisbursalDate"].dt.day )/31
    
    train["Disbursaldayofweek"] = train["DisbursalDate"].dt.dayofweek / 7
    test["Disbursaldayofweek"] = test["DisbursalDate"].dt.dayofweek / 7
    
    train["fake_dob"] = train["Date.of.Birth"].astype(str).str.contains("01-01")
    test["fake_dob"] = test["Date.of.Birth"].astype(str).str.contains("01-01")

    train, test = prep_base_feats(train, test)

    tr = train.loc[(train.DisbursalDate < pd.to_datetime("2018-10-24")) &
                   (train.DisbursalDate >= pd.to_datetime("2018-08-01"))].reset_index(drop=True)    
    val = train.loc[(train.DisbursalDate >= pd.to_datetime("2018-10-24"))].reset_index(drop=True)
   
    w2v_model = KeyedVectors.load(str(Path(UTILITY) / "w2v_model.vectors"))
    tok, emb_matrix = tokenize_sentence(train, test, w2v_model)
    
    
    bfeats = ["bf_"+str(i) for i in range(train_br.shape[1])]
    y_tr, y_val = tr["loan_default"].values, val["loan_default"].values

    base_feats = ["disbursed_amount", "ltv", "age", "loan_ratio", "disbur_to_sanction",
                  "disbur_to_sanction2", "Aadhar_flag", "PAN_flag", "VoterID_flag", "Driving_flag", "Passport_flag", "etype",
                  "Disbursaldayofweek", "Disbursalday", "Disbursalweek", "fake_dob"]
    
    tr_inputs, val_inputs = prep_nn_inputs(tr, val, tok, bfeats, base_feats)
    train_inputs, test_inputs = prep_nn_inputs(train, test, tok, bfeats, base_feats)

    roc_auc = ROC_AUC((val_inputs, y_val))
    lr_schedule = LearningRateScheduler(lambda epoch: 0.001 if epoch <= 6 else 0.00001, verbose=True)
    val_preds = []
    test_preds = []
    for i in range(3):
        checkpoint = ModelCheckpoint("nn_iter_{}.hdf5".format(i), save_weights_only=True, save_best_only=True, monitor="val_rocauc", verbose=True, mode="max")
        nnv1 = NNv1(weights=emb_matrix, w2v_feats=6, bureau_feats=len(bfeats), cont_feats=len(base_feats),  trainable=True)
        nnv1.model.fit(tr_inputs, y_tr, epochs=12, batch_size=1024, callbacks=[roc_auc, lr_schedule, checkpoint])
        nnv1.model.load_weights("nn_iter_{}.hdf5".format(i))
        val_pred = nnv1.model.predict(val_inputs)
        test_pred = nnv1.model.predict(test_inputs)
        val_preds.append(val_pred)
        test_preds.append(test_pred)
    y_val_preds = gmean(val_preds, axis=0)
    y_test_preds = gmean(test_preds, axis=0)
    np.save(str(Path(UTILITY) / "y_val_preds_nnv1.npy"), y_val_preds)
    np.save(str(Path(UTILITY) / "y_test_preds_nnv1.npy"), y_test_preds)
    print("ROC-AUC Score is ", roc_auc_score(y_val, y_val_preds))
    
    sub_file = "sub_nn_v1.csv"
    sub = test[["UniqueID"]]
    sub["loan_default"] = y_test_preds
    #logger.info("Writing out submission to {}".format(sub_file))
    sub.to_csv(str(Path(SUBMISSIONS) / sub_file), index=False)

