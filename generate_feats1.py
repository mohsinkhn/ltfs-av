import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import QuantileTransformer, StandardScaler, LabelEncoder

from utils import read_data 
from config import UTILITY


def preprocess_data(train , test):
    lbl = LabelEncoder()
    train["cns_desc"] = lbl.fit_transform(train["PERFORM_CNS.SCORE.DESCRIPTION"])
    test["cns_desc"] = lbl.transform(test["PERFORM_CNS.SCORE.DESCRIPTION"])
    
    emp_map = {"Self employed":0, "Salaried":1}
    train["etype"] = train["Employment.Type"].map(emp_map)
    test["etype"] = test["Employment.Type"].map(emp_map)
    return train, test


def acct_age_float(x):
    x=x.split(" ")
    x[0] = int(x[0].replace("yrs", ""))
    x[1] = int(x[1].replace("mon", ""))
    return x[0] + x[1]/12 


def get_extra_feats(df):
    df["loan_ratio"] = df["disbursed_amount"] / df["asset_cost"]
    df["avg_acc_age"] = df["AVERAGE.ACCT.AGE"].apply(acct_age_float)
    df["cr_hist_len"] = df["CREDIT.HISTORY.LENGTH"].apply(acct_age_float)
    df["age"] = (pd.to_datetime("2019-01-12") - df["Date.of.Birth"]).dt.days // 365
    df["disbursed_pri_amt"] = np.log1p(df["PRI.DISBURSED.AMOUNT"].clip(0, 1e9))
    df["sanctioned_pri_amt"] = np.log1p(df["PRI.SANCTIONED.AMOUNT"].clip(0, 1e9))
    df["disbur_to_sanction"] = np.log1p(df["PRI.DISBURSED.AMOUNT"]/(1+df["PRI.SANCTIONED.AMOUNT"].clip(0, 1e9)))
    df["disbur_to_sanction2"] = np.log1p(df["SEC.DISBURSED.AMOUNT"]/(1+df["SEC.SANCTIONED.AMOUNT"].clip(0, 1e9)))
    df["total_disbursed"] = (df["PRI.DISBURSED.AMOUNT"] + df["SEC.DISBURSED.AMOUNT"]).clip(0, 1e9)
    df["total_sanctioned"] = (df["SEC.SANCTIONED.AMOUNT"] + df["SEC.SANCTIONED.AMOUNT"]).clip(0, 1e9)
    df["total_disbur_to_sanction_ratio"] = df["total_disbursed"]/(1+df["total_sanctioned"])
    return df


def scale_data(train, test, feats):
    scaler = QuantileTransformer(output_distribution="normal", n_quantiles=2000, subsample=5e5, random_state=12345786)
    df_all = pd.concat([train[feats], test[feats]], axis=0)
    scaler.fit(df_all)
    qnt_feats = [f+"_qnt" for f in feats]
    train_qnt = pd.DataFrame(scaler.transform(train[feats]), columns=qnt_feats)
    test_qnt = pd.DataFrame(scaler.transform(test[feats]), columns=qnt_feats)
    return train_qnt, test_qnt
    

if __name__=="__main__":
    print("Reading data")
    train = read_data("train")
    test = read_data("test")

    print("preprocess data")
    train, test = preprocess_data(train, test)
    
    print("Generating extra features")
    train = get_extra_feats(train)
    test = get_extra_feats(test)
    
    feats = ["etype", "cns_desc", "loan_ratio", "avg_acc_age", "cr_hist_len", "age", "disbursed_pri_amt",
             "sanctioned_pri_amt", "disbur_to_sanction", "disbur_to_sanction2",
             "total_disbursed", "total_sanctioned",  "total_disbur_to_sanction_ratio"]
    
    print("Quantile transformer")
    train_qnt, test_qnt = scale_data(train, test, feats)
    
    print("Saving stuff")
    train[feats].to_csv(str(Path(UTILITY) / "train_feats1.csv"), index=False)
    test[feats].to_csv(str(Path(UTILITY) / "test_feats1.csv"), index=False)

    train_qnt.to_csv(str(Path(UTILITY) / "train_feats1_qnt.csv"), index=False)
    test_qnt.to_csv(str(Path(UTILITY) / "test_feats1_qnt.csv"), index=False)

