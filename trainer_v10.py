import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_predict, GroupKFold
from sklearn.metrics import roc_auc_score
from collections import Counter
from config import UTILITY, SUBMISSIONS, LOGS
from utils import read_data, write_sub, get_cvlist, get_expanding_count

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

LOGGER_FILE=str(Path(LOGS) / "trainer_v10.log")
handler = logging.FileHandler(LOGGER_FILE)
handler.setLevel(logging.INFO)
# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


def run_lightgbm(X_tr, y_tr, X_val, y_val, lgb_params, feat_name=None, cat_feats=None):
    model = lgb.LGBMClassifier(**lgb_params)
    if cat_feats is None:
        cat_feats = []
    if feat_name is None:
        feat_name = []
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric=["auc"], verbose=50, early_stopping_rounds=500,
              categorical_feature=cat_feats, feature_name=feat_name)
    val_preds = model.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val, val_preds)
    logger.info("ROC AUC for this cvfold is {}".format(score))
    return score, model, val_preds


def get_test_preds(X, y, X_test, lgb_params, feat_name=None, cat_feats=None):
    if feat_name is None:
        feat_name = []
    if cat_feats is None:
        cat_feats = []
    model = lgb.LGBMClassifier(**lgb_params)
    y_test_preds = model.fit(X, y, feature_name=feat_name, categorical_feature=cat_feats).predict_proba(X_test)[:, 1]
    return y_test_preds


def read_base_feats(flag="train"):
    return pd.read_feather(str(Path(UTILITY) / "{}_basefeats.ftr".format(flag)))


def exp_cnt_feats(train, test, idcols):
    for idcol in idcols:
        fname1 = idcol + "_expcnt"
        train.loc[:, fname1] = get_expanding_count(train["DisbursalDate"].astype(int).values, train[idcol].values)
        #train[fname1] = train[idcol].map(train.groupby(idcol)["loan_default"].count()).fillna(0)
        test.loc[:, fname1] = test[idcol].map(train.groupby(idcol)["loan_default"].count()).fillna(0).astype(float)

        fname2 = idcol + "_expsum"
        train.loc[:, fname2] = get_expanding_count(train["DisbursalDate"].astype(int).values, train[idcol].values, train["loan_default"].values)
        #train[fname2] = train[idcol].map(train.loc[train["loan_default"] == 1].groupby(idcol)["loan_default"].count()).fillna(0)
        test.loc[:, fname2] = test[idcol].map(train.groupby([idcol])["loan_default"].sum()).fillna(0).astype(float)

        fname3 = idcol + "_expmean"
        train.loc[:, fname3] = train[fname2]/(1+train[fname1])
        test.loc[:, fname3] = test[fname2]/(1+test[fname1])
    return train, test


def get_aggregate_feats(train, test, combs):
    for comb in combs:
        fname = "_".join(comb)
        grp, col, agg = comb
        df_all = pd.concat([train[[grp, col]], test[[grp, col]]])
        fmap = getattr(df_all.groupby(grp)[col], agg)()
        train[fname] = train[grp].map(fmap)
        test[fname] = test[grp].map(fmap)
    return train, test


if __name__=="__main__":
    logger.info("")
    logger.info("<===========STARTING SCRIPT==================>")
    logger.info("Reading data")
    train = read_base_feats("train")
    test = read_base_feats("test")
    logger.info(train.columns)
    train["DisbursalDay"] = train["DisbursalDate"].dt.day
    test["DisbursalDay"] = test["DisbursalDate"].dt.day
    train["Disbursalweek"] = train["DisbursalDate"].dt.day // 7
    test["Disbursalweek"] = test["DisbursalDate"].dt.day // 7
    train["fake_dob"] = train["Date.of.Birth"].astype(str).str.contains("01-01")
    test["fake_dob"] = test["Date.of.Birth"].astype(str).str.contains("01-01")

    train["no_code"] = (train["Employee_code_ID"] == 3235).astype(int)+ (train["supplier_id"] == 2689).astype(int) + (train["Current_pincode_ID"] == 5047).astype(int)
    test["no_code"] = (test["Employee_code_ID"] == 3235).astype(int) + (test["supplier_id"] == 2689).astype(int) + (test["Current_pincode_ID"] == 5047).astype(int)

    most_common_map = train.groupby("branch_id")["State_ID"].apply(lambda x: Counter(x).most_common(1)[0][0])
    train["state_err"] = train.State_ID != train.branch_id.map(most_common_map)
    test["state_err"] = test.State_ID != test.branch_id.map(most_common_map)
    
    most_common_map = train.groupby("Employee_code_ID")["branch_id"].apply(lambda x: Counter(x).most_common(1)[0][0])
    train["emp_err"] = train.branch_id != train.Employee_code_ID.map(most_common_map)
    test["emp_err"] = test.branch_id != test.Employee_code_ID.map(most_common_map)

    #most_common_map = train.groupby("Current_pincode_ID")["branch_id"].apply(lambda x: Counter(x).most_common(1)[0][0])
    #train["pin_branch"] = train.Current_pincode_ID.map(most_common_map)
    #test["pin_branch"] = test.Current_pincode_ID.map(most_common_map)
    
    tr = train.loc[(train.DisbursalDate < pd.to_datetime("2018-10-01")) &
                   (train.DisbursalDate >= pd.to_datetime("2018-08-01"))].reset_index(drop=True)    
    val = train.loc[(train.DisbursalDate >= pd.to_datetime("2018-10-01"))].reset_index(drop=True)
    
    feats_for_cnt = ["branch_id", "supplier_id", "Current_pincode_ID", "cns_desc", "Employee_code_ID", "State_ID", "supplier_id"]
    tr, val = exp_cnt_feats(tr, val, feats_for_cnt)
    train, test = exp_cnt_feats(train, test, feats_for_cnt)

    combs = [("branch_id", "PERFORM_CNS.SCORE", "std"),
             #("branch_id", "PERFORM_CNS.SCORE", "mean"),
             #("supplier_id", "disbursed_amount", "count"),
             #("supplier_id", "disbursed_amount", "mean"),
             #("supplier_id", "ltv", "std"),
             #("supplier_id", "asset_cost", "count"),
             ("supplier_id", "PERFORM_CNS.SCORE", "mean"),
             ("supplier_id", "PERFORM_CNS.SCORE", "sum"),
             ("supplier_id", "PERFORM_CNS.SCORE", "std"),
             #("supplier_id", "age", "count"),
             #("supplier_id", "disbursed_pri_amt", "mean"),
             #("supplier_id", "disbursed_pri_amt", "sum"),
             #("Employee_code_ID", "PERFORM_CNS.SCORE", "mean"),
             #("Employee_code_ID", "PERFORM_CNS.SCORE", "sum"),
             #("Employee_code_ID", "PERFORM_CNS.SCORE", "std"),
             #("Employee_code_ID", "age", "count"),
             #("Employee_code_ID", "disbursed_pri_amt", "mean"),
             #("Employee_code_ID", "disbursed_amount", "skew"),
             #("Employee_code_ID", "age", "skew"),
             #("Employee_code_ID", "DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS", "skew"),
             #("Current_pincode_ID", "disbursed_amount", "count"),
             #("Current_pincode_ID", "ltv", "mean"),
             #("Current_pincode_ID", "ltv", "sum"),
             ("Current_pincode_ID", "PERFORM_CNS.SCORE", "mean"),
             ("Current_pincode_ID", "PERFORM_CNS.SCORE", "sum"),           
             ("DisbursalDate", "disbursed_amount", "count"),            
             ("DisbursalDate", "disbursed_amount", "mean"),            
             #("DisbursalDate", "PERFORM_CNS.SCORE", "std"),
             ("disbursed_amount", "PERFORM_CNS.SCORE", "count")            
            ]

    tr, val = get_aggregate_feats(tr, val, combs)   
    train, test = get_aggregate_feats(train, test, combs)   
    base_feats = ['disbursed_amount', 'ltv', 'branch_id',
                  'manufacturer_id', 
                  'State_ID', 'Employee_code_ID', 'PAN_flag',
                  'Aadhar_flag', 'VoterID_flag',
                  'Driving_flag', 'Passport_flag', 'PERFORM_CNS.SCORE',
                  'PRI.NO.OF.ACCTS', 'PRI.ACTIVE.ACCTS',
                  'PRI.OVERDUE.ACCTS', 'PRI.CURRENT.BALANCE', 'SEC.NO.OF.ACCTS',
                  'SEC.OVERDUE.ACCTS', 'SEC.CURRENT.BALANCE', 'SEC.SANCTIONED.AMOUNT',
                  'SEC.DISBURSED.AMOUNT', 'PRIMARY.INSTAL.AMT', 'SEC.INSTAL.AMT',
                  'NEW.ACCTS.IN.LAST.SIX.MONTHS', 'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS',
                  'NO.OF_INQUIRIES', "etype", "cns_desc", "loan_ratio", "avg_acc_age", "cr_hist_len", "age", "disbursed_pri_amt",
                  "sanctioned_pri_amt", "disbur_to_sanction", "disbur_to_sanction2",
                  "total_disbursed", "total_sanctioned",  "total_disbur_to_sanction_ratio"]

    cat_feats = ["branch_id", "cns_desc", "manufacturer_id", "Employee_code_ID"]
    expmean_feats = [f+"_expmean" for f in feats_for_cnt]
    expcnt_feats = [f+"_expcnt" for f in feats_for_cnt]
    feats_combs = ["_".join(c) for c in combs]

    all_feats = base_feats + feats_combs + expmean_feats + ["DisbursalDay", "Disbursalweek", "fake_dob", "no_code", "state_err", "emp_err"]
    #X = train[all_feats].values
    #y = train["loan_default"].values
   
    X_tr = tr[all_feats].values
    X_val = val[all_feats].values

    y_tr = tr["loan_default"].values
    y_val = val["loan_default"].values
    
    lgb_params = {
                 "n_estimators": 20000,
                 "learning_rate": 0.05,
                 "subsample": 0.7,
                 "colsample_bytree": 0.2,
                 "num_leaves": 15,
                 "reg_lambda": 1000,
                 "reg_alpha": 10,
                 "max_depth": 5,
                 "min_child_samples": 100,
                 "min_child_weight": 1,
                 "metrics": None,
                 "seed": 1,
                 "boosting_type": 'gbdt',
                 "min_data_per_group": 100
               }

    logger.info("Training model with {}".format(lgb_params))
    base_score, model, _ = run_lightgbm(X_tr, y_tr, X_val, y_val, lgb_params, feat_name=all_feats, cat_feats=cat_feats)
    logger.info("Base score with {} is {}".format(all_feats, base_score))
    
    lgb_params["n_estimators"] = 1250
    logger.info("Predicting for test set with params {}".format(lgb_params))
    X_train = train[all_feats].values
    y = train["loan_default"].values
    X_test = test[all_feats].values
    y_test_preds = get_test_preds(X_train, y, X_test, lgb_params, feat_name=all_feats, cat_feats=cat_feats)
    
    sub_file = "sub_v10_6647.csv"
    sub = test[["UniqueID"]]
    sub["loan_default"] = y_test_preds
    logger.info("Writing out submission to {}".format(sub_file))
    sub.to_csv(str(Path(SUBMISSIONS) / sub_file), index=False)
    
