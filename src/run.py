import os
import json
import pandas as pd
import joblib
from sklearn.inspection import permutation_importance
from data import load_data, save_raw, split_data
from models import get_models
from evaluate import cv_eval, tune_model, eval_test
import plots

def run():
    d = os.path.join(os.path.dirname(__file__), "..", "data")
    r = os.path.join(os.path.dirname(__file__), "..", "results")
    f = os.path.join(os.path.dirname(__file__), "..", "figures")
    m = os.path.join(os.path.dirname(__file__), "..", "models")
    doc = os.path.join(os.path.dirname(__file__), "..", "docs")
    
    os.makedirs(d, exist_ok=True)
    os.makedirs(r, exist_ok=True)
    os.makedirs(f, exist_ok=True)
    os.makedirs(m, exist_ok=True)
    os.makedirs(doc, exist_ok=True)
    
    df = load_data()
    save_raw(df, os.path.join(d, "raw.csv"))
    
    plots.p_dist(df["SalePrice"], f)
    plots.p_miss(df, f)
    plots.p_feat(df, f)
    plots.p_feat2(df, f)
    plots.p_corr(df, f)
    plots.p_arch(doc)
    
    Xtr, Xte, ytr, yte = split_data(df)
    
    ms = get_models()
    res = cv_eval(ms, Xtr, ytr)
    pd.DataFrame(res).T.to_csv(os.path.join(r, "cv_results.csv"))
    
    plots.p_comp(res, f)
    
    bm_name = min(res, key=lambda k: res[k]["test_rmse"])
    pl = ms[bm_name]
    
    bm, bp, bsc = tune_model(pl, bm_name, Xtr, ytr)
    
    with open(os.path.join(r, "metrics.json"), "w") as jf:
        json.dump({"best_model": bm_name, "best_params": bp, "best_cv_rmse": bsc}, jf)
    
    plots.p_lc(bm, Xtr, ytr, f)
    
    tres, ypr = eval_test(bm, Xte, yte)
    with open(os.path.join(r, "test_results.json"), "w") as jf:
        json.dump(tres, jf)
    
    plots.p_eval(yte, ypr, f)
    plots.p_res(yte, ypr, f)
    plots.p_resh(yte, ypr, f)
    plots.p_err(yte, ypr, Xte, f)
    
    joblib.dump(bm, os.path.join(m, "final_model.joblib"))
    
    pi = permutation_importance(bm, Xte, yte, n_repeats=10, random_state=42, scoring="neg_root_mean_squared_error")
    fi = pd.DataFrame({"Feature": Xte.columns, "Importance": pi.importances_mean})
    fi = fi.sort_values(by="Importance", ascending=False)
    fi.to_csv(os.path.join(r, "feature_importance.csv"), index=False)
    
    plots.p_imp(fi, f)

if __name__ == "__main__":
    run()
