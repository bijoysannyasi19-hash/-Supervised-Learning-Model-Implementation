import numpy as np
from sklearn.model_selection import KFold, cross_validate, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def cv_eval(models, X, y):
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    res = {}
    for n, m in models.items():
        sc = cross_validate(m, X, y, cv=cv, scoring=("neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"), return_train_score=True, n_jobs=-1)
        res[n] = {
            "test_rmse": float(-np.mean(sc["test_neg_root_mean_squared_error"])),
            "test_rmse_std": float(np.std(sc["test_neg_root_mean_squared_error"])),
            "train_rmse": float(-np.mean(sc["train_neg_root_mean_squared_error"])),
            "test_mae": float(-np.mean(sc["test_neg_mean_absolute_error"])),
            "test_r2": float(np.mean(sc["test_r2"]))
        }
    return res

def tune_model(pl, bm_name, X, y):
    if bm_name == "HGB":
        pg = {"m__learning_rate": [0.05, 0.1], "m__max_iter": [100, 200], "m__max_depth": [3, 5]}
    elif bm_name == "RF":
        pg = {"m__n_estimators": [100, 200], "m__max_depth": [None, 10, 20]}
    elif bm_name == "Ridge":
        pg = {"m__alpha": [0.1, 1.0, 10.0]}
    else:
        pg = {}
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    if pg:
        gs = GridSearchCV(pl, pg, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
        gs.fit(X, y)
        return gs.best_estimator_, gs.best_params_, float(-gs.best_score_)
    else:
        pl.fit(X, y)
        return pl, {}, -1.0

def eval_test(m, X, y):
    yp = m.predict(X)
    rmse = np.sqrt(mean_squared_error(y, yp))
    mae = mean_absolute_error(y, yp)
    r2 = r2_score(y, yp)
    return {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}, yp
