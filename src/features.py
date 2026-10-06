import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer

def fe_func(X):
    d = X.copy()
    if "YearBuilt" in d.columns and "YrSold" in d.columns:
        d["HouseAge"] = d["YrSold"] - d["YearBuilt"]
        d.loc[d["HouseAge"] < 0, "HouseAge"] = 0
    if "YearRemodAdd" in d.columns and "YrSold" in d.columns:
        d["RemodAge"] = d["YrSold"] - d["YearRemodAdd"]
        d.loc[d["RemodAge"] < 0, "RemodAge"] = 0
    if "TotalBsmtSF" in d.columns and "GrLivArea" in d.columns:
        d["TotalSF"] = d["TotalBsmtSF"] + d["GrLivArea"]
    
    dc = ["Id", "Utilities", "YearBuilt", "YearRemodAdd"]
    dc = [c for c in dc if c in d.columns]
    d = d.drop(columns=dc)
    return d

def get_pl():
    npl = Pipeline(steps=[
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler())
    ])
    
    cpl = Pipeline(steps=[
        ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    pp = ColumnTransformer(transformers=[
        ("num", npl, make_column_selector(dtype_include=np.number)),
        ("cat", cpl, make_column_selector(dtype_exclude=np.number))
    ])
    
    pl = Pipeline(steps=[
        ("fe", FunctionTransformer(fe_func, validate=False)),
        ("pp", pp)
    ])
    return pl
