import nbformat as nbf
import os
os.makedirs("notebooks", exist_ok=True)
nb = nbf.v4.new_notebook()
code = """\
import sys
sys.path.append('../src')
import pandas as pd
import numpy as np
from data import load_data, split_data
from evaluate import cv_eval, tune_model, eval_test
from models import get_models
import plots
import warnings
warnings.filterwarnings('ignore')

df = load_data()
Xtr, Xte, ytr, yte = split_data(df)

ms = get_models()
res = cv_eval(ms, Xtr, ytr)
bm_name = min(res, key=lambda k: res[k]['test_rmse'])
pl = ms[bm_name]

bm, bp, bsc = tune_model(pl, bm_name, Xtr, ytr)

tres, ypr = eval_test(bm, Xte, yte)

print('Best Model:', bm_name)
print('Best Params:', bp)
print('Test RMSE:', tres['rmse'])
"""
nb['cells'] = [nbf.v4.new_code_cell(code)]
with open('notebooks/model_building.ipynb', 'w') as f:
    nbf.write(nb, f)
