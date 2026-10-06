import os
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

def load_data():
    df = fetch_openml(name="house_prices", as_frame=True, parser="auto").frame
    return df

def save_raw(df, p):
    df.to_csv(p, index=False)

def split_data(df):
    X = df.drop(columns=["SalePrice"])
    y = df["SalePrice"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    return Xtr, Xte, ytr, yte
