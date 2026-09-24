from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

def make_preprocessor(X):
    cat = X.select_dtypes(include=["object","category"]).columns.tolist()
    num = [c for c in X.columns if c not in cat]
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat),
    ])

def load_data(path):
    return pd.read_csv(path)

def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def bootstrap_ci(values, seed=42, n_boot=10000, alpha=0.05):
    rng = np.random.default_rng(seed)
    x = np.asarray(values, dtype=float)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = rng.choice(x, size=len(x), replace=True).mean()
    return float(np.quantile(means, alpha/2)), float(np.quantile(means, 1-alpha/2))
