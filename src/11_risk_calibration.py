#!/usr/bin/env python3
"""Risk calibration diagnostics for balanced logistic regression on the fixed test set."""
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from config import *
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"results"/"uncertainty"; FIG=ROOT/"figures"; OUT.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
df=pd.read_csv(DATA); split=pd.read_csv(RESULTS/"data/train_test_split_ids.csv")
tr=df[df.student_id.isin(split.loc[split.partition=="train","student_id"])]; te=df[df.student_id.isin(split.loc[split.partition=="test","student_id"])]
X=tr[FULL]; cat=X.select_dtypes(include=["object","category"]).columns.tolist(); num=[c for c in FULL if c not in cat]
pre=ColumnTransformer([("num",Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]),num),("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore"))]),cat)])
m=Pipeline([("pre",pre),("model",LogisticRegression(max_iter=5000,class_weight="balanced"))]).fit(tr[FULL],tr.learning_risk_level)
proba=m.predict_proba(te[FULL]); classes=m.classes_; y=te.learning_risk_level.to_numpy()
rows=[]
for j,c in enumerate(classes):
    yy=(y==c).astype(int); pp=proba[:,j]; rows.append({"class":c,"Brier_score":brier_score_loss(yy,pp),"n_test":int(yy.sum()),"mean_predicted_probability":pp.mean()})
pd.DataFrame(rows).to_csv(OUT/"risk_calibration_brier.csv",index=False)
# simple reliability curves, 5 bins
fig,ax=plt.subplots(figsize=(6.2,4.8))
for j,c in enumerate(classes):
    yy=(y==c).astype(int); pp=proba[:,j]; bins=np.linspace(0,1,6); xs=[]; ys=[]
    for lo,hi in zip(bins[:-1],bins[1:]):
        mask=(pp>=lo)&(pp<=hi) if hi==1 else (pp>=lo)&(pp<hi)
        if mask.sum(): xs.append(pp[mask].mean()); ys.append(yy[mask].mean())
    ax.plot(xs,ys,marker='o',label=c)
ax.plot([0,1],[0,1],linestyle='--',linewidth=1)
ax.set_xlabel('Mean predicted probability'); ax.set_ylabel('Observed frequency'); ax.set_title('Risk-label calibration: balanced logistic regression')
ax.legend(); fig.tight_layout(); fig.savefig(FIG/'risk_calibration.png',dpi=220); plt.close(fig)
