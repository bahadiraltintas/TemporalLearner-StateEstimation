#!/usr/bin/env python3
"""Student-level percentile bootstrap uncertainty for principal fixed-test metrics.

The independent 200-student test partition is fixed. Bootstrap resamples are drawn
at the student level with replacement; no model is refit. This quantifies sampling
variability conditional on the fitted models and fixed holdout, rather than producing
population-level external-validation intervals.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Lasso, Ridge, LinearRegression, LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score, balanced_accuracy_score
from config import *

B=1000
BOOT_SEED=42042
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results"/"uncertainty"; OUT.mkdir(parents=True,exist_ok=True)

df=pd.read_csv(DATA)
split=pd.read_csv(RESULTS/"data/train_test_split_ids.csv")
train_ids=split.loc[split.partition=="train","student_id"]
test_ids=split.loc[split.partition=="test","student_id"]
tr=df[df.student_id.isin(train_ids)].copy(); te=df[df.student_id.isin(test_ids)].copy()

def pipe(est, cols=FULL):
    X=tr[cols]
    cat=X.select_dtypes(include=["object","category"]).columns.tolist()
    num=[c for c in cols if c not in cat]
    prep=ColumnTransformer([
        ("num",Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]),num),
        ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore"))]),cat),
    ])
    return Pipeline([("prep",prep),("model",est)])

rng=np.random.default_rng(BOOT_SEED)
idx=rng.integers(0,len(te),size=(B,len(te)))

def ci_metric(metric,y,p):
    vals=np.empty(B)
    for i in range(B): vals[i]=metric(y[idx[i]],p[idx[i]])
    return np.quantile(vals,[.025,.975])

def ci_diff(metric,y,p1,p2):
    vals=np.empty(B)
    for i in range(B):
        ii=idx[i]; vals[i]=metric(y[ii],p1[ii])-metric(y[ii],p2[ii])
    return np.quantile(vals,[.025,.975]), float(vals.mean())

rows=[]
reg_specs={"Lasso":Lasso(alpha=.01,max_iter=20000),"Ridge":Ridge(alpha=1.0),"Linear Regression":LinearRegression()}
reg_preds={}
y=te.final_grade.to_numpy(float)
for name,est in reg_specs.items():
    m=pipe(est); m.fit(tr[FULL],tr.final_grade); p=m.predict(te[FULL]); reg_preds[name]=p
    for metric_name,metric,value in [("MAE",mean_absolute_error,mean_absolute_error(y,p)),("RMSE",lambda a,b:mean_squared_error(a,b)**.5,mean_squared_error(y,p)**.5),("R2",r2_score,r2_score(y,p))]:
        lo,hi=ci_metric(metric,y,p); rows.append(["final_grade",name,metric_name,value,lo,hi,"student-level percentile bootstrap",B,BOOT_SEED])

# Sequential ablation R2 differences on the same test students.
groups=[ACADEMIC, ACADEMIC+BEHAVIORAL, ACADEMIC+BEHAVIORAL+PSYCHOLOGICAL, FULL]
aps=[]
for cols in groups:
    m=pipe(Ridge(alpha=1.0),cols);m.fit(tr[cols],tr.final_grade);aps.append(m.predict(te[cols]))
for i in range(1,4):
    ci,mean=ci_diff(r2_score,y,aps[i],aps[i-1])
    rows.append(["ablation_R2_difference",f"step_{i}","R2_increment",mean,ci[0],ci[1],"paired student-level percentile bootstrap",B,BOOT_SEED])

for a,b in [("Lasso","Ridge"),("Lasso","Linear Regression"),("Ridge","Linear Regression")]:
    ci,mean=ci_diff(r2_score,y,reg_preds[a],reg_preds[b])
    rows.append(["regression_R2_difference",f"{a} minus {b}","R2_difference",mean,ci[0],ci[1],"paired student-level percentile bootstrap",B,BOOT_SEED])

# Classification intervals for principal models.
cls_specs=[
    ("risk","SVM",SVC(C=2.0,random_state=SEED)),
    ("risk","Balanced Logistic Regression",LogisticRegression(max_iter=5000,class_weight="balanced")),
    ("topic","Logistic Regression",LogisticRegression(max_iter=5000)),
    ("recommendation","Logistic Regression",LogisticRegression(max_iter=5000)),
]
targets={"risk":"learning_risk_level","topic":"weakest_topic","recommendation":"recommended_activity"}
for task,name,est in cls_specs:
    target=targets[task]; m=pipe(est);m.fit(tr[FULL],tr[target]); yb=te[target].to_numpy();p=m.predict(te[FULL])
    labels=np.unique(yb)
    metrics={"accuracy":accuracy_score,"macro_F1":lambda a,b,labels=labels:f1_score(a,b,average="macro",labels=labels,zero_division=0),"balanced_accuracy":balanced_accuracy_score}
    for mn,metric in metrics.items():
        val=metric(yb,p); lo,hi=ci_metric(metric,yb,p);rows.append([task,name,mn,val,lo,hi,"student-level percentile bootstrap",B,BOOT_SEED])

out=pd.DataFrame(rows,columns=["analysis","model_or_step","metric","estimate","ci_low","ci_high","bootstrap_method","n_bootstrap","seed"])
out.to_csv(OUT/"principal_metric_bootstrap_cis.csv",index=False)
print(out.to_string(index=False))
