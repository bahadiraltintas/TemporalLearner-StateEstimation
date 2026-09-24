import joblib
import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance
from config import *
from utils import make_preprocessor

df=pd.read_csv(DATA)
split=pd.read_csv(RESULTS/"data/train_test_split_ids.csv")
tr=df[df.student_id.isin(split.loc[split.partition=="train","student_id"])].copy()
te=df[df.student_id.isin(split.loc[split.partition=="test","student_id"])].copy()
out=RESULTS/"interpretability"; out.mkdir(parents=True,exist_ok=True)

# Permutation importance is used as the reproducible baseline. SHAP can be added as an optional
# dependency without changing the core pipeline.
models = {
    "performance_lasso":("final_grade","regression"),
    "risk_svm":("learning_risk_level","classification"),
    "topic_logistic":("weakest_topic","classification"),
    "recommendation_logistic":("recommended_activity","classification"),
    "learning_gain_lasso":("expected_learning_gain","regression"),
}
files = {
    "performance_lasso":"final_grade_Lasso.joblib",
    "risk_svm":"risk_SVM.joblib",
    "topic_logistic":"topic_Logistic.joblib",
    "recommendation_logistic":"recommendation_Logistic.joblib",
    "learning_gain_lasso":"expected_learning_gain_Lasso.joblib",
}
rows=[]
for name,(target,kind) in models.items():
    model=joblib.load(RESULTS/"models"/files[name])
    r=permutation_importance(model,te[FULL],te[target],n_repeats=5,random_state=SEED,n_jobs=-1,scoring="r2" if kind=="regression" else "f1_macro")
    # Pipeline permutation importance is on original feature columns.
    imp=pd.DataFrame({"feature":FULL,"importance_mean":r.importances_mean,"importance_std":r.importances_std})
    imp=imp.sort_values("importance_mean",ascending=False)
    imp.to_csv(out/f"permutation_{name}.csv",index=False)
    for rank,(_,x) in enumerate(imp.head(15).iterrows(),1):
        rows.append({"model":name,"rank":rank,"feature":x.feature,"importance_mean":x.importance_mean})
pd.DataFrame(rows).to_csv(out/"interpretability_summary.csv",index=False)
print("Permutation-importance interpretability outputs written.", flush=True)

# Force clean termination after all result files are written.
import os
os._exit(0)
