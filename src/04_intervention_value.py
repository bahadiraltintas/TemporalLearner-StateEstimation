#!/usr/bin/env python3
"""State-dependent intervention-value model using explicit activity × learner-state interactions.

Important: this is predictive, not causal. Each student has only one observed activity/gain pair.
"""
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from config import *

OUT=RESULTS/"intervention_value"; OUT.mkdir(parents=True,exist_ok=True)
TARGET="expected_learning_gain"
ACTION="recommended_activity"

NUM=[c for c in FULL if c not in ["gender","preferred_learning_type"]]
CAT=["gender","preferred_learning_type",ACTION]

def make_design(df):
    x=df[FULL+[ACTION]].copy()
    blocks=[x]
    for a in ACTIVITIES:
        tag=a.lower().replace(" ","_")
        mask=(x[ACTION]==a).astype(float)
        blocks.append(pd.DataFrame({
            f"{tag}__{c}": x[c].to_numpy()*mask.to_numpy()
            for c in NUM
        }, index=x.index))
    return pd.concat(blocks,axis=1)

def make_model():
    dummy=make_design(pd.DataFrame([{**{c:0 for c in FULL},ACTION:ACTIVITIES[0]}]))
    num=[c for c in dummy.columns if c not in CAT]
    prep=ColumnTransformer([
        ("num",Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]),num),
        ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore"))]),CAT)
    ])
    return Pipeline([("prep",prep),("model",Ridge(alpha=10.0))])

df=pd.read_csv(DATA)
y=df[TARGET].astype(float)
tr_idx,te_idx=train_test_split(np.arange(len(df)),test_size=TEST_SIZE,random_state=SEED)
X=make_design(df)
model=make_model()
model.fit(X.iloc[tr_idx],y.iloc[tr_idx])
pred=model.predict(X.iloc[te_idx])
metrics={
    "n_train":len(tr_idx),"n_test":len(te_idx),
    "MAE":float(mean_absolute_error(y.iloc[te_idx],pred)),
    "RMSE":float(mean_squared_error(y.iloc[te_idx],pred)**0.5),
    "R2":float(r2_score(y.iloc[te_idx],pred))
}
cv=KFold(n_splits=5,shuffle=True,random_state=SEED)
cv_rmse=float((-cross_val_score(model,X.iloc[tr_idx],y.iloc[tr_idx],cv=cv,scoring="neg_root_mean_squared_error")).mean())
cv_r2=float(cross_val_score(model,X.iloc[tr_idx],y.iloc[tr_idx],cv=cv,scoring="r2").mean())

# Fit on all observations only for generating the state-dependent candidate-value matrix.
model.fit(X,y)
rows=[]
for _,r in df.iterrows():
    candidates=[]
    for a in ACTIVITIES:
        d={c:r[c] for c in FULL}; d[ACTION]=a; candidates.append(d)
    scores=model.predict(make_design(pd.DataFrame(candidates)))
    for a,s in zip(ACTIVITIES,scores):
        rows.append({
            "student_id":int(r.student_id),"activity":a,
            "estimated_gain":float(s),"observed_activity":r[ACTION],
            "observed_gain":float(r[TARGET])
        })
scores_df=pd.DataFrame(rows)
scores_df.to_csv(OUT/"student_activity_value_matrix.csv",index=False)

rec=scores_df.loc[scores_df.groupby("student_id").estimated_gain.idxmax()].copy()
rec=rec.rename(columns={"activity":"selected_activity","estimated_gain":"selected_estimated_gain"})
rec=rec[["student_id","selected_activity","selected_estimated_gain","observed_activity","observed_gain"]]
rec["selection_matches_observed_policy"]=rec.selected_activity.eq(rec.observed_activity)
rec.to_csv(OUT/"state_dependent_decisions.csv",index=False)

pivot=scores_df.pivot(index="student_id",columns="activity",values="estimated_gain")
vals=np.sort(pivot.values,axis=1)[:,::-1]
summary=pd.DataFrame([{
    "mean_selected_estimated_gain":float(vals[:,0].mean()),
    "median_selected_estimated_gain":float(np.median(vals[:,0])),
    "mean_top1_top2_margin":float((vals[:,0]-vals[:,1]).mean()),
    "policy_match_rate_vs_observed":float(rec.selection_matches_observed_policy.mean()),
    "n_unique_selected_activities":int(rec.selected_activity.nunique()),
    "cv_RMSE_train":cv_rmse,"cv_R2_train":cv_r2,**metrics
}])
summary.to_csv(OUT/"intervention_value_summary.csv",index=False)
(OUT/"README.md").write_text(
"""# State-dependent intervention value

An action-conditioned Ridge model uses explicit activity × learner-state interactions
to estimate `G_hat(S_t, a)` for five candidate activities. Guided Review is excluded
because it has only three observations.

**Critical limitation:** the frozen dataset contains only one observed activity and one
observed gain per student. There are no counterfactual outcomes for unchosen activities.
Therefore these are conditional predictive estimates, not causal treatment effects.
""",encoding="utf-8")
print(json.dumps({**metrics,"cv_RMSE":cv_rmse,"cv_R2":cv_r2,
                  "match_rate":float(rec.selection_matches_observed_policy.mean()),
                  "n_unique_selected":int(rec.selected_activity.nunique())},indent=2))
