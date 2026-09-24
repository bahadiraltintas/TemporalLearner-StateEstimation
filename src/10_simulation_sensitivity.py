#!/usr/bin/env python3
"""One-at-a-time sensitivity analysis for Simulation B.

Perturbations are fixed modeling assumptions, not parameters estimated from outcomes:
- targeted-topic update coefficient: 0.28, 0.35, 0.42
- non-target spillover: 0.04, 0.05, 0.06
- all ten diminishing-response factors multiplied by 0.80, 1.00, 1.20
Paired student-level percentile bootstrap uses 10,000 replicates and seed 42.
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from config import *

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"results"/"simulation_sensitivity"; OUT.mkdir(parents=True,exist_ok=True)
df=pd.read_csv(DATA); split=pd.read_csv(RESULTS/"data/train_test_split_ids.csv")
tr=df[df.student_id.isin(split.loc[split.partition=="train","student_id"])].copy(); te=df[df.student_id.isin(split.loc[split.partition=="test","student_id"])].copy().reset_index(drop=True)
ELIGIBLE=ACTIVITIES; TOPIC_COL_MAP={"Concept Review":"conceptual_understanding","Practice Questions":"problem_solving","Application Exercise":"application_skills","Critical Thinking Activity":"critical_thinking","Advanced Practice":"advanced_topics"}
NUM=[c for c in FULL if c not in ["gender","preferred_learning_type"]]
def design(x):
    x=x[FULL+["recommended_activity"]].copy(); parts=[x]
    for a in ELIGIBLE:
        mask=(x.recommended_activity==a).astype(float); tag=a.lower().replace(" ","_")
        parts.append(pd.DataFrame({f"{tag}__{c}":x[c].to_numpy()*mask.to_numpy() for c in NUM},index=x.index))
    return pd.concat(parts,axis=1)
dummy=design(pd.DataFrame([{**{c:0 for c in FULL},"recommended_activity":ELIGIBLE[0]}])); cat=["gender","preferred_learning_type","recommended_activity"]; num=[c for c in dummy.columns if c not in cat]
pre=ColumnTransformer([("num",Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]),num),("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore"))]),cat)])
model=Pipeline([("pre",pre),("model",Ridge(alpha=10.0))]).fit(design(tr),tr.expected_learning_gain)

def score(state):
    cs=[]
    for a in ELIGIBLE:
        c=state.copy(); c["recommended_activity"]=a; cs.append(c)
    return model.predict(design(pd.concat(cs,ignore_index=True))).reshape(len(ELIGIBLE),len(state)).T
DECAY=np.array([1.00,.72,.50,.40,.32,.27,.23,.20,.18,.16])
def run(target=.35,spill=.05,decay_mult=1.0):
    adaptive=te.copy(); static=te.copy(); static_action=np.array(ELIGIBLE)[score(adaptive).argmax(1)]; ca=np.zeros(len(te)); cs=np.zeros(len(te))
    for d in DECAY*decay_mult:
        sa=score(adaptive); ss=score(static); aa=np.array(ELIGIBLE)[sa.argmax(1)]
        for i in range(len(te)):
            a=aa[i]; s=static_action[i]; ag=max(0,float(sa[i,ELIGIBLE.index(a)])); sg=max(0,float(ss[i,ELIGIBLE.index(s)])); comp=float(te.iloc[i].recommendation_completion_rate)
            for state,a2,g in [(adaptive,a,ag*comp*d),(static,s,sg*comp*d)]:
                col=TOPIC_COL_MAP[a2]; delta=float(np.clip(g,0,8))*target; state.loc[i,col]=min(100,float(state.loc[i,col]+delta))
                for t in TOPIC_COL_MAP.values():
                    if t!=col: state.loc[i,t]=min(100,float(state.loc[i,t]+delta*spill))
            ca[i]+=ag*comp*d; cs[i]+=sg*comp*d
    return ca-cs

def ci(x):
    rng=np.random.default_rng(42); vals=np.empty(10000); x=np.asarray(x)
    for i in range(10000): vals[i]=rng.choice(x,size=len(x),replace=True).mean()
    return np.quantile(vals,[.025,.975])
scenarios=[("Target update",.28,.05,1), ("Target update",.35,.05,1),("Target update",.42,.05,1),("Spillover",.35,.04,1),("Spillover",.35,.05,1),("Spillover",.35,.06,1),("Decay multiplier",.35,.05,.8),("Decay multiplier",.35,.05,1.0),("Decay multiplier",.35,.05,1.2)]
rows=[]
for label,t,s,d in scenarios:
    diff=run(t,s,d); c=ci(diff)
    rows.append({"perturbation_group":label,"target_update":t,"spillover":s,"decay_multiplier":d,"mean_adaptive_minus_static":diff.mean(),"ci_low":c[0],"ci_high":c[1],"adaptive_higher_pct":(diff>0).mean()*100,"equal_pct":(diff==0).mean()*100,"adaptive_lower_pct":(diff<0).mean()*100})
pd.DataFrame(rows).to_csv(OUT/"simulation_B_sensitivity.csv",index=False)
print(pd.DataFrame(rows).to_string(index=False))
