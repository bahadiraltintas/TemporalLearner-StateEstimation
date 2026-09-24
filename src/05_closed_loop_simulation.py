#!/usr/bin/env python3
"""Ten-cycle closed-loop simulation using state-dependent intervention value."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from config import *
from utils import bootstrap_ci

OUT=RESULTS/"intervention_value_loop"; OUT.mkdir(parents=True,exist_ok=True)
ACTIVITIES=ACTIVITIES
TOPICS=list(TOPIC_COL.values())
DECAY=[1.00,.72,.50,.40,.32,.27,.23,.20,.18,.16]

def design(df):
    x=df[FULL+["recommended_activity"]].copy()
    parts=[x]
    NUM=[c for c in FULL if c not in ["gender","preferred_learning_type"]]
    for a in ACTIVITIES:
        mask=(x["recommended_activity"]==a).astype(float)
        parts.append(pd.DataFrame({
            f'{a.lower().replace(" ","_")}__{c}':x[c].to_numpy()*mask.to_numpy()
            for c in NUM
        },index=x.index))
    return pd.concat(parts,axis=1)

def fit(train):
    dummy=design(train.iloc[:1])
    CAT=["gender","preferred_learning_type","recommended_activity"]
    num=[c for c in dummy.columns if c not in CAT]
    pre=ColumnTransformer([
        ("num",Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]),num),
        ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore"))]),CAT)
    ])
    return Pipeline([("pre",pre),("model",Ridge(alpha=10.0))]).fit(design(train),train.expected_learning_gain)

def score(model,state):
    # Predict all candidate activities in one batch for substantially faster local execution.
    candidates=[]
    for a in ACTIVITIES:
        c=state.copy()
        c["recommended_activity"]=a
        candidates.append(c)
    all_candidates=pd.concat(candidates, ignore_index=True)
    pred=model.predict(design(all_candidates))
    return pred.reshape(len(ACTIVITIES), len(state)).T

def update(row,activity,gain):
    r=row.copy()
    target=ACTIVITY_TO_TOPIC[activity]
    col=TOPIC_COL[target]
    delta=float(np.clip(gain,0,8))*0.35
    r[col]=min(100,float(r[col]+delta))
    for t in TOPICS:
        if t!=col:
            r[t]=min(100,float(r[t]+delta*.05))
    return r

df=pd.read_csv(DATA)
tr_idx,te_idx=train_test_split(np.arange(len(df)),test_size=TEST_SIZE,random_state=SEED)
train=df.iloc[tr_idx].copy()
test=df.iloc[te_idx].copy().reset_index(drop=True)
model=fit(train)

initial=score(model,test)
initial_action=np.array(ACTIVITIES)[initial.argmax(1)]
static_action=initial_action.copy()
adaptive=test.copy()
static=test.copy()
records=[]

for cycle,decay in enumerate(DECAY,1):
    print(f"  simulation cycle {cycle}/{len(DECAY)}", flush=True)
    ascore=score(model,adaptive)
    sscore=score(model,static)
    aa=np.array(ACTIVITIES)[ascore.argmax(1)]
    for i in range(len(test)):
        a=aa[i]
        s=static_action[i]
        ag=max(0,float(ascore[i,ACTIVITIES.index(a)]))
        sg=max(0,float(sscore[i,ACTIVITIES.index(s)]))
        comp=float(test.loc[i,"recommendation_completion_rate"])
        again=ag*comp*decay
        sgain=sg*comp*decay
        adaptive.loc[i]=update(adaptive.loc[i],a,again)
        static.loc[i]=update(static.loc[i],s,sgain)
        records.append({
            "student_id":int(test.loc[i,"student_id"]),
            "cycle":cycle,
            "adaptive_activity":a,
            "static_activity":s,
            "adaptive_estimated_gain":ag,
            "static_estimated_gain":sg,
            "adaptive_simulated_gain":again,
            "static_simulated_gain":sgain
        })

traj=pd.DataFrame(records)
traj["adaptive_cumulative_gain"]=traj.groupby("student_id").adaptive_simulated_gain.cumsum()
traj["static_cumulative_gain"]=traj.groupby("student_id").static_simulated_gain.cumsum()
traj.to_csv(OUT/"student_cycle_trajectories.csv",index=False)

final=traj.groupby("student_id").tail(1).copy()
final["difference"]=final.adaptive_cumulative_gain-final.static_cumulative_gain
final.to_csv(OUT/"student_level_comparison.csv",index=False)

cyc=traj.groupby("cycle").agg(
    adaptive_mean_gain=("adaptive_simulated_gain","mean"),
    static_mean_gain=("static_simulated_gain","mean"),
    adaptive_mean_cumulative=("adaptive_cumulative_gain","mean"),
    static_mean_cumulative=("static_cumulative_gain","mean")
).reset_index()
cyc.to_csv(OUT/"cycle_summary.csv",index=False)

changes=traj.groupby("student_id").adaptive_activity.apply(
    lambda x:int((x.values[1:]!=x.values[:-1]).sum())
)
first=traj.groupby("student_id").adaptive_activity.first()
last=traj.groupby("student_id").adaptive_activity.last()
initial_final=first.ne(last)

report=pd.DataFrame([{
    "n_test_students":len(test),
    "cycles":len(DECAY),
    "adaptive_mean_final_cumulative_gain":final.adaptive_cumulative_gain.mean(),
    "static_mean_final_cumulative_gain":final.static_cumulative_gain.mean(),
    "mean_difference":final.difference.mean(),
    "sd_difference":final.difference.std(ddof=1),
    "adaptive_better_pct":(final.difference>0).mean()*100,
    "equal_pct":(final.difference==0).mean()*100,
    "adaptive_worse_pct":(final.difference<0).mean()*100,
    "mean_number_of_recommendation_changes":changes.mean(),
    "initial_to_final_change_pct":initial_final.mean()*100
}])
report.to_csv(OUT/"intervention_value_loop_summary.csv",index=False)

ci=bootstrap_ci(final.difference.to_numpy(),seed=SEED,n_boot=10000)
(OUT/"bootstrap_ci.txt").write_text(
    f"mean_difference={final.difference.mean():.6f}\n"
    f"bootstrap_95ci=[{ci[0]:.6f}, {ci[1]:.6f}]\n",
    encoding="utf-8"
)
print(report.to_string(index=False))
print(f"bootstrap_95ci=[{ci[0]:.6f}, {ci[1]:.6f}]", flush=True)

# Force clean termination after all result files are written.
# This avoids long-lived numerical thread-pool shutdown delays on some Linux environments.
import os
os._exit(0)
