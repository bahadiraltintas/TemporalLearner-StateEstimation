#!/usr/bin/env python3
"""Intervention-value support diagnostics and independent potential-outcome benchmark.

The support analysis is descriptive. A candidate state-action pair is flagged by two
operational criteria: (1) any numeric state variable outside the training min-max range
for that action; and (2) nearest-neighbour distance above the 95th percentile of
leave-one-out training nearest-neighbour distances for that action.

The potential-outcome benchmark is an independent synthetic environment with known
potential outcomes. It is not a validation of the original educational dataset; it is
a methodological stress test for action-conditioned prediction and ranking.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from config import *

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"results"/"intervention_support"; OUT.mkdir(parents=True,exist_ok=True)
POUT=ROOT/"results"/"potential_outcome_benchmark"; POUT.mkdir(parents=True,exist_ok=True)
ACTIVITIES=ACTIVITIES
ELIGIBLE=ACTIVITIES

df=pd.read_csv(DATA); split=pd.read_csv(RESULTS/"data/train_test_split_ids.csv")
tr=df[df.student_id.isin(split.loc[split.partition=="train","student_id"])].copy()
te=df[df.student_id.isin(split.loc[split.partition=="test","student_id"])].copy()
num_cols=[c for c in FULL if df[c].dtype!='object']

# ----- Support / overlap -----
support=[]
for a in ELIGIBLE:
    tra=tr[tr.recommended_activity==a]
    Xtr=tra[num_cols].to_numpy(float); Xte=te[num_cols].to_numpy(float)
    mins=Xtr.min(axis=0); maxs=Xtr.max(axis=0)
    range_out=((Xte<mins)|(Xte>maxs)).any(axis=1)
    scaler=StandardScaler().fit(Xtr); ztr=scaler.transform(Xtr); zte=scaler.transform(Xte)
    nn=NearestNeighbors(n_neighbors=2).fit(ztr)
    dtr=nn.kneighbors(ztr,return_distance=True)[0][:,1]
    threshold=float(np.quantile(dtr,.95))
    dte=NearestNeighbors(n_neighbors=1).fit(ztr).kneighbors(zte,return_distance=True)[0].ravel()
    nn_out=dte>threshold
    support.append({"activity":a,"training_observations":len(tra),"test_candidate_states":len(te),
                    "minmax_outside_n":int(range_out.sum()),"minmax_outside_pct":float(range_out.mean()*100),
                    "nn95_threshold":threshold,"nn_outside_n":int(nn_out.sum()),"nn_outside_pct":float(nn_out.mean()*100),
                    "median_nearest_neighbor_distance":float(np.median(dte))})
support_df=pd.DataFrame(support)
support_df.to_csv(OUT/"action_support_overlap.csv",index=False)
summary=pd.DataFrame([{
    "eligible_actions":len(ELIGIBLE),"training_observations":sum((tr.recommended_activity==a).sum() for a in ELIGIBLE),
    "evaluated_candidate_pairs":len(te)*len(ELIGIBLE),
    "minmax_outside_pct":sum(r["minmax_outside_n"] for r in support)/sum(r["test_candidate_states"] for r in support)*100,
    "nn95_outside_pct":sum(r["nn_outside_n"] for r in support)/sum(r["test_candidate_states"] for r in support)*100,
    "definition":"NN-outside = candidate-state nearest-neighbour distance exceeds the 95th percentile of leave-one-out training NN distance within the same action."
}])
summary.to_csv(OUT/"support_summary.csv",index=False)

# ----- Action-conditioned model per-action performance -----
NUM=[c for c in FULL if c not in ["gender","preferred_learning_type"]]
def design(x):
    x=x[FULL+["recommended_activity"]].copy(); parts=[x]
    for a in ELIGIBLE:
        mask=(x.recommended_activity==a).astype(float); tag=a.lower().replace(" ","_")
        parts.append(pd.DataFrame({f"{tag}__{c}":x[c].to_numpy()*mask.to_numpy() for c in NUM},index=x.index))
    return pd.concat(parts,axis=1)
dummy=design(pd.DataFrame([{**{c:0 for c in FULL},"recommended_activity":ELIGIBLE[0]}]))
cat=["gender","preferred_learning_type","recommended_activity"]; num=[c for c in dummy.columns if c not in cat]
pre=ColumnTransformer([("num",Pipeline([("imp",SimpleImputer(strategy="median")),("sc",StandardScaler())]),num),
                       ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("oh",OneHotEncoder(handle_unknown="ignore"))]),cat)])
model=Pipeline([("pre",pre),("model",Ridge(alpha=10.0))]).fit(design(tr),tr.expected_learning_gain)
pred=model.predict(design(te)); perf=[]
for a in ELIGIBLE:
    s=te.recommended_activity==a; yt=te.loc[s,"expected_learning_gain"].to_numpy(); yp=pred[s]
    perf.append({"activity":a,"test_observations":int(s.sum()),"MAE":mean_absolute_error(yt,yp),"RMSE":mean_squared_error(yt,yp)**.5,"R2":r2_score(yt,yp) if len(yt)>1 else np.nan})
pd.DataFrame(perf).to_csv(OUT/"per_action_test_performance.csv",index=False)

# ----- Independent known-potential-outcome environment -----
rng=np.random.default_rng(20260924)
base={"Concept Review":.20,"Practice Questions":.35,"Application Exercise":.30,"Critical Thinking Activity":.25,"Advanced Practice":.40}
topic_for={"Concept Review":"conceptual_understanding","Practice Questions":"problem_solving","Application Exercise":"application_skills","Critical Thinking Activity":"critical_thinking","Advanced Practice":"advanced_topics"}
rows=[]
for _,r in df.iterrows():
    for a in ELIGIBLE:
        mastery=float(r[topic_for[a]])/100; sr=float(r.self_regulation)/100; mot=float(r.motivation)/100; anx=float(r.academic_anxiety)/100; comp=float(r.recommendation_completion_rate)/100
        y=1.0+2.0*(1-mastery)+.55*sr+.30*mot-.20*anx+base[a]+.50*(1-mastery)*comp
        rows.append((int(r.student_id),a,y))
pot=pd.DataFrame(rows,columns=["student_id","activity","true_potential_gain"])
fact=pot.merge(df[["student_id","recommended_activity"]],on="student_id")
fact=fact[fact.activity==fact.recommended_activity].copy(); fact["factual_gain"]=fact.true_potential_gain+rng.normal(0,.35,len(fact))
train=fact[fact.student_id.isin(tr.student_id)].merge(tr,on="student_id",suffixes=("","_state"))
# use observed state columns from tr; action is the original recommended_activity
m=Pipeline([("pre",pre),("model",Ridge(alpha=10.0))]).fit(design(train),train.factual_gain)
cands=[]
for _,r in te.iterrows():
    for a in ELIGIBLE:
        rr=r.copy(); rr["recommended_activity"]=a; cands.append(rr)
cands=pd.DataFrame(cands)
cands["predicted_potential_gain"]=m.predict(design(cands))
ev=cands[["student_id","recommended_activity","predicted_potential_gain"]].merge(pot.rename(columns={"activity":"recommended_activity"}),on=["student_id","recommended_activity"])
mae=mean_absolute_error(ev.true_potential_gain,ev.predicted_potential_gain); rmse=mean_squared_error(ev.true_potential_gain,ev.predicted_potential_gain)**.5; r2=r2_score(ev.true_potential_gain,ev.predicted_potential_gain)
true_best=ev.loc[ev.groupby("student_id").true_potential_gain.idxmax(),["student_id","recommended_activity"]].rename(columns={"recommended_activity":"true_best_activity"})
pred_best=ev.loc[ev.groupby("student_id").predicted_potential_gain.idxmax(),["student_id","recommended_activity"]].rename(columns={"recommended_activity":"predicted_best_activity"})
rank=true_best.merge(pred_best,on="student_id"); top1=float(rank.true_best_activity.eq(rank.predicted_best_activity).mean())
pd.DataFrame([{"n_train_students":len(tr),"n_test_students":len(te),"candidate_actions":len(ELIGIBLE),"candidate_pairs_test":len(ev),"MAE":mae,"RMSE":rmse,"R2":r2,"top1_best_action_recovery":top1,"noise_sd_factual_gain":.35,"seed":20260924}]).to_csv(POUT/"potential_outcome_benchmark_summary.csv",index=False)
ev.to_csv(POUT/"test_candidate_potential_outcomes.csv",index=False); rank.to_csv(POUT/"best_action_recovery.csv",index=False)
print(support_df.to_string(index=False)); print("\nPotential-outcome benchmark:",mae,rmse,r2,top1)
