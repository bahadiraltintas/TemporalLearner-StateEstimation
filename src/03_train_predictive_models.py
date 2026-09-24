import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso, Ridge, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, accuracy_score,
    f1_score, balanced_accuracy_score, precision_score, recall_score
)
from config import *
from utils import make_preprocessor

df = pd.read_csv(DATA)
train_ids = pd.read_csv(RESULTS/"data/train_test_split_ids.csv")
train_ids = train_ids.loc[train_ids.partition=="train","student_id"]
test_ids = train_ids = None
split = pd.read_csv(RESULTS/"data/train_test_split_ids.csv")
train_ids = split.loc[split.partition=="train","student_id"]
test_ids = split.loc[split.partition=="test","student_id"]
tr = df[df.student_id.isin(train_ids)].copy()
te = df[df.student_id.isin(test_ids)].copy()

OUT = RESULTS/"models"; OUT.mkdir(parents=True, exist_ok=True)

def fit_pipe(est, X):
    return Pipeline([("prep", make_preprocessor(X)), ("model", est)])

# ----- Regression -----
reg_models = {
    "Lasso": Lasso(alpha=0.01, max_iter=20000),
    "Ridge": Ridge(alpha=1.0),
    "Linear Regression": __import__("sklearn").linear_model.LinearRegression(),
    "Gradient Boosting": GradientBoostingRegressor(random_state=SEED),
    "Random Forest": RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=-1),
    "KNN": KNeighborsRegressor(n_neighbors=15),
    "SVR": SVR(C=10.0),
    "Decision Tree": DecisionTreeRegressor(random_state=SEED),
}
reg_rows = []
for target_name in ["final_grade","expected_learning_gain"]:
    for name, est in reg_models.items():
        p = fit_pipe(est, tr[FULL])
        p.fit(tr[FULL], tr[target_name])
        pred = p.predict(te[FULL])
        reg_rows.append({
            "target":target_name,"model":name,
            "MAE":mean_absolute_error(te[target_name],pred),
            "RMSE":mean_squared_error(te[target_name],pred)**0.5,
            "R2":r2_score(te[target_name],pred)
        })
        if (target_name=="final_grade" and name=="Lasso") or (target_name=="expected_learning_gain" and name=="Lasso"):
            import joblib
            joblib.dump(p, OUT/f"{target_name}_Lasso.joblib")
pd.DataFrame(reg_rows).to_csv(RESULTS/"prediction_model_comparison.csv", index=False)

# ----- Classification -----
cls_specs = {
    "risk": {
        "target":"learning_risk_level",
        "models":{
            "Logistic Regression":LogisticRegression(max_iter=5000),
            "SVM":SVC(C=2.0, probability=True, random_state=SEED),
            "Balanced Logistic Regression":LogisticRegression(max_iter=5000, class_weight="balanced"),
            "Balanced SVM":SVC(C=2.0, class_weight="balanced", probability=True, random_state=SEED),
            "KNN":KNeighborsClassifier(n_neighbors=15),
            "Decision Tree":DecisionTreeClassifier(random_state=SEED),
        }
    },
    "topic":{
        "target":"weakest_topic",
        "models":{
            "Logistic Regression":LogisticRegression(max_iter=5000),
            "SVM":SVC(C=2.0, random_state=SEED),
            "Decision Tree":DecisionTreeClassifier(random_state=SEED),
            "KNN":KNeighborsClassifier(n_neighbors=15)
        }
    },
    "recommendation":{
        "target":"recommended_activity",
        "models":{
            "Logistic Regression":LogisticRegression(max_iter=5000),
            "SVM":SVC(C=2.0, random_state=SEED),
            "Decision Tree":DecisionTreeClassifier(random_state=SEED),
            "KNN":KNeighborsClassifier(n_neighbors=15)
        }
    }
}
cls_rows=[]
for task,spec in cls_specs.items():
    for name,est in spec["models"].items():
        p=fit_pipe(est,tr[FULL]); p.fit(tr[FULL],tr[spec["target"]])
        pred=p.predict(te[FULL])
        cls_rows.append({
            "task":task,"model":name,
            "accuracy":accuracy_score(te[spec["target"]],pred),
            "macro_f1":f1_score(te[spec["target"]],pred,average="macro"),
            "balanced_accuracy":balanced_accuracy_score(te[spec["target"]],pred),
            "precision_macro":precision_score(te[spec["target"]],pred,average="macro",zero_division=0),
            "recall_macro":recall_score(te[spec["target"]],pred,average="macro",zero_division=0)
        })
        if name in ["SVM","Logistic Regression"] and task in ["risk","topic","recommendation"]:
            import joblib
            suffix = {"risk":"risk_SVM","topic":"topic_Logistic","recommendation":"recommendation_Logistic"}[task]
            joblib.dump(p, OUT/f"{suffix}.joblib")
pd.DataFrame(cls_rows).to_csv(RESULTS/"classification_model_comparison.csv",index=False)

# ----- Performance ablation -----
groups=[ACADEMIC, ACADEMIC+BEHAVIORAL, ACADEMIC+BEHAVIORAL+PSYCHOLOGICAL, FULL]
rows=[]
for cols in groups:
    p=Pipeline([("prep",make_preprocessor(tr[cols])),("model",Ridge(alpha=1.0))])
    p.fit(tr[cols],tr.final_grade)
    pred=p.predict(te[cols])
    rows.append({"feature_set":"Academic" if len(cols)==len(ACADEMIC) else
                 "Academic + Behavioral" if len(cols)==len(ACADEMIC)+len(BEHAVIORAL) else
                 "Academic + Behavioral + Psychological" if len(cols)==len(ACADEMIC)+len(BEHAVIORAL)+len(PSYCHOLOGICAL) else
                 "Full learner state + Topic",
                 "MAE":mean_absolute_error(te.final_grade,pred),
                 "RMSE":mean_squared_error(te.final_grade,pred)**0.5,
                 "R2":r2_score(te.final_grade,pred)})
pd.DataFrame(rows).to_csv(RESULTS/"performance_ablation.csv",index=False)
print("Predictive model evaluation complete.")
