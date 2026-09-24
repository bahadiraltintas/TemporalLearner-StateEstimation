from pathlib import Path
import pandas as pd
from config import DATA, FULL, TARGETS, RESULTS

df = pd.read_csv(DATA)
rows = []
rules = {
    "performance": set(FULL) - {"final_grade"},
    "risk": set(FULL) - {"learning_risk_score","learning_risk_level"},
    "topic": set(FULL) - {"weakest_topic"},
    "recommendation": set(FULL) - {"recommended_activity","weakest_topic","recommended_duration_min","expected_learning_gain"},
    "gain": set(FULL) - {"expected_learning_gain","recommended_activity","recommended_duration_min"},
}
for task, features in rules.items():
    target = TARGETS[task]
    forbidden = [c for c in FULL if c not in features and c != target]
    rows.append({
        "task":task, "target":target, "n_features":len(features),
        "forbidden_target_like_variables": ";".join(forbidden)
    })
out = RESULTS/"leakage_audit"; out.mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv(out/"predictor_target_audit.csv", index=False)

topic_cols = ["conceptual_understanding","problem_solving","application_skills","critical_thinking","advanced_topics"]
argmin = df[topic_cols].idxmin(axis=1).map({
    "conceptual_understanding":"Conceptual Understanding",
    "problem_solving":"Problem Solving",
    "application_skills":"Application Skills",
    "critical_thinking":"Critical Thinking",
    "advanced_topics":"Advanced Topics",
})
audit = pd.DataFrame([{
    "weakest_topic_exact_argmin_match": bool((argmin == df.weakest_topic).all()),
    "recommendation_is_synthetic_rule_target": True,
    "interpretation": "Topic and recommendation results reconstruct the synthetic data-generating structure; they are not empirical pedagogical validation."
}])
audit.to_csv(out/"leakage_audit_summary.csv", index=False)
print(audit.to_string(index=False))
