from pathlib import Path
import pandas as pd
from config import RESULTS

out=RESULTS/"paper_tables"; out.mkdir(parents=True,exist_ok=True)

# Compact publication-oriented tables.
reg=pd.read_csv(RESULTS/"prediction_model_comparison.csv")
cls=pd.read_csv(RESULTS/"classification_model_comparison.csv")
abl=pd.read_csv(RESULTS/"performance_ablation.csv")
iv=pd.read_csv(RESULTS/"intervention_value/intervention_value_model_metrics.csv")
dec=pd.read_csv(RESULTS/"intervention_value/decision_summary.csv")
loop=pd.read_csv(RESULTS/"closed_loop/cycle10_summary.csv")
changes=pd.read_csv(RESULTS/"closed_loop/recommendation_change_rates.csv")

reg.to_csv(out/"Table_1_regression.csv",index=False)
cls.to_csv(out/"Table_2_classification.csv",index=False)
abl.to_csv(out/"Table_3_ablation.csv",index=False)
iv.to_csv(out/"Table_4_intervention_value_model.csv",index=False)
dec.to_csv(out/"Table_5_intervention_decision_summary.csv",index=False)
loop.to_csv(out/"Table_6_closed_loop_cycle10.csv",index=False)
changes.to_csv(out/"Table_7_recommendation_change_rates.csv",index=False)

readme = """# Final Pipeline Outputs

All tables in this folder are generated from the frozen synthetic dataset and the deterministic
train/test split. They are intended to map directly to the manuscript's Results section.

Important interpretation:
- These are computational proof-of-concept results.
- Recommendation and learning-gain targets are synthetic.
- Closed-loop learning responses are simulated.
- No causal educational-effectiveness claim is supported.
"""
(out/"README.md").write_text(readme,encoding="utf-8")
print("Paper tables generated.")


print("Report tables generated.", flush=True)
