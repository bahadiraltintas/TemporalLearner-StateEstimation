from pathlib import Path
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
STEPS = [
    "01_prepare_data.py",
    "02_leakage_audit.py",
    "03_train_predictive_models.py",
    "04_intervention_value.py",
    "05A_state_transition_simulation.py",
    "05_closed_loop_simulation.py",
    "06_interpretability.py",
    "07_generate_report_tables.py",
    "08_uncertainty_analysis.py",
    "09_intervention_support_and_potential_outcomes.py",
    "10_simulation_sensitivity.py",
    "11_risk_calibration.py",
    "12_generate_dag.py",
]

env = dict(os.environ)
env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
env.update({"OMP_NUM_THREADS":"1", "MKL_NUM_THREADS":"1", "OPENBLAS_NUM_THREADS":"1", "NUMEXPR_NUM_THREADS":"1"})

for step in STEPS:
    print("\n" + "="*72 + f"\nRUNNING {step}\n" + "="*72, flush=True)
    subprocess.run([sys.executable, str(ROOT / "src" / step)], cwd=ROOT, env=env, check=True)

print("\nFINAL PIPELINE COMPLETE", flush=True)
