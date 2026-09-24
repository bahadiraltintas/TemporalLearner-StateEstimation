from pathlib import Path
import hashlib

DATA = Path(__file__).resolve().parent / "data_frozen_v1" / "synthetic_learner_state_dataset_v1.0.csv"
EXPECTED = "7593b4561a2b72ff3b9753d28dceaffcd53e426bfa434b3381e16cabf7b5fed5"
actual = hashlib.sha256(DATA.read_bytes()).hexdigest()
print(f"Dataset: {DATA}")
print(f"SHA-256: {actual}")
if actual != EXPECTED:
    raise SystemExit("ERROR: dataset hash does not match the frozen v1.0 artifact.")
print("OK: frozen dataset verified.")
