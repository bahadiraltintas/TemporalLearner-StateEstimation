from pathlib import Path
import hashlib
import pandas as pd
from config import DATA, RESULTS, SEED, TEST_SIZE
from sklearn.model_selection import train_test_split

df = pd.read_csv(DATA)
assert len(df) == 1000
assert df.isna().sum().sum() == 0
assert df.student_id.is_unique

train_ids, test_ids = train_test_split(
    df.student_id.astype(int), test_size=TEST_SIZE, random_state=SEED
)
split = pd.DataFrame({"student_id": sorted(test_ids), "partition":"test"})
split2 = pd.DataFrame({"student_id": sorted(train_ids), "partition":"train"})
split = pd.concat([split2, split], ignore_index=True).sort_values("student_id")
out = RESULTS/"data"
out.mkdir(parents=True, exist_ok=True)
split.to_csv(out/"train_test_split_ids.csv", index=False)

sha = hashlib.sha256(DATA.read_bytes()).hexdigest()
manifest = out/"RUN_DATA_MANIFEST.md"
manifest.write_text(
    f"# Final Run Data Manifest\n\n"
    f"- Rows: {len(df)}\n- Columns: {len(df.columns)}\n"
    f"- Missing cells: {int(df.isna().sum().sum())}\n"
    f"- Duplicate rows: {int(df.duplicated().sum())}\n"
    f"- Duplicate student_id: {int(df.student_id.duplicated().sum())}\n"
    f"- SHA-256: `{sha}`\n- Seed: {SEED}\n- Test size: {TEST_SIZE}\n",
    encoding="utf-8"
)
print("Prepared frozen dataset and deterministic train/test split.")
