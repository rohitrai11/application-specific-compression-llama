import pandas as pd
from itertools import combinations

def read_preds(path):
    df = pd.read_csv(path, sep=None, engine="python")

    for col in ["ground_truth", "prediction"]:
        df[col] = df[col].astype(str).str.strip().str.lower()

    return df

def jaccard(a, b):
    union = a | b
    return len(a & b) / len(union) if union else 1.0

def error_sets(df):
    overall = set(df.loc[df["ground_truth"] != df["prediction"], "id"])

    false_pos = set(df.loc[
        (df["ground_truth"] == "no") & (df["prediction"] == "yes"), "id"
    ])
    false_neg = set(df.loc[
        (df["ground_truth"] == "yes") & (df["prediction"] == "no"), "id"
    ])

    return {
        "overall": overall,
        "false_pos": false_pos,
        "false_neg": false_neg,
    }

# File paths
files = {
    "7B": "test_predictions_7B.csv",
    "13B": "test_predictions_13B.csv",
    "70B": "test_predictions_70B.csv",
}

# Load data
dfs = {name: read_preds(path) for name, path in files.items()}

# Sanity check (all IDs same)
id_sets = {name: set(df["id"]) for name, df in dfs.items()}
all_same_ids = len(set(map(tuple, id_sets.values()))) == 1
print("Same IDs across all files:", all_same_ids)

for name, df in dfs.items():
    print(f"{name}: {len(df)} rows")

# Compute error sets
errors = {name: error_sets(df) for name, df in dfs.items()}

# ---- Pairwise Jaccard ----
print("\nPairwise Jaccard similarity:")
for (m1, m2) in combinations(files.keys(), 2):
    print(f"\n{m1} vs {m2}")
    for k in ["overall", "false_pos", "false_neg"]:
        j = jaccard(errors[m1][k], errors[m2][k])
        inter = len(errors[m1][k] & errors[m2][k])
        union = len(errors[m1][k] | errors[m2][k])
        print(f"{k:10s}: {j:.4f} (intersection={inter}, union={union})")

# ---- All 3 overlap ----
print("\nCommon errors across ALL 3 models:")
common_all = errors["7B"]["overall"] & errors["13B"]["overall"] & errors["70B"]["overall"]
print("Count:", len(common_all))

# ---- Unique errors per model ----
print("\nUnique errors per model:")
for m in files.keys():
    others = set().union(*[
        errors[o]["overall"] for o in files.keys() if o != m
    ])
    unique = errors[m]["overall"] - others
    print(f"{m}: {len(unique)}")

# ---- Optional: detailed breakdown ----
common_7_13 = errors["7B"]["overall"] & errors["13B"]["overall"] - errors["70B"]["overall"]
common_7_70 = errors["7B"]["overall"] & errors["70B"]["overall"] - errors["13B"]["overall"]
common_13_70 = errors["13B"]["overall"] & errors["70B"]["overall"] - errors["7B"]["overall"]

print("\nPairwise-only overlaps (excluding third model):")
print("7B & 13B only:", len(common_7_13))
print("7B & 70B only:", len(common_7_70))
print("13B & 70B only:", len(common_13_70))

'''
python3 -u jaccard.py 
Same IDs across all files: True
7B: 20218 rows
13B: 20218 rows
70B: 20218 rows

Pairwise Jaccard similarity:

7B vs 13B
overall   : 0.6378 (intersection=1597, union=2504)
false_pos : 0.6969 (intersection=1023, union=1468)
false_neg : 0.5541 (intersection=574, union=1036)

7B vs 70B
overall   : 0.4429 (intersection=1385, union=3127)
false_pos : 0.5528 (intersection=791, union=1431)
false_neg : 0.3502 (intersection=594, union=1696)

13B vs 70B
overall   : 0.4530 (intersection=1402, union=3095)
false_pos : 0.5564 (intersection=789, union=1418)
false_neg : 0.3655 (intersection=613, union=1677)

Common errors across ALL 3 models:
Count: 1207

Unique errors per model:
7B: 283
13B: 251
70B: 874

Pairwise-only overlaps (excluding third model):
7B & 13B only: 390
7B & 70B only: 178
13B & 70B only: 195
'''
