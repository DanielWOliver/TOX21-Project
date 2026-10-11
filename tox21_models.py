# Predicting Tox21 assay activity from molecular structure.
#
# To run: open a terminal in this folder, activate the tox21 environment
# (conda activate tox21), then type: python tox21_pipeline.py

import sqlite3

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from tdc.single_pred import Tox
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score


# ---- 1. Load the NR-AR assay and look at the class balance ----

data = Tox(name="Tox21", label_name="NR-AR")
df = data.get_data()
print(df.shape)
print(df["Y"].value_counts())


# ---- 2. Turn molecules into fingerprints ----

gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def smiles_to_fp(smiles):
    # SMILES text -> molecule -> list of 2,048 zeros and ones
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return gen.GetFingerprintAsNumPy(mol)


def featurise(frame):
    # Turn a table of SMILES and labels into X (fingerprints) and y (labels)
    fps = frame["Drug"].apply(smiles_to_fp)
    mask = fps.notnull()
    X = np.stack(fps[mask].values)
    y = frame.loc[mask, "Y"].values
    return X, y


# ---- 3. Scaffold split: train, validation, test ----

split = data.get_split(method="scaffold", seed=1)
train_df = split["train"].copy()
valid_df = split["valid"].copy()
test_df = split["test"].copy()
print(len(train_df), len(valid_df), len(test_df))


# ---- 4. Store the splits in a SQLite database ----

train_df["split"] = "train"
valid_df["split"] = "valid"
test_df["split"] = "test"

all_df = pd.concat([train_df, valid_df, test_df])

conn = sqlite3.connect("tox21.db")
all_df.to_sql("nr_ar_results", conn, if_exists="replace", index=False)

# Class balance in each split, using GROUP BY
query = """
SELECT split, COUNT(*) AS n_molecules, SUM(Y) AS n_active, ROUND(AVG(Y), 3) AS fraction_active
FROM nr_ar_results
GROUP BY split
"""
print(pd.read_sql(query, conn))


# ---- 5. Read the splits back out of the database and make X and y ----

def load_split(name):
    q = "SELECT Drug, Y FROM nr_ar_results WHERE split = ?"
    return pd.read_sql(q, conn, params=(name,))


train_df = load_split("train")
valid_df = load_split("valid")
test_df = load_split("test")

X_train, y_train = featurise(train_df)
X_valid, y_valid = featurise(valid_df)
X_test, y_test = featurise(test_df)
print(X_train.shape, X_valid.shape, X_test.shape)


# ---- 6. Train three models and score them on the validation set ----

# Logistic regression (the baseline)
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)
probs = model.predict_proba(X_valid)[:, 1]

# Random forest
rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=1, n_jobs=-1)
rf.fit(X_train, y_train)
rf_probs = rf.predict_proba(X_valid)[:, 1]

# Gradient boosting
gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4,
                                    class_weight="balanced", random_state=1)
gb.fit(X_train, y_train)
gb_probs = gb.predict_proba(X_valid)[:, 1]

print("Validation scores")
print("Logistic regression  AUROC:", roc_auc_score(y_valid, probs),
      " AP:", average_precision_score(y_valid, probs))
print("Random forest        AUROC:", roc_auc_score(y_valid, rf_probs),
      " AP:", average_precision_score(y_valid, rf_probs))
print("Gradient boosting    AUROC:", roc_auc_score(y_valid, gb_probs),
      " AP:", average_precision_score(y_valid, gb_probs))
print("Random guessing (AP):", y_valid.mean())


# ---- 7. Final test: the random forest, chosen before using the test set ----

rf_test_probs = rf.predict_proba(X_test)[:, 1]

print("Test AUROC:", roc_auc_score(y_test, rf_test_probs))
print("Test average precision:", average_precision_score(y_test, rf_test_probs))
print("Random guessing (AP):", y_test.mean())
print("Actives in test set:", int(y_test.sum()))


# ---- 8. How uncertain are the test scores? Bootstrap ----

rng = np.random.default_rng(1)
n = len(y_test)
boot_auroc = []
boot_ap = []

for _ in range(1000):
    idx = rng.integers(0, n, size=n)
    if y_test[idx].sum() == 0:
        continue
    boot_auroc.append(roc_auc_score(y_test[idx], rf_test_probs[idx]))
    boot_ap.append(average_precision_score(y_test[idx], rf_test_probs[idx]))

print("AUROC 95% interval:", np.percentile(boot_auroc, [2.5, 97.5]))
print("AP 95% interval:", np.percentile(boot_ap, [2.5, 97.5]))


# ---- 9. All 12 assays: store as two tables ----

assays = ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
          "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]

frames = []
for assay in assays:
    d = Tox(name="Tox21", label_name=assay).get_data()
    d["assay"] = assay
    frames.append(d)

labels_df = pd.concat(frames)

# One table of molecules, one table of labels, linked by Drug_ID
molecules = labels_df[["Drug_ID", "Drug"]].drop_duplicates()
labels = labels_df[["Drug_ID", "assay", "Y"]]

molecules.to_sql("molecules", conn, if_exists="replace", index=False)
labels.to_sql("labels", conn, if_exists="replace", index=False)
print(len(molecules), len(labels))


# ---- 10. Use JOINs to combine the two tables ----

def load_assay(assay):
    q = """
    SELECT m.Drug, l.Y
    FROM molecules AS m
    JOIN labels AS l ON m.Drug_ID = l.Drug_ID
    WHERE l.assay = ?
    """
    return pd.read_sql(q, conn, params=(assay,))


# Check: the JOIN should give back the original NR-AR data (7265 rows, 309 active)
ar = load_assay("NR-AR")
print(ar.shape, ar["Y"].sum())

# The molecules that were active in the most assays
query = """
SELECT m.Drug_ID, m.Drug, COUNT(*) AS n_assays, SUM(l.Y) AS n_active
FROM molecules AS m
JOIN labels AS l ON m.Drug_ID = l.Drug_ID
GROUP BY m.Drug_ID, m.Drug
ORDER BY n_active DESC
LIMIT 10
"""
print(pd.read_sql(query, conn))


# ---- 11. Random forest on all 12 assays ----

rows = []

for assay in assays:
    d = Tox(name="Tox21", label_name=assay)
    sp = d.get_split(method="scaffold", seed=1)
    Xtr, ytr = featurise(sp["train"])
    Xte, yte = featurise(sp["test"])

    m = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=1, n_jobs=-1)
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]

    rows.append({
        "assay": assay,
        "n_test": len(yte),
        "n_active_test": int(yte.sum()),
        "auroc": roc_auc_score(yte, p),
        "avg_precision": average_precision_score(yte, p),
        "random_floor": yte.mean(),
    })
    print("finished", assay)

results_12 = pd.DataFrame(rows)
results_12.to_sql("assay_results", conn, if_exists="replace", index=False)

# Average precision divided by the random-guessing floor, calculated in SQL
query = """
SELECT assay,
       ROUND(auroc, 3) AS auroc,
       ROUND(avg_precision, 3) AS avg_precision,
       ROUND(avg_precision / random_floor, 1) AS ratio_to_floor
FROM assay_results
ORDER BY assay
"""
print(pd.read_sql(query, conn))

conn.close()
