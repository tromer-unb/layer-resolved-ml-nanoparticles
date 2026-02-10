import os
import random
import csv
import numpy as np
import pandas as pd
from collections import Counter

import optuna
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor

# ============================================================
# PARAMETER LOADER (MULTILINE SAFE)
# ============================================================
def load_params(fname="param_rank.txt"):
    params = {}
    key = None
    buffer = ""

    with open(fname, "r") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            if key is None:
                if "=" not in line:
                    continue
                key, val = map(str.strip, line.split("=", 1))
                buffer = val
                if buffer.count("{") == buffer.count("}") and \
                   buffer.count("[") == buffer.count("]"):
                    params[key] = eval(buffer)
                    key = None
            else:
                buffer += line
                if buffer.count("{") == buffer.count("}") and \
                   buffer.count("[") == buffer.count("]"):
                    params[key] = eval(buffer)
                    key = None

    return params

# ============================================================
# LOAD PARAMETERS
# ============================================================
PARAMS = load_params("param_rank.txt")

N_TEMPLATES = PARAMS["N_TEMPLATES"]
N_NEW_PER_TEMPLATE = PARAMS["N_NEW_PER_TEMPLATE"]
SEED = PARAMS["SEED"]

STRUCT_DIR = PARAMS["STRUCT_DIR"]
NEW_STRUCT_DIR = PARAMS["NEW_STRUCT_DIR"]
CSV_DESCR = PARAMS["CSV_DESCR"]

N_TRIALS = PARAMS.get("N_TRIALS", 300)
TEST_SIZE = PARAMS.get("TEST_SIZE", 0.2)
N_SPLITS = PARAMS.get("N_SPLITS", 5)

os.makedirs(NEW_STRUCT_DIR, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)

# ==========================================================
# 1. LEITURA DOS DESCRITORES
# ==========================================================
df = pd.read_csv(CSV_DESCR)

if "Etot" not in df.columns or "id" not in df.columns:
    raise ValueError("CSV need same lines 'id' e 'Etot'")

y = df["Etot"]
X = df.drop(columns=["Etot"])
X_num = X.select_dtypes(include=[np.number])

# ==========================================================
# 2. SPLIT TREINO / TESTE
# ==========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_num, y, test_size=TEST_SIZE, random_state=SEED
)

# ==========================================================
# 3. OPTUNA + K-FOLD
# ==========================================================
def objective(trial):

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.7, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        "objective": "reg:squarederror",
        "random_state": SEED,
        "n_jobs": -1,
        "tree_method": "hist",
    }

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    maes = []

    for tr, val in kf.split(X_train):
        model = XGBRegressor(**params)
        model.fit(X_train.iloc[tr], y_train.iloc[tr])
        pred = model.predict(X_train.iloc[val])
        maes.append(mean_absolute_error(y_train.iloc[val], pred))

    return np.mean(maes)

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=N_TRIALS)

best_params = study.best_params
best_params.update({
    "objective": "reg:squarederror",
    "random_state": SEED,
    "tree_method": "hist",
    "n_jobs": -1,
})

# ==========================================================
# 4. TREINO FINAL
# ==========================================================
model = XGBRegressor(**best_params)
model.fit(X_train, y_train)

# ==========================================================
# 5. RANKING DAS ESTRUTURAS EXISTENTES
# ==========================================================
df["Etot_pred"] = model.predict(X_num)
df = df.sort_values("Etot_pred").reset_index(drop=True)
df["rank"] = np.arange(1, len(df) + 1)

df.to_csv("ranking_xgboost.csv", index=False)
print("✔ Ranking save at ranking_xgboost.csv")

# ==========================================================
# 6. SELECIONA TOP-K TEMPLATES
# ==========================================================
top_templates = df.head(N_TEMPLATES)["id"].astype(int).tolist()

# ==========================================================
# 7. FUNÇÕES XYZ
# ==========================================================
def read_xyz(path):
    with open(path) as f:
        lines = f.readlines()
    n = int(lines[0])
    atoms = []
    for line in lines[2:2+n]:
        el, x, y, z = line.split()[:4]
        atoms.append([el, x, y, z])
    return n, atoms

def write_xyz(path, n, atoms, comment):
    with open(path, "w") as f:
        f.write(f"{n}\n{comment}\n")
        for a in atoms:
            f.write(" ".join(a) + "\n")

def permute_atoms(atoms):
    elems = [a[0] for a in atoms]
    coords = [a[1:] for a in atoms]
    random.shuffle(elems)
    return [[elems[i], *coords[i]] for i in range(len(atoms))]

# ==========================================================
# 8. GERA NOVAS ESTRUTURAS
# ==========================================================
seen = set()
new_id = 1
log = []

for tid in top_templates:
    xyz_path = os.path.join(STRUCT_DIR, f"{tid}.xyz")
    n, atoms = read_xyz(xyz_path)
    comp = Counter([a[0] for a in atoms])

    generated = 0
    while generated < N_NEW_PER_TEMPLATE:
        new_atoms = permute_atoms(atoms)
        sig = tuple(a[0] for a in new_atoms)

        if sig in seen:
            continue
        seen.add(sig)

        out_xyz = os.path.join(NEW_STRUCT_DIR, f"{new_id}.xyz")
        write_xyz(out_xyz, n, new_atoms, f"template={tid}")

        log.append({
            "new_id": new_id,
            "template_id": tid,
            "composition": dict(comp)
        })

        new_id += 1
        generated += 1

print(f"✔ {new_id-1} new structures generated")

# ==========================================================
# 9. SALVA LOG
# ==========================================================
pd.DataFrame(log).to_csv("new_structures_log.csv", index=False)

print("✔ Log save at new_structures_log.csv")
print("🚀 RUN OK")
