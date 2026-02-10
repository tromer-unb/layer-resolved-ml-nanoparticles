import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from collections import Counter, deque, defaultdict
from ase.io import read
from ase.neighborlist import NeighborList, natural_cutoffs
from ase.data import atomic_numbers

# ============================================================
# PARAMETER LOADER (MULTILINE SAFE)
# ============================================================
def load_params(fname="param_inter.txt"):
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
PARAMS = load_params("param_inter.txt")

STRUCT_DIR = PARAMS["STRUCT_DIR"]
OUT_DIR = PARAMS.get("OUT_DIR", "analysis")
GLOBAL_DIR = os.path.join(OUT_DIR, "_GLOBAL")

CUTOFF_MULT = PARAMS["CUTOFF_MULT"]
MAX_LAYER_PLOT = PARAMS.get("MAX_LAYER_PLOT", 6)

COMPUTE_LAYER_CHEM = PARAMS.get("COMPUTE_LAYER_CHEM", True)
LAYER_INTERNAL_NEIGHBORS_ONLY = PARAMS.get("LAYER_INTERNAL_NEIGHBORS_ONLY", True)
COMPUTE_LAYER_PAIR_BONDS = PARAMS.get("COMPUTE_LAYER_PAIR_BONDS", True)

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(GLOBAL_DIR, exist_ok=True)

# ------------------------------------------------------------
# Química mínima (FIXA – não parametrizada)
# ------------------------------------------------------------
PAULING_EN = {
    "Al": 1.61, "Co": 1.88, "Fe": 1.83, "Ni": 1.91, "Cu": 1.90,
}
VEC = {
    "Al": 3,
    "Fe": 8, "Co": 9, "Ni": 10, "Cu": 11,
}

# ============================================================
# HELPERS
# ============================================================
def numeric_key(fname):
    m = re.match(r"(\d+)", fname)
    return (0, int(m.group(1))) if m else (1, fname)

def _pair_label(a, b):
    return f"{a}_{b}" if a <= b else f"{b}_{a}"

def safe_nanmean(x):
    return float(np.nanmean(x)) if len(x) else np.nan

# ============================================================
# GRAFO QUÍMICO
# ============================================================
def build_graph(atoms):
    cutoffs = natural_cutoffs(atoms, mult=CUTOFF_MULT)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)

    neighbors = {i: [] for i in range(len(atoms))}
    edges = []

    for i in range(len(atoms)):
        idxs, _ = nl.get_neighbors(i)
        for j in idxs:
            j = int(j)
            neighbors[i].append(j)
            if j > i:
                edges.append((i, j))

    deg = np.array([len(neighbors[i]) for i in range(len(atoms))], dtype=int)
    return neighbors, edges, deg

# ============================================================
# CAMADAS TOPOLOGICAS
# ============================================================
def infer_ideal_degree(deg):
    vals = deg[deg > 0]
    if len(vals) == 0:
        return max(int(np.max(deg)), 1)

    ideal = Counter(vals.tolist()).most_common(1)[0][0]
    mean, std = np.mean(deg), np.std(deg)
    if ideal > mean + 2 * std:
        ideal = int(round(mean))
    return max(int(ideal), 1)

def compute_layers(neighbors, deg, ideal):
    surface = deg < ideal
    if not np.any(surface):
        surface = deg == np.min(deg)

    layer = np.full(len(deg), -1, dtype=int)
    q = deque(np.where(surface)[0].tolist())

    for i in q:
        layer[i] = 0

    while q:
        u = q.popleft()
        for v in neighbors[u]:
            if layer[v] == -1:
                layer[v] = layer[u] + 1
                q.append(v)

    if np.all(layer == -1):
        layer[:] = 0
    return layer

# ============================================================
# ANALISE POR ESTRUTURA
# ============================================================
def analyze_structure(path):
    atoms = read(path)
    symbols = atoms.get_chemical_symbols()

    neighbors, edges, deg = build_graph(atoms)
    ideal = infer_ideal_degree(deg)
    layers = compute_layers(neighbors, deg, ideal)

    elements = sorted(set(symbols))
    rows = []

    for L in sorted(set(layers)):
        if L > MAX_LAYER_PLOT:
            continue
        idx = np.where(layers == L)[0]
        if len(idx) == 0:
            continue

        syms = [symbols[i] for i in idx]
        row = {
            "layer": L,
            "deg_mean": np.mean(deg[idx]),
            "EN_mean": safe_nanmean([PAULING_EN.get(s, np.nan) for s in syms]),
            "VEC_mean": safe_nanmean([VEC.get(s, np.nan) for s in syms]),
        }

        cnt = Counter(syms)
        for el in elements:
            row[f"x_{el}"] = cnt.get(el, 0) / len(idx)

        rows.append(row)

    return pd.DataFrame(rows)

# ============================================================
# PLOTS
# ============================================================
def plot_vs_layer(df, y, outpath):
    if y not in df.columns or df[y].isna().all():
        return
    plt.figure()
    plt.plot(df["layer"], df[y], marker="o")
    plt.xlabel("Camada topológica")
    plt.ylabel(y)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

def plot_composition(df, outpath):
    cols = [c for c in df.columns if c.startswith("x_")]
    if not cols:
        return
    plt.figure()
    for c in cols:
        plt.plot(df["layer"], df[c], marker="o", label=c)
    plt.xlabel("Camada topológica")
    plt.ylabel("Fração atômica")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()

# ============================================================
# ANALISE GLOBAL
# ============================================================
def global_mean_df(dfs):
    df = pd.concat(dfs, ignore_index=True)
    num = df.select_dtypes(include=[np.number])
    rows = []

    for L, g in num.groupby(df["layer"]):
        row = {"layer": L}
        for c in g.columns:
            row[f"{c}_mean"] = g[c].mean()
            row[f"{c}_std"] = g[c].std()
        rows.append(row)

    return pd.DataFrame(rows)

def plot_global(df, col, out):
    if f"{col}_mean" not in df.columns:
        return
    plt.figure()
    plt.errorbar(df["layer"], df[f"{col}_mean"],
                 yerr=df[f"{col}_std"], marker="o", capsize=3)
    plt.xlabel("Camada topológica")
    plt.ylabel(col)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()

def plot_global_composition(df, out):
    cols = [c for c in df.columns if c.startswith("x_") and c.endswith("_mean")]
    if not cols:
        return
    plt.figure()
    for c in cols:
        plt.plot(df["layer"], df[c], marker="o", label=c.replace("_mean",""))
    plt.xlabel("Camada topológica")
    plt.ylabel("Fração atômica (média)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()

# ============================================================
# MAIN
# ============================================================
def main():
    dfs = []

    for f in sorted(os.listdir(STRUCT_DIR), key=numeric_key):
        if not f.lower().endswith((".xyz",".cif",".vasp",".poscar",".traj")):
            continue

        print("Analisando:", f)
        name = os.path.splitext(f)[0]
        outdir = os.path.join(OUT_DIR, name)
        os.makedirs(outdir, exist_ok=True)

        df = analyze_structure(os.path.join(STRUCT_DIR, f))
        df.to_csv(os.path.join(outdir, "layers.csv"), index=False)

        plot_vs_layer(df, "deg_mean", os.path.join(outdir, "deg_vs_layer.png"))
        plot_vs_layer(df, "EN_mean", os.path.join(outdir, "EN_vs_layer.png"))
        plot_vs_layer(df, "VEC_mean", os.path.join(outdir, "VEC_vs_layer.png"))
        plot_composition(df, os.path.join(outdir, "composition_vs_layer.png"))

        dfs.append(df)

    dfG = global_mean_df(dfs)
    dfG.to_csv(os.path.join(GLOBAL_DIR, "layers_mean.csv"), index=False)

    plot_global(dfG, "deg_mean", os.path.join(GLOBAL_DIR, "deg_vs_layer_GLOBAL.png"))
    plot_global(dfG, "EN_mean", os.path.join(GLOBAL_DIR, "EN_vs_layer_GLOBAL.png"))
    plot_global(dfG, "VEC_mean", os.path.join(GLOBAL_DIR, "VEC_vs_layer_GLOBAL.png"))
    plot_global_composition(dfG, os.path.join(GLOBAL_DIR, "composition_vs_layer_GLOBAL.png"))

    print("\n✅ Plots individuais e globais gerados com sucesso.")

if __name__ == "__main__":
    main()
