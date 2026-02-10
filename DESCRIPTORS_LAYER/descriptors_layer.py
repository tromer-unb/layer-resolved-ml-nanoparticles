import os
import re
import numpy as np
import pandas as pd
from collections import Counter, deque, defaultdict
from itertools import product, combinations_with_replacement

from ase.io import read
from ase.neighborlist import NeighborList, natural_cutoffs

# ============================================================
# PARAMETER LOADER (SUPPORTS MULTILINE DICTS / LISTS)
# ============================================================
def load_params(fname="param.txt"):
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
PARAMS = load_params("param.txt")

STRUCT_DIR = PARAMS["STRUCT_DIR"]
OUT_CSV = PARAMS["OUT_CSV"]

CUTOFF_MULT = PARAMS["CUTOFF_MULT"]
BULK_LAYERS = PARAMS["BULK_LAYERS"]

LAYER_WEIGHTS = PARAMS["LAYER_WEIGHTS"]

COMPUTE_CHEMISTRY = PARAMS["COMPUTE_CHEMISTRY"]
REGION_INTERNAL_NEIGHBORS_ONLY = PARAMS["REGION_INTERNAL_NEIGHBORS_ONLY"]

ELEMENTS = PARAMS["ELEMENTS"]

ORDERED_PAIRS = [(a, b) for a, b in product(ELEMENTS, ELEMENTS)]
UNORDERED_PAIRS = sorted(
    {tuple(sorted(p)) for p in combinations_with_replacement(ELEMENTS, 2)}
)

# ============================================================
# HELPERS
# ============================================================
def numeric_key(fname):
    m = re.match(r"(\d+)", fname)
    return (0, int(m.group(1))) if m else (1, fname)

# ============================================================
# GRAPH CONSTRUCTION
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

    deg = np.array([len(neighbors[i]) for i in range(len(atoms))])
    return neighbors, edges, deg

# ============================================================
# TOPOLOGICAL DISTANCE
# ============================================================
def infer_ideal_degree(deg):
    vals = deg[deg > 0]
    return Counter(vals).most_common(1)[0][0] if len(vals) else 0

def compute_layer_distance(neighbors, deg, ideal):
    surface = deg < ideal
    dist = np.full(len(deg), -1, dtype=int)
    q = deque(np.where(surface)[0])
    dist[surface] = 0

    while q:
        u = q.popleft()
        for v in neighbors[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)

    if np.all(dist == -1):
        dist[:] = 999999
    return dist

def layer_weights(dist):
    w = np.zeros_like(dist, dtype=float)
    for L, val in LAYER_WEIGHTS.items():
        w[dist == L] = val
    return w

# ============================================================
# WEIGHTED STATISTICS
# ============================================================
def stats_from_mask_weighted(deg, mask, weights, edges, atoms):
    idx = np.where(mask & (weights > 0))[0]
    if len(idx) == 0:
        return dict(n_atoms=0, deg_mean=0.0, deg_std=0.0,
                    bond_mean=0.0, bond_std=0.0)

    w = weights[idx]
    w /= w.sum()

    deg_mean = float(np.sum(w * deg[idx]))
    deg_std = float(np.sqrt(np.sum(w * (deg[idx] - deg_mean) ** 2)))

    bonds, bw = [], []
    for i, j in edges:
        if mask[i] and mask[j]:
            wi, wj = weights[i], weights[j]
            if wi > 0 and wj > 0:
                bonds.append(atoms.get_distance(i, j, mic=True))
                bw.append(0.5 * (wi + wj))

    if bonds:
        bonds = np.array(bonds)
        bw = np.array(bw)
        bw /= bw.sum()
        bond_mean = float(np.sum(bw * bonds))
        bond_std = float(np.sqrt(np.sum(bw * (bonds - bond_mean) ** 2)))
    else:
        bond_mean, bond_std = 0.0, 0.0

    return dict(
        n_atoms=int(len(idx)),
        deg_mean=deg_mean,
        deg_std=deg_std,
        bond_mean=bond_mean,
        bond_std=bond_std,
    )

# ============================================================
# WEIGHTED CHEMISTRY + SRO
# ============================================================
def neighbor_fractions_and_sro_weighted(symbols, neighbors, mask, weights, prefix):
    out = {}

    for el in ELEMENTS:
        out[f"{prefix}_frac_{el}"] = 0.0
    for A, B in ORDERED_PAIRS:
        out[f"{prefix}_P_{A}_nn_{B}"] = 0.0
        out[f"{prefix}_alpha_{A}_{B}"] = 0.0
    out[f"{prefix}_chem_entropy"] = 0.0

    idx = np.where(mask & (weights > 0))[0]
    if len(idx) == 0:
        return out

    w = weights[idx]
    w /= w.sum()

    comp = defaultdict(float)
    for i, wi in zip(idx, w):
        comp[symbols[i]] += wi

    for el in ELEMENTS:
        out[f"{prefix}_frac_{el}"] = comp.get(el, 0.0)

    den = defaultdict(float)
    num = defaultdict(float)

    for i, wi in zip(idx, w):
        A = symbols[i]
        neigh = neighbors[i]
        if REGION_INTERNAL_NEIGHBORS_ONLY:
            neigh = [j for j in neigh if mask[j]]
        den[A] += wi * len(neigh)
        for j in neigh:
            num[(A, symbols[j])] += wi

    for A, B in ORDERED_PAIRS:
        if den[A] > 0:
            P = num[(A, B)] / den[A]
            out[f"{prefix}_P_{A}_nn_{B}"] = P
            cB = out[f"{prefix}_frac_{B}"]
            out[f"{prefix}_alpha_{A}_{B}"] = 1 - P / cB if cB > 0 else 0.0

    ent = 0.0
    for el in ELEMENTS:
        p = out[f"{prefix}_frac_{el}"]
        if p > 0:
            ent -= p * np.log(p)
    out[f"{prefix}_chem_entropy"] = ent

    return out

# ============================================================
# WEIGHTED BOND STATISTICS
# ============================================================
def bond_stats_by_pair_weighted(symbols, edges, atoms, mask, weights, prefix):
    out = {}
    buckets = {f"{a}_{b}": [] for a, b in UNORDERED_PAIRS}
    bw = {f"{a}_{b}": [] for a, b in UNORDERED_PAIRS}

    for i, j in edges:
        if mask[i] and mask[j]:
            wi, wj = weights[i], weights[j]
            if wi > 0 and wj > 0:
                a, b = sorted((symbols[i], symbols[j]))
                buckets[f"{a}_{b}"].append(atoms.get_distance(i, j, mic=True))
                bw[f"{a}_{b}"].append(0.5 * (wi + wj))

    for pair in buckets:
        vals = buckets[pair]
        w = bw[pair]
        if vals:
            vals = np.array(vals)
            w = np.array(w)
            w /= w.sum()
            mean = float(np.sum(w * vals))
            std = float(np.sqrt(np.sum(w * (vals - mean) ** 2)))
            out[f"{prefix}_bond_n_{pair}"] = len(vals)
            out[f"{prefix}_bond_mean_{pair}"] = mean
            out[f"{prefix}_bond_std_{pair}"] = std
        else:
            out[f"{prefix}_bond_n_{pair}"] = 0
            out[f"{prefix}_bond_mean_{pair}"] = 0.0
            out[f"{prefix}_bond_std_{pair}"] = 0.0

    return out

# ============================================================
# PER-STRUCTURE ROUTINE
# ============================================================
def process_structure(path):
    atoms = read(path)
    symbols = np.array([a.symbol for a in atoms])

    neighbors, edges, deg = build_graph(atoms)
    ideal = infer_ideal_degree(deg)
    dist = compute_layer_distance(neighbors, deg, ideal)
    weights = layer_weights(dist)

    surface = dist < BULK_LAYERS
    bulk = ~surface
    total = np.ones(len(atoms), bool)

    out = dict(
        system=os.path.basename(path),
        ideal_degree=int(ideal),
        bulk_layers=BULK_LAYERS,
        n_atoms_total=len(atoms),
        n_atoms_bulk=int(np.sum(bulk)),
        n_atoms_surface=int(np.sum(surface)),
        frac_bulk_atoms=float(np.mean(bulk)),
        frac_surface_atoms=float(np.mean(surface)),
    )

    for name, mask in [("total", total), ("bulk", bulk), ("surface", surface)]:
        out.update(stats_from_mask_weighted(deg, mask, weights, edges, atoms))
        if COMPUTE_CHEMISTRY:
            out.update(neighbor_fractions_and_sro_weighted(
                symbols, neighbors, mask, weights, name
            ))
            out.update(bond_stats_by_pair_weighted(
                symbols, edges, atoms, mask, weights, name
            ))

    return out

# ============================================================
# DRIVER
# ============================================================
def main():
    files = sorted(
        [f for f in os.listdir(STRUCT_DIR)
         if f.lower().endswith((".xyz", ".cif", ".vasp", ".traj"))],
        key=numeric_key
    )

    rows = []
    for f in files:
        print("Processing:", f)
        rows.append(process_structure(os.path.join(STRUCT_DIR, f)))

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)

    print("\n✅ Multilayer-embedded descriptor generated successfully")
    print("Total features:", df.shape[1])
    print("Output file:", OUT_CSV)

if __name__ == "__main__":
    main()
