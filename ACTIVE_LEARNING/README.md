# Ranking and active learning

This module performs:
1. XGBoost regression optimized with Optuna
2. Ranking of nanoparticle stability
3. Active learning candidate generation

## Input
- `file.csv`: numerical dataset containing the layer-resolved descriptors and
  the DFT target property (`Etot`, energy per atom)

### Optional preprocessing
If the descriptor file and the target energies are stored separately, the
auxiliary script `sidebyside.py` can be used to generate `file.csv` by
concatenating the descriptor matrix with the corresponding target values
(structure-by-structure, in the same order).

## Output
- `ranking_xgboost.csv`: predicted energies and stability ranking
- `structures_new/`: newly generated nanoparticle candidates (active learning)
- `new_structures_log.csv`: log linking new structures to their parent templates

## Run
```bash
python rank.py
