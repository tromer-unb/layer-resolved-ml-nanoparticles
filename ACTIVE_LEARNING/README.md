
---

## README.md (ranking)

```markdown
# Ranking and active learning

This module performs:
1. XGBoost regression optimized with Optuna
2. Ranking of nanoparticle stability
3. Active learning candidate generation

## Input
- `file.csv`: descriptors + Etot (per atom) 

## Output
- ranking_xgboost.csv
- structures_new/
- novas_estruturas_log.csv

## Run
```bash
python rank.py

