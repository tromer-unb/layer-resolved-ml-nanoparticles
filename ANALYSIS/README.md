
---

## README.md (analysis)

```markdown
# Layer-resolved physical analysis

This module performs **layer-by-layer physical analysis** of nanoparticles.
param_inter.txt contain the parameters used in the simulation.

## Outputs
For each nanoparticle:
- layers.csv
- coordination vs layer
- electronegativity vs layer
- VEC vs layer
- composition vs layer

Global averages:
- mean ± std for all properties per layer

## Run
```bash
python inter.py

