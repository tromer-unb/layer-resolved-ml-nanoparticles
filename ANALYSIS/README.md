# Layer-resolved physical analysis

This module performs **layer-by-layer physical and chemical analysis** of
nanoparticles using a topology-driven definition of atomic layers.

## Configuration
- **`param_inter.txt`**: parameter file controlling graph construction,
  layer definition, and analysis options

## Outputs

### Per-nanoparticle results
For each individual nanoparticle:
- **`layers.csv`**: layer-resolved numerical data
- **Coordination vs layer** plots
- **Electronegativity vs layer** plots
- **Valence electron concentration (VEC) vs layer** plots
- **Composition vs layer** plots

### Global statistics
Across the full dataset:
- **Mean ± standard deviation** of all layer-resolved properties

## Run
```bash
python inter.py
