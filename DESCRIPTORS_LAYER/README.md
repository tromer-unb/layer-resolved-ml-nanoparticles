# Layer-resolved descriptor

This module constructs a **topology-driven, multilayer descriptor** for
chemically complex nanoparticles.

## Main script
- **`descriptors_layer.py`**: core implementation of the layer-resolved descriptor

## Input
- **Atomic structures** located in `structures/`
- **Descriptor parameters** defined in `param.txt`

## Output
- **`descriptors.csv`**: numerical descriptor matrix (one row per nanoparticle)

## Key concepts
- **Chemical neighbor graph** constructed using ASE natural cutoffs
- **Surface atoms** identified via coordination-number deficit
- **Topological distance** from the surface computed using breadth-first search (BFS)
- **Layer-dependent weighting** applied without increasing feature dimensionality

## Run
```bash
python descriptors_layer.py
