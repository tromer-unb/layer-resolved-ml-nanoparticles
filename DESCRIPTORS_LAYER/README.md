
---

## README.md (descriptor)

```markdown
# Layer-resolved descriptor

This module constructs a **topology-driven, multilayer descriptor** for nanoparticles.

## Main script
- `descriptors_layer.py`

## Input
- Atomic structures in `structures/`
- Parameters defined in `param.txt`

## Output
- `descriptors.csv`

## Key concepts
- Neighbor graph built using ASE natural cutoffs
- Surface atoms identified by coordination deficit
- Topological distance computed via BFS
- Layer-dependent weighting without increasing feature dimensionality

## Run
```bash
python descriptors_layer.py
