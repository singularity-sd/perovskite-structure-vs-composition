# Crystal structure versus composition for the formation energy of unseen perovskite formulas

From a v3 forecast: for unseen perovskite formulas, a crystal graph network cuts the error about seven times.

**Paper:** [paper.pdf](paper.pdf) · **Page:** [singularitysd.com/research/perovskites.html](https://singularitysd.com/research/perovskites.html) · Preprint, not yet peer reviewed.

## Abstract

Composition-only models are a common first screen for new ceramic materials. We measure how much a small crystal graph neural network gains over them when the formulas to be predicted are entirely absent from training. On the Matbench perovskites benchmark (18,928 structures, DFT formation energy), with all structures of a reduced formula kept in one partition, the graph network reached a mean test error of 0.059 and 0.055 eV per unit cell in two independent splits, against 0.403 and 0.419 for LightGBM on elemental fractions and 0.437 and 0.451 for ridge regression. The graph network explained 98.7–98.8% of the variance, LightGBM 47–49%. Paired bootstrap intervals over 1,930 held-out formula groups put the reduction against LightGBM at 0.341 to 0.368 and 0.359 to 0.388 eV. A deliberately small network, trained in about five minutes per seed on a CPU, is enough to obtain this gap.

## How it was made

This study started from the Singularity forecast **T3v3-2026-0098**: a method predicted to enter a new field. It was then tested once, with the plan fixed before looking at the results.

## What is in this repository

| file | what it is |
|---|---|
| `code/` | full code, protocol and intermediate results |
| `experiment.py` | reproduces the study end to end |
| `fig_results.png` | figure |
| `main.tex` | LaTeX source of the paper |
| `paper.pdf` | the paper |
| `results.json` | every number reported in the paper |

## Reproduce

```bash
pip install -r code/requirements.txt
python experiment.py
```

It rewrites the results and figures.

## Cite

```bibtex
@misc{singularity_perovskite_structure_vs_composition_2026,
  title  = {Crystal structure versus composition for the formation energy of unseen perovskite formulas},
  author = {Singularity},
  year   = {2026},
  url    = {https://github.com/singularity-sd/perovskite-structure-vs-composition}
}
```

## License

Code: MIT (`LICENSE`). Paper, text and figures: CC BY 4.0 (`LICENSE-paper.md`). Third-party data keep their own licence where stated.

---

Part of [Singularity](https://singularitysd.com): forecasting where science goes next, then testing it.
