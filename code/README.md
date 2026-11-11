# Crystal GNN for perovskite formation energy

Test of forecast **T3v3-2026-0098 — graph neural networks in Ceramics and Composites** (position 6; 1,160 supporting records).

## The question

Can a crystal-graph neural network predict formation energy for perovskite structures with reduced chemical formulas absent from training? We compare it with two composition-only baselines.

## What we did

We used the Matbench perovskites dataset: 18,928 crystal structures and DFT formation energies in eV per unit cell. Perovskites are a materials family with clear ceramics relevance, but this test does not cover composites or every kind of ceramic. The target values are calculated, not experimental.

The split keeps each reduced formula entirely in train, validation, or test. This asks whether the models transfer to unseen compositions, which is harder than randomly separating structures with the same composition on both sides. We ran two different formula-group splits. The graph model uses atomic-number embeddings, periodic neighbor distances, and message passing. Ridge and LightGBM use elemental fractions only. Each graph-model split uses three random initializations.

## Results

Across the two splits, the CrystalGNN mean test MAE was 0.059 and 0.055 eV per unit cell. The composition-only LightGBM baselines scored 0.403 and 0.419 eV; Ridge scored 0.437 and 0.451 eV. The CrystalGNN reached R-squared 0.987 and 0.988. Formula-group bootstrap intervals for the three-seed GNN average versus LightGBM were [-0.368, -0.341] and [-0.388, -0.359] eV per cell. The two splits show the same strong pattern on this benchmark.

## Files

- main.pdf, main.tex: paper and LaTeX source.
- experiment.py, experiment_split2.py: model and two split runs.
- consolidate.py: combined metrics and formula overlap audit.
- protocol.md: dataset split, model architecture, and evaluation choices.
- results_full.json, metrics_split_comparison.csv, metrics_summary_full.csv, test_predictions.csv, split2_test_predictions.csv, bootstrap_formulas.csv, split2_bootstrap_formulas.csv, audit_formula.csv: outputs.
- graphs_cache.npz: precomputed crystal graphs.
- fig_results.png: paper figure.
- data/: Matbench source data and attribution.

## Reproduce

Use Python 3.13 with PyTorch, Matminer 0.10.1, pymatgen, LightGBM, NumPy, pandas, scikit-learn, and Matplotlib. The source dataset is retained locally.

py -3.13 experiment.py
py -3.13 experiment_split2.py
py -3.13 consolidate.py

The Matbench project and dataset: https://matbench.materialsproject.org and https://github.com/materialsproject/matbench. See Dunn et al., npj Computational Materials 6, 138 (2020), DOI 10.1038/s41524-020-00406-3. This study is private and has not been added to the project website.
