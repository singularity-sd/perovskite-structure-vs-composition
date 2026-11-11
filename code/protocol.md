# Protocol: T3v3-2026-0098

## Question
Test a graph neural network for a ceramic-family property task: predict DFT formation energy of perovskite structures, then compare structural message passing with composition-only regressors.

## Data
Matbench perovskites contains 18,928 crystal structures and the target e_form in eV per unit cell. The material family is relevant to ceramics. The target is DFT-calculated formation energy; no experimental measurements are used. The source data is stored locally in data/matbench_perovskites.json.gz. There are 9,646 unique reduced formulas.

## Split
Split 1: GroupShuffleSplit by reduced formula, outer test fraction 20%, seed 2026; then validation fraction 12.5% of the remaining records, seed 2027.
Split 2: Same fractions, seeds 2028 and 2029.
Both splits put all records for each reduced formula in only one partition. The validation set selects the GNN checkpoint by MAE; the test set is not used for early stopping or model selection.

## Models
- CrystalGNN: learned atomic-number embeddings; up to 16 nearest periodic neighbors per atom within an 8 angstrom radius; 16 radial-basis distance features; three mean-aggregating message-passing updates; mean graph pooling; standardized target during fitting.
- Composition ridge: elemental fraction vector (118 dimensions), standard scaling, alpha 10.
- Composition LightGBM: 700 estimators, learning rate 0.04, 31 leaves, min child samples 20, subsample 0.85, column sample 0.8.
- GNN: AdamW, learning rate 0.001, weight decay 0.0001, batch size 64, at most 60 epochs, validation MAE early stopping with patience 8, three fixed seeds (17, 29, 43) per split.

## Outcomes and limits
Primary metric is test MAE in eV per unit cell. Also report RMSE, R-squared, fit time, seed variation, and a paired bootstrap of MAE difference by held-out reduced formula. This is one computational dataset with two formula-group splits; it does not establish experimental accuracy or performance on all ceramics and composites.
