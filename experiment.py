"""Reproduces the study by running the original scripts in code/ in their historical order.

1. code/experiment.py          - split 1 (formula groups, seeds 2026/2027): crystal graph network, LightGBM, ridge
2. code/experiment_split2.py   - split 2 (seeds 2028/2029), same models
3. code/consolidate.py            - combined tables, formula-group bootstrap, fig_results.png
Settings fixed before the runs: code/protocol.md.

Data: Matbench perovskites (18,928 structures), saved as code/data/matbench_perovskites.json.gz, e.g. with
matminer: matminer.datasets.load_dataset("matbench_perovskites"). Requires Python 3.12 and code/requirements.txt
(PyTorch on CPU; about five minutes per network initialisation).
"""

import os
import runpy
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent / "code"

for step in ["experiment.py", "experiment_split2.py", "consolidate.py"]:
    script = CODE / step
    os.chdir(script.parent)
    sys.path.insert(0, str(script.parent))
    print(f"== {step}", flush=True)
    runpy.run_path(str(script), run_name="__main__")
