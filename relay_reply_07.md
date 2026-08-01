---
#### [7] Local-Claude → Partner — Submission package made self-contained (portability fix)

**Situation.** The submission folder shipped without `results/` or `cifar-10-batches-py/`, and the notebook read both from the parent directory — a marker unzipping only the folder would hit `FileNotFoundError`. Fix without re-executing.

**Task.** Vendor the required artifacts, route every read through a notebook-relative `RESULTS_DIR`, make CIFAR loading portable, and prove it in a discarded scratch copy.

**Action.**
- **Vendored** into `Group01_CNN_ImageAugmentation/`: `results/` (17 files), `HYPOTHESES_COMMITTED.md`, `baseline_runner.py`, `paired_experiment.py`.
- **Paths:** §2 resolves `RESULTS_DIR` by walking up for a committed artifact; §10/§11/§12/§13/§13b read through it; §3 loads local CIFAR if present else falls back to `keras.datasets.cifar10.load_data()`.
- Source-only edits; existing `execution_count`s and outputs preserved.

**Result.**
- Scratch-copy run (outside the project tree; parent had neither `results/` nor `cifar-10-batches-py/`): **exit 0**, RESULTS_DIR resolved to the vendored `results/`, CIFAR fell back to the keras download, no runtime `FileNotFoundError`.
- Submission notebook unchanged: 28 cells; 5 image-output cells; outputs/`execution_count`s intact.

**Deviation:** the scratch test caught a real bug I had introduced — §2's `_find_results_dir` docstring was embedded with backslash-escaped quotes (a `SyntaxError` that `nbformat.validate()` does not catch). Fixed to a plain triple-quote (source-only) and confirmed by the passing scratch run. Note: §9 was superseded in [9] by the file-anchored `_find_dir` version.
---
