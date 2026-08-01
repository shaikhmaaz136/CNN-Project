---
#### [9] Local-Claude → Partner — Statistics computed in-notebook; paths hardened; Colab-clean

**Situation.** Audit findings: make §11/§12 *compute* their statistics rather than read the JSON, and harden the path lookup. Done after [8].

**Task.** Replace §11/§12 with the computing versions, adopt the file-anchored path block, execute only those cells, verify the numbers match, and assess Colab.

**Action.**
- **Finding 1 (compute):** replaced §11 and §12 with your versions that compute from `results/paired_runs.csv` (paired *t*-test, Wilcoxon, 95% CI, Cohen's *d*, with `NOISE_FLOOR` read from `baseline_summary.json`). Fixed §13b's `RESULTS_DIR / '…'` → `os.path.join(RESULTS_DIR, '…')` (RESULTS_DIR is a string now). Executed **only** §11, §12, §13b; every other cell kept its output.
- **Finding 2 (paths):** replaced §2's path block with your `_find_dir(name, anchor_file)` version (anchors on `results/comparison_table.csv` and `cifar-10-batches-py/batches.meta`, walks up to 5 levels with a root-stop). Updated §3 to load from `DATA_DIR`, or fall back to the keras download when it is `None`. *(My [7] version already anchored on the file `results/comparison_table.csv` — not the broken cifar-directory anchor — but I adopted your audited version.)*

**Result:**
- **Recomputed statistics match the committed values exactly.**
  - test accuracy: mean d **−0.0278**, SD 0.0246, 95% CI **[−0.0454, −0.0102]**, paired *t* **−3.5728** p **0.005998**, Wilcoxon 6.0 p **0.027344**, Cohen's *d* **−1.1298**.
  - gap: mean d **−0.0201**, SD 0.0153, 95% CI **[−0.0310, −0.0092]**, *t* **−4.1639** p **0.001217**, Wilcoxon 2.0 p **0.002930**, Cohen's *d* **−1.3167**.
  - manipulation check: baseline **0.5120**, augmented **0.4744**, diff **−0.0376**. No CSV/JSON disagreement.
- **One value moved, expectedly:** `mean|d| / noise floor` is now **0.795** (test) / **0.574** (gap), vs the earlier 0.794 / 0.573. This is **not** a data disagreement — §11 now reads the *exact* noise floor `0.0376667` from `baseline_summary.json`, whereas `paired_experiment.py` used the rounded constant `0.0377`; the printed label rounds it back to `(0.0377)`. The more precise value is what your new code intends. I did not "stop", since the CSV and JSON agree on everything.
- **NumPy 2.x cleanup:** your §11 printed per-seed d as `[np.float64(-0.033), …]`. I added `float()` in the list comprehension so it prints `[-0.033, …]`; values identical.
- **Colab compatibility — no blockers.** Every import (`tensorflow`/`keras`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `scipy.stats`, `sklearn.metrics`, `IPython.display`, stdlib) is preinstalled on stock Colab; `%matplotlib inline` is native; the §3 keras fallback (when `DATA_DIR is None`, which it will be on Colab) downloads CIFAR fine. Only requirement: run with `results/` alongside the notebook (the packaged layout).
- **Scratch portability re-run** (isolated copy; parent had neither `results/` nor `cifar-10-batches-py/`; `nbconvert --execute`): **exit 0, ~204s**; RESULTS_DIR resolved to the vendored `results/`; CIFAR used the keras download; §11/§12 ran correctly in the Jupyter kernel with clean, matching output.
- **Final counts:** 28 cells, 5 image-output cells, **0 `[DRAFT`**, **0 `[PENDING`**; §9 training output unchanged; all code cells compile.
---
