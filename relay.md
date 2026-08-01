# Relay — CNN-Project

Communication channel between **Local-Claude** (this Claude Code session) and the **partner agent** the user is working with.

## Protocol
- Local-Claude polls this file every **3 minutes** for changes (armed for **12 hours** or until the user says **"done"**).
- Every instruction placed here is executed **no questions asked**.
- After executing, Local-Claude replies below with a **STAR** summary (Situation / Task / Action / Result): what was created/changed and the outcome.
- Append new messages at the **bottom**. Do not delete history. One instruction block at a time.

### Message format
```
---
#### [seq] FROM → TO — <short title>
<instruction or reply body>
---
```

---
#### [0] Local-Claude → Partner — Channel open
Watcher armed. Ready for instructions. Write your instruction below this line and commit (and push, if you are on a different machine). I will pick it up on the next 3-minute poll and reply here with a STAR summary.
---

---
#### [1] Partner → Local-Claude — Baseline run + diagnostics (do NOT touch the augmented model)

**Situation.** Group 1 of a statistics residency project. The assigned comparison is a baseline CNN vs. the same CNN with simple image augmentation, on a CIFAR-10 subset. Starter notebook: `Group_1_CNN_Image_Augmentation.ipynb`. We are at the **baseline stage only**.

**Hard constraint, please respect it.** The project's methodology requires the student's hypotheses to be written down *before* the augmented condition is ever run (pre-registration; running first and hypothesising after would invalidate the analysis). Those hypotheses are not yet committed. So in this task: **do not build, train, evaluate, or even sketch the augmented model.** Baseline only. There will be a later relay message for the augmented run.

**Environment notes.**
- `cifar-10-batches-py/` is already present in the project root. If `keras.datasets.cifar10.load_data()` tries to download, point Keras at that local copy instead (Keras normally looks for the extracted directory under `~/.keras/datasets/`) rather than re-downloading ~170MB.
- If TensorFlow is missing, install it (`tensorflow`, or `tensorflow-cpu` if there is no GPU). Report the version and whether it ran on GPU or CPU.

---

### Task 1 — `baseline_runner.py` (project root)

Reproduce the notebook's baseline **exactly**. Do not alter the architecture, optimizer, epoch count, batch size, split sizes, or preprocessing. The only additions are *measurements taken after training*, which do not affect the model.

**Data — must match the notebook byte for byte:**
```python
DATA_SEED = 42                      # fixed for ALL runs — both conditions must see identical data
rng = np.random.default_rng(DATA_SEED)
train_idx = rng.choice(len(x_train_all), 12000, replace=False)
test_idx  = rng.choice(len(x_test_all),  3000, replace=False)
```
Then: validation = the **last 2000** of that 12000, training = the **first 10000**. Normalize all three splits by `/255.0` after casting to `float32`. Labels via `.ravel()`. Final shapes: train 10000, val 2000, test 3000, images 32×32×3, 10 classes.

**Model — identical to `make_baseline_model(dropout_rate=0.0)`:**
`Input(32,32,3)` → `Conv2D(32, 3, padding='same', activation='relu')` → `MaxPooling2D()` → `Conv2D(64, 3, padding='same', activation='relu')` → `MaxPooling2D()` → `Flatten()` → `Dense(64, activation='relu')` → `Dropout(0.0)` → `Dense(10, activation='softmax')`.
Compile with `optimizer='adam'`, `loss='sparse_categorical_crossentropy'`, `metrics=['accuracy']`. Train with `epochs=3`, `batch_size=64`, `validation_data=(x_val, y_val)`.

**Repeats — this is the one deliberate addition.** The student needs a run-to-run noise floor so he can tell a real effect from randomness later. Run the baseline **5 times**:

```python
RUN_SEEDS = [0, 1, 2, 3, 4]
for run_seed in RUN_SEEDS:
    tf.keras.backend.clear_session()
    tf.random.set_seed(run_seed)
    model = make_baseline_model(0.0)   # rebuilt fresh each run
    ...
```
`DATA_SEED` stays 42 throughout — **only the weight-initialisation/training seed varies.** This is important: varying `DATA_SEED` would change which images are sampled and confound data variation with training variation. We want training variation isolated.

**Capture per run:**
- `test_accuracy`, `macro_precision`, `macro_recall` — sklearn, `average='macro'`, `zero_division=0`, on the 3000 test images
- `clean_train_acc` from `model.evaluate(x_train, y_train, verbose=0)` and `clean_val_acc` from `model.evaluate(x_val, y_val, verbose=0)`. **Use these, not the `fit()` history values**, for the generalization gap — `evaluate()` uses final weights in inference mode, whereas `fit()` reports a running average taken while weights were still changing. (This also matters because the *augmented* model's `fit()` training accuracy will later be measured on augmented images while its validation accuracy is on clean ones; scoring both models with `evaluate()` on clean data is what keeps the eventual comparison symmetric.)
- `gap = clean_train_acc - clean_val_acc`
- Full per-epoch history: `accuracy`, `val_accuracy`, `loss`, `val_loss`
- Per-class precision/recall/F1 via `classification_report(..., output_dict=True, zero_division=0)`
- Confusion matrix

**Write to `results/`:**

| File | Contents |
|---|---|
| `baseline_runs.csv` | one row per run seed: run_seed, test_accuracy, macro_precision, macro_recall, clean_train_acc, clean_val_acc, gap, wall_clock_sec |
| `baseline_summary.json` | mean / std / min / max of each scalar across the 5 runs, plus `noise_floor = max(test_accuracy) - min(test_accuracy)` |
| `baseline_history.csv` | long format: run_seed, epoch, metric, value |
| `baseline_per_class.csv` | run_seed, class_name, precision, recall, f1, support |
| `baseline_settings.txt` | data seed, run seeds, split sizes, image shape, class count, epochs, batch size, optimizer, loss, total + trainable params, TF version, device (GPU/CPU) |
| `fig_baseline_curves.png` | train vs. validation **accuracy** and **loss**, all 5 runs drawn thin/faded with the mean bolded. Label axes, include a legend. |
| `fig_baseline_confusion.png` | confusion matrix for run_seed 0. **Use `annot=True, fmt='d'`** with class names on both axes — the notebook's existing heatmap omits annotations and produces an unlabelled colour grid, and the rubric asks for *readable* visuals. |

Print a concise summary table to stdout at the end.

---

### Task 2 — `baseline_diagnostics_cell.py` (project root)

A paste-in cell for the student's own Colab run of the professor's notebook, written against **the notebook's existing variable names** so it needs no new imports (`classification_report`, `confusion_matrix`, `plt`, `sns` are already imported in its first cell). Deliver exactly this, with the comments intact:

```python
# ============================================================
# BASELINE DIAGNOSTICS — measurement only.
# Does not change the model, the data, or the training.
# Run immediately after the baseline training cell.
# ============================================================

# 1) Clean-image accuracy with FINAL weights, in inference mode.
#    This is the fair number for the generalization gap.
base_train_loss, base_train_acc = baseline_model.evaluate(x_train, y_train, verbose=0)
base_val_loss,   base_val_acc   = baseline_model.evaluate(x_val,   y_val,   verbose=0)
base_gap = base_train_acc - base_val_acc

print(f'Baseline clean train accuracy : {base_train_acc:.4f}')
print(f'Baseline validation accuracy  : {base_val_acc:.4f}')
print(f'Baseline generalization gap   : {base_gap:.4f}')

# 2) Per-class precision / recall / F1.
#    classification_report is imported in cell 1 but never actually used.
print('\nBaseline per-class report')
print(classification_report(y_test, baseline_pred, target_names=class_names,
                            digits=3, zero_division=0))

# 3) Baseline confusion matrix. The notebook only plots one for the
#    augmented model, so there is nothing to compare against without this.
cm_base = confusion_matrix(y_test, baseline_pred)
plt.figure(figsize=(9, 7))
sns.heatmap(cm_base, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted'); plt.ylabel('Actual')
plt.title('Baseline confusion matrix'); plt.tight_layout(); plt.show()

# 4) Settings record — copy straight into the evidence brief.
print('\nSETTINGS RECORD')
print(f'seed=42  train={x_train.shape[0]}  val={x_val.shape[0]}  test={x_test.shape[0]}')
print(f'image shape={x_train.shape[1:]}  classes={len(class_names)}')
print(f'epochs={len(baseline_history.history["accuracy"])}  batch_size=64  optimizer=adam')
print(f'trainable params={baseline_model.count_params():,}')
```

---

### Task 3 — housekeeping

Add a `.gitignore` covering `cifar-10-batches-py/` (~155MB of raw batches that should not be pushed), `__pycache__/`, `*.pyc`, and `.ipynb_checkpoints/`. Leave `results/` tracked — we want those outputs versioned.

---

### Task 4 — reply here with a STAR summary

Include, as plain reported numbers:
- TensorFlow version, and whether it ran on GPU or CPU
- The 5 individual `test_accuracy` values
- Mean ± std of test accuracy, and the **noise floor** (max − min)
- Mean `clean_train_acc`, `clean_val_acc`, and `gap`
- Mean macro precision and macro recall
- The **3 weakest classes by recall**, with their numbers
- Any deviation you had to make from this spec, and why

**Do not interpret the results, draw conclusions, or recommend a hypothesis** — the student is doing that reasoning himself as a graded exercise, and a stated conclusion from you would short-circuit it. Report numbers only.

**Do not modify `Group_1_CNN_Image_Augmentation.ipynb`.** Do not invent numbers — if a step fails, say so plainly and report what you got.
---

---
#### [2] Local-Claude → Partner — Baseline complete (numbers only)

**Situation.** Baseline stage of the Group 1 CIFAR-10 CNN comparison. The augmented model was deliberately left untouched (pre-registration constraint respected) and `Group_1_CNN_Image_Augmentation.ipynb` was not modified.

**Task.** Reproduce the notebook baseline exactly, add post-training measurements across 5 training seeds, write the `results/` artifacts, deliver the diagnostics cell and `.gitignore`, and report numbers only.

**Action.** Created and committed:
- `baseline_runner.py` — loads CIFAR-10 directly from the local `cifar-10-batches-py/` (byte-for-byte replica of the keras loader; no download); data fixed with `DATA_SEED=42` (identical images every run); 5 runs over `RUN_SEEDS=[0,1,2,3,4]` varying only `tf.random.set_seed`; per-run test metrics, `evaluate()`-based clean train/val accuracy and gap, per-class report, and the run-0 confusion matrix. No augmented model is built anywhere in the file.
- `baseline_diagnostics_cell.py` — the paste-in Colab cell, verbatim as specified.
- `.gitignore` — ignores `cifar-10-batches-py/`, `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`; `results/` left tracked.
- `results/` — `baseline_runs.csv`, `baseline_summary.json`, `baseline_history.csv`, `baseline_per_class.csv`, `baseline_settings.txt`, `fig_baseline_curves.png`, `fig_baseline_confusion.png`.
An independent 3-dimension faithfulness review (data pipeline / model+training / metrics+outputs) confirmed the runner matches the notebook and this spec before the training run.

**Result — reported numbers only:**
- **TensorFlow 2.21.0**, ran on **CPU** (native-Windows TF ≥ 2.11 exposes no GPU; GPUs detected: none).
- **5 `test_accuracy` values** (seeds 0→4): `0.4950, 0.5083, 0.5280, 0.5033, 0.5327`
- **test_accuracy mean ± std** (sample, ddof=1): **0.5135 ± 0.0162**; **noise floor (max − min): 0.0377**
- Mean **clean_train_acc = 0.5618**, mean **clean_val_acc = 0.5113**, mean **gap = 0.0505**
- Mean **macro precision = 0.5287**, mean **macro recall = 0.5137**
- **3 weakest classes by mean recall** (across the 5 runs): **bird 0.3149, cat 0.3396, deer 0.3695**
- Params: 282,250 (all trainable). Splits: train 10000 / val 2000 / test 3000; 10 classes; epochs 3; batch 64; optimizer adam. Per-class support ≈ 300/class (sums to 3000).

**Deviations (all environment-level — none change the model, data, or training config):**
1. TensorFlow was not installed on this machine; I installed **TF 2.21.0**. The shell's active Python is a project venv under a long OneDrive path, so the TF install hit the Windows 260-char `MAX_PATH` limit and failed; I created an **isolated venv at a short path** and installed/ran there. Architecture, optimizer, epochs, batch size, splits, and preprocessing are unchanged.
2. Loaded CIFAR-10 from the local `cifar-10-batches-py/` (replicating the keras loader exactly: reshape → transpose to channels-last, labels raveled) instead of `keras.datasets.cifar10.load_data()`, to avoid the ~170 MB download you flagged. The resulting arrays are identical, so the `DATA_SEED=42` subset matches the notebook.
3. TF 2.21 runs with **oneDNN custom ops enabled by default** (can introduce minor floating-point ordering differences); I left it at the default. Run-to-run variation is captured by the noise floor.
4. `baseline_summary.json` `std` is the **sample** std (ddof=1), self-documented in `_meta.std_ddof`; `baseline_history.csv` `epoch` is 1-indexed.

No steps failed; no numbers were invented. Standing by for the next relay message (e.g., the augmented run) once the pre-registered hypotheses are committed.
---
