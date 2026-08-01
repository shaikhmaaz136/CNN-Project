"""
paired_experiment.py -- Version 1 (two-condition) paired baseline-vs-augmented experiment.

Pre-registered in HYPOTHESES_COMMITTED.md and committed to git BEFORE this ran.
Both conditions, ten seeds, paired by seed. Reports statistics only; it does not
state whether any hypothesis was supported (graded exercise -- the student concludes).

Design (per the pre-registration):
- DATA_SEED = 42, identical images for every run and every condition.
- SEEDS = 0..9; for each seed run BOTH conditions with tf.random.set_seed(seed).
  (The notebook's augmented cell omitted set_seed; that asymmetry is a reproducibility
  bug and is fixed here -- both conditions get the same per-seed init.)
- Data pipeline identical to baseline_runner.py (same subset, split, normalisation).
- Gap = clean_train_acc - clean_val_acc, both from evaluate() in inference mode, so
  augmentation layers are skipped and both conditions are scored on identical clean
  images. fit_final_train_acc (last-epoch fit history) is captured SEPARATELY and is
  used only for the manipulation check (for the augmented model it is measured on
  augmented images, which is exactly why it is unsuitable for the gap).
"""
import os
import time
import json
import pickle
import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
)
from scipy import stats

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_SEED = 42
SEEDS = list(range(10))          # 0..9
CONDITIONS = ["baseline", "augmented"]
EPOCHS = 3
BATCH_SIZE = 64
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]
NOISE_FLOOR = 0.0377             # baseline 5-seed (max - min), per HYPOTHESES_COMMITTED.md

os.environ["PYTHONHASHSEED"] = str(DATA_SEED)
random.seed(DATA_SEED)
np.random.seed(DATA_SEED)
tf.random.set_seed(DATA_SEED)


# --------------------------------------------------------------------------- #
# Data -- IDENTICAL pipeline to baseline_runner.py (local load, no download)
# --------------------------------------------------------------------------- #
def _load_batch(fpath):
    with open(fpath, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    d = {k.decode("utf8"): v for k, v in d.items()}
    data = d["data"].reshape(d["data"].shape[0], 3, 32, 32)
    labels = np.array(d["labels"], dtype="uint8")
    return data, labels


def load_cifar10():
    local = os.path.join(HERE, "cifar-10-batches-py")
    if os.path.isdir(local):
        x_train = np.empty((50000, 3, 32, 32), dtype="uint8")
        y_train = np.empty((50000,), dtype="uint8")
        for i in range(1, 6):
            data, labels = _load_batch(os.path.join(local, "data_batch_%d" % i))
            x_train[(i - 1) * 10000:i * 10000] = data
            y_train[(i - 1) * 10000:i * 10000] = labels
        x_test, y_test = _load_batch(os.path.join(local, "test_batch"))
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        return (x_train, y_train), (x_test, y_test), "local:%s" % local
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    return (x_train, y_train), (x_test, y_test), "keras.datasets.cifar10 (download/cache)"


(x_train_all, y_train_all), (x_test_all, y_test_all), DATA_SOURCE = load_cifar10()
y_train_all = y_train_all.ravel()
y_test_all = y_test_all.ravel()

rng = np.random.default_rng(DATA_SEED)
train_idx = rng.choice(len(x_train_all), 12000, replace=False)
test_idx = rng.choice(len(x_test_all), 3000, replace=False)
x_train, y_train = x_train_all[train_idx], y_train_all[train_idx]
x_test, y_test = x_test_all[test_idx], y_test_all[test_idx]
x_val, y_val = x_train[-2000:], y_train[-2000:]
x_train, y_train = x_train[:-2000], y_train[:-2000]
x_train = x_train.astype("float32") / 255.0
x_val = x_val.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

assert x_train.shape == (10000, 32, 32, 3)
assert x_val.shape == (2000, 32, 32, 3)
assert x_test.shape == (3000, 32, 32, 3)


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def _core_layers():
    return [
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.0),
        layers.Dense(10, activation="softmax"),
    ]


def make_baseline_model():
    model = keras.Sequential([layers.Input(shape=(32, 32, 3))] + _core_layers())
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def make_augmented_model(seed):
    # Same core architecture; augmentation inserted after Input, before the first Conv2D.
    aug = [
        layers.RandomFlip("horizontal", seed=seed),
        layers.RandomRotation(0.05, seed=seed),
        layers.RandomZoom(0.10, seed=seed),
    ]
    model = keras.Sequential([layers.Input(shape=(32, 32, 3))] + aug + _core_layers())
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def build(condition, seed):
    return make_augmented_model(seed) if condition == "augmented" else make_baseline_model()


GPUS = tf.config.list_physical_devices("GPU")
DEVICE = "GPU" if GPUS else "CPU"


# --------------------------------------------------------------------------- #
# Run both conditions on each seed
# --------------------------------------------------------------------------- #
run_rows = []
history_rows = []
per_class_rows = []
histories = {}       # (seed, condition) -> history.history
cms = {}             # condition -> confusion matrix for the first seed

print("=" * 70)
print("PAIRED EXPERIMENT -- seeds %s x conditions %s" % (SEEDS, CONDITIONS))
print("TensorFlow %s | device %s | data source %s" % (tf.__version__, DEVICE, DATA_SOURCE))
print("=" * 70)

t_start = time.time()
for seed in SEEDS:
    for condition in CONDITIONS:
        tf.keras.backend.clear_session()
        tf.random.set_seed(seed)          # BOTH conditions get the same per-seed init
        model = build(condition, seed)

        t0 = time.time()
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
        )
        wall_clock_sec = time.time() - t0
        histories[(seed, condition)] = history.history

        probs = model.predict(x_test, verbose=0)
        pred = probs.argmax(axis=1)
        test_accuracy = accuracy_score(y_test, pred)
        macro_precision = precision_score(y_test, pred, average="macro", zero_division=0)
        macro_recall = recall_score(y_test, pred, average="macro", zero_division=0)

        # Clean-image accuracy, final weights, inference mode (augmentation skipped).
        _, clean_train_acc = model.evaluate(x_train, y_train, verbose=0)
        _, clean_val_acc = model.evaluate(x_val, y_val, verbose=0)
        gap = clean_train_acc - clean_val_acc

        # Manipulation-check number ONLY. For 'augmented' this is on augmented images.
        fit_final_train_acc = history.history["accuracy"][-1]

        run_rows.append({
            "seed": seed,
            "condition": condition,
            "test_accuracy": test_accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "clean_train_acc": clean_train_acc,
            "clean_val_acc": clean_val_acc,
            "gap": gap,
            "fit_final_train_acc": fit_final_train_acc,
            "wall_clock_sec": wall_clock_sec,
        })

        for epoch in range(EPOCHS):
            for metric in ("accuracy", "val_accuracy", "loss", "val_loss"):
                history_rows.append({
                    "seed": seed, "condition": condition,
                    "epoch": epoch + 1, "metric": metric,
                    "value": history.history[metric][epoch],
                })

        report = classification_report(y_test, pred, target_names=CLASS_NAMES,
                                       output_dict=True, zero_division=0)
        for cname in CLASS_NAMES:
            per_class_rows.append({
                "seed": seed, "condition": condition, "class_name": cname,
                "precision": report[cname]["precision"],
                "recall": report[cname]["recall"],
                "f1": report[cname]["f1-score"],
            })

        if seed == SEEDS[0]:
            cms[condition] = confusion_matrix(y_test, pred)

        print("seed=%d %-9s test_acc=%.4f gap=%.4f fit_train=%.4f %.1fs"
              % (seed, condition, test_accuracy, gap, fit_final_train_acc, wall_clock_sec))

total_wall = time.time() - t_start


# --------------------------------------------------------------------------- #
# Assemble frames
# --------------------------------------------------------------------------- #
runs_df = pd.DataFrame(run_rows)
runs_df.to_csv(os.path.join(RESULTS_DIR, "paired_runs.csv"), index=False)

pd.DataFrame(history_rows).to_csv(
    os.path.join(RESULTS_DIR, "paired_history.csv"), index=False)

# Per-seed, aligned baseline/augmented for pairing.
base = runs_df[runs_df.condition == "baseline"].set_index("seed").sort_index()
aug = runs_df[runs_df.condition == "augmented"].set_index("seed").sort_index()
assert list(base.index) == SEEDS and list(aug.index) == SEEDS

diff_df = pd.DataFrame({"seed": SEEDS})
for metric, col in [("test_accuracy", "d_test_accuracy"),
                    ("gap", "d_gap"),
                    ("macro_precision", "d_macro_precision"),
                    ("macro_recall", "d_macro_recall"),
                    ("fit_final_train_acc", "d_fit_final_train_acc")]:
    diff_df[col] = aug[metric].values - base[metric].values
diff_df.to_csv(os.path.join(RESULTS_DIR, "paired_differences.csv"), index=False)


def fmt_mean_sd(series):
    return "%.4f ± %.4f" % (series.mean(), series.std(ddof=1))


comparison_rows = []
for cond, dfc in [("baseline", base), ("augmented", aug)]:
    comparison_rows.append({
        "condition": cond,
        "test_accuracy": fmt_mean_sd(dfc["test_accuracy"]),
        "macro_precision": fmt_mean_sd(dfc["macro_precision"]),
        "macro_recall": fmt_mean_sd(dfc["macro_recall"]),
        "clean_train_acc": fmt_mean_sd(dfc["clean_train_acc"]),
        "clean_val_acc": fmt_mean_sd(dfc["clean_val_acc"]),
        "gap": fmt_mean_sd(dfc["gap"]),
    })
comparison_df = pd.DataFrame(comparison_rows)
comparison_df.to_csv(os.path.join(RESULTS_DIR, "comparison_table.csv"), index=False)

# Per-class means across seeds, per condition.
per_class_df = pd.DataFrame(per_class_rows)
per_class_comp = (per_class_df.groupby(["class_name", "condition"])[["precision", "recall", "f1"]]
                  .mean().round(4).reset_index())
per_class_comp.to_csv(os.path.join(RESULTS_DIR, "per_class_comparison.csv"), index=False)


# --------------------------------------------------------------------------- #
# Task 3 -- paired statistics
# --------------------------------------------------------------------------- #
def paired_block(aug_vals, base_vals, alternative):
    """alternative applies to BOTH the paired t-test and the Wilcoxon test.
    'two-sided' for test accuracy (Pair 1), 'less' for gap (Pair 2, H1: mean d < 0)."""
    a = np.asarray(aug_vals, dtype=float)
    b = np.asarray(base_vals, dtype=float)
    d = a - b
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    tcrit = float(stats.t.ppf(0.975, n - 1))          # 95% CI, df = n-1 = 9
    ci = [mean - tcrit * se, mean + tcrit * se]
    cohen_d = float(mean / sd) if sd > 0 else float("nan")
    t_res = stats.ttest_rel(a, b, alternative=alternative)
    w_res = stats.wilcoxon(a, b, alternative=alternative)
    return {
        "differences": [float(v) for v in d],
        "n": n,
        "mean_difference": mean,
        "sd_difference": sd,
        "ci95_mean_difference": ci,
        "cohen_d_paired": cohen_d,
        "paired_ttest": {"alternative": alternative,
                         "t": float(t_res.statistic), "p": float(t_res.pvalue), "df": n - 1},
        "wilcoxon": {"alternative": alternative,
                     "statistic": float(w_res.statistic), "p": float(w_res.pvalue)},
        "mean_abs_diff_over_noise_floor": float(np.mean(np.abs(d)) / NOISE_FLOOR),
    }


stats_out = {
    "n_seeds": len(SEEDS),
    "noise_floor_reference": NOISE_FLOOR,
    "test_accuracy": paired_block(aug["test_accuracy"].values,
                                  base["test_accuracy"].values, "two-sided"),
    "gap": paired_block(aug["gap"].values, base["gap"].values, "less"),
    "manipulation_check": {
        "mean_fit_final_train_acc_baseline": float(base["fit_final_train_acc"].mean()),
        "mean_fit_final_train_acc_augmented": float(aug["fit_final_train_acc"].mean()),
        "difference_aug_minus_base": float(aug["fit_final_train_acc"].mean()
                                           - base["fit_final_train_acc"].mean()),
    },
    "total_wall_clock_sec": float(total_wall),
    "tensorflow": tf.__version__,
    "device": DEVICE,
}
with open(os.path.join(RESULTS_DIR, "paired_stats.json"), "w", encoding="utf-8") as f:
    json.dump(stats_out, f, indent=2)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
# 1) Slope plot: one line per seed, baseline -> augmented test accuracy.
plt.figure(figsize=(6, 6))
for s in SEEDS:
    yb = base.loc[s, "test_accuracy"]
    ya = aug.loc[s, "test_accuracy"]
    plt.plot([0, 1], [yb, ya], marker="o", color="tab:gray", alpha=0.6, linewidth=1)
plt.plot([0, 1], [base["test_accuracy"].mean(), aug["test_accuracy"].mean()],
         color="black", linewidth=3, marker="o", label="mean")
plt.xticks([0, 1], ["baseline", "augmented"])
plt.ylabel("test accuracy")
plt.title("Paired test accuracy per seed (one line per seed)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig_paired_slope.png"), dpi=120)
plt.close()

# 2) Per-seed differences with mean and 95% CI, zero line -- test accuracy and gap.
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, key, label in ((axes[0], "test_accuracy", "d test accuracy (aug - base)"),
                       (axes[1], "gap", "d gap (aug - base)")):
    d = np.array(stats_out[key]["differences"])
    m = stats_out[key]["mean_difference"]
    ci = stats_out[key]["ci95_mean_difference"]
    ax.axhline(0, color="grey", linestyle="--", linewidth=1)
    ax.scatter(SEEDS, d, color="tab:blue", zorder=3, label="per-seed d")
    ax.errorbar(len(SEEDS) + 0.5, m, yerr=[[m - ci[0]], [ci[1] - m]],
                fmt="o", color="black", capsize=5, label="mean ± 95% CI")
    ax.set_xlabel("seed")
    ax.set_ylabel(label)
    ax.set_title(label)
    ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig_difference_ci.png"), dpi=120)
plt.close(fig)

# 3) Mean train/val accuracy & loss for both conditions; individual runs faded.
epochs_axis = list(range(1, EPOCHS + 1))
cond_color = {"baseline": "tab:blue", "augmented": "tab:orange"}
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (train_key, val_key, name) in zip(
        axes, (("accuracy", "val_accuracy", "accuracy"), ("loss", "val_loss", "loss"))):
    for condition in CONDITIONS:
        tr = np.array([histories[(s, condition)][train_key] for s in SEEDS])
        va = np.array([histories[(s, condition)][val_key] for s in SEEDS])
        c = cond_color[condition]
        for s_i in range(len(SEEDS)):
            ax.plot(epochs_axis, tr[s_i], color=c, alpha=0.12, linewidth=1)
            ax.plot(epochs_axis, va[s_i], color=c, alpha=0.12, linewidth=1, linestyle="--")
        ax.plot(epochs_axis, tr.mean(axis=0), color=c, linewidth=2.5,
                label="%s train (mean)" % condition)
        ax.plot(epochs_axis, va.mean(axis=0), color=c, linewidth=2.5, linestyle="--",
                label="%s val (mean)" % condition)
    ax.set_xlabel("epoch")
    ax.set_ylabel(name)
    ax.set_xticks(epochs_axis)
    ax.set_title("Baseline vs augmented: %s" % name)
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig_curves_comparison.png"), dpi=120)
plt.close(fig)

# 4) Confusion matrices side by side (first seed), shared colour scale.
vmax = int(max(cms["baseline"].max(), cms["augmented"].max()))
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
for ax, condition in zip(axes, CONDITIONS):
    sns.heatmap(cms[condition], annot=True, fmt="d", cmap="Blues", vmin=0, vmax=vmax,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax, cbar=True)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("%s confusion matrix (seed %d)" % (condition, SEEDS[0]))
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig_confusion_side_by_side.png"), dpi=120)
plt.close(fig)


# --------------------------------------------------------------------------- #
# Compact stdout summary (numbers only)
# --------------------------------------------------------------------------- #
print("\n" + "=" * 70)
print("COMPARISON TABLE (mean ± SD across 10 seeds)")
print("=" * 70)
print(comparison_df.to_string(index=False))

for key, tail in (("test_accuracy", "two-sided"), ("gap", "less (H1: mean d < 0)")):
    b = stats_out[key]
    print("\n--- %s  [%s] ---" % (key, tail))
    print("per-seed d : %s" % [round(v, 4) for v in b["differences"]])
    print("mean d = %.4f | SD = %.4f | 95%% CI = [%.4f, %.4f]"
          % (b["mean_difference"], b["sd_difference"],
             b["ci95_mean_difference"][0], b["ci95_mean_difference"][1]))
    print("paired t = %.4f, p = %.4f (df=%d) | Wilcoxon stat = %.1f, p = %.4f"
          % (b["paired_ttest"]["t"], b["paired_ttest"]["p"], b["paired_ttest"]["df"],
             b["wilcoxon"]["statistic"], b["wilcoxon"]["p"]))
    print("Cohen's d (paired) = %.4f | mean|d| / noise_floor(0.0377) = %.3f"
          % (b["cohen_d_paired"], b["mean_abs_diff_over_noise_floor"]))

mc = stats_out["manipulation_check"]
print("\n--- manipulation check (fit_final_train_acc) ---")
print("baseline mean = %.4f | augmented mean = %.4f | diff (aug-base) = %.4f"
      % (mc["mean_fit_final_train_acc_baseline"],
         mc["mean_fit_final_train_acc_augmented"],
         mc["difference_aug_minus_base"]))
print("\ntotal wall clock = %.1fs" % total_wall)
print("Wrote outputs to %s" % RESULTS_DIR)
