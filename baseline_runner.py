"""
baseline_runner.py -- Group 1 CNN baseline reproduction (BASELINE ONLY).

Reproduces the baseline from Group_1_CNN_Image_Augmentation.ipynb exactly
(architecture, optimizer, epoch count, batch size, split sizes, preprocessing).
The only additions are measurements taken *after* training, plus a 5-run repeat
so a run-to-run noise floor is available. None of these additions change the
model, the data, or the training.

Deliberately DOES NOT build, train, evaluate, or sketch any augmented model.

Determinism note: the data subset is drawn once with DATA_SEED = 42, so every
run sees the identical images. Only the weight-initialisation / training seed
(tf.random.set_seed(run_seed)) varies across the 5 runs, which isolates
training variation from data variation.
"""
import os
import time
import json
import pickle
import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless: render figures straight to files
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

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

DATA_SEED = 42                    # fixed for ALL runs -- data identical every run
RUN_SEEDS = [0, 1, 2, 3, 4]       # only weight-init / training seed varies
EPOCHS = 3
BATCH_SIZE = 64
CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

# Reproducibility hygiene mirroring the notebook's first cell. The per-run
# tf.random.set_seed(run_seed) below is what actually governs each run.
os.environ["PYTHONHASHSEED"] = str(DATA_SEED)
random.seed(DATA_SEED)
np.random.seed(DATA_SEED)
tf.random.set_seed(DATA_SEED)


# --------------------------------------------------------------------------- #
# Data -- load the local CIFAR-10 copy (no download); byte-for-byte identical
# to keras.datasets.cifar10.load_data().
# --------------------------------------------------------------------------- #
def _load_batch(fpath):
    """Replicates keras.datasets.cifar10.load_batch exactly."""
    with open(fpath, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    d = {k.decode("utf8"): v for k, v in d.items()}
    data = d["data"].reshape(d["data"].shape[0], 3, 32, 32)
    labels = np.array(d["labels"], dtype="uint8")
    return data, labels


def load_cifar10():
    """Prefer the local cifar-10-batches-py folder; fall back to keras loader."""
    local = os.path.join(HERE, "cifar-10-batches-py")
    if os.path.isdir(local):
        x_train = np.empty((50000, 3, 32, 32), dtype="uint8")
        y_train = np.empty((50000,), dtype="uint8")
        for i in range(1, 6):
            data, labels = _load_batch(os.path.join(local, "data_batch_%d" % i))
            x_train[(i - 1) * 10000:i * 10000] = data
            y_train[(i - 1) * 10000:i * 10000] = labels
        x_test, y_test = _load_batch(os.path.join(local, "test_batch"))
        # keras returns channels_last images and (N, 1) labels
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

# Fixed subsets keep the same cases across every run (identical to the notebook).
rng = np.random.default_rng(DATA_SEED)
train_idx = rng.choice(len(x_train_all), 12000, replace=False)
test_idx = rng.choice(len(x_test_all), 3000, replace=False)
x_train, y_train = x_train_all[train_idx], y_train_all[train_idx]
x_test, y_test = x_test_all[test_idx], y_test_all[test_idx]

# Validation = last 2000 of the 12000; training = first 10000.
x_val, y_val = x_train[-2000:], y_train[-2000:]
x_train, y_train = x_train[:-2000], y_train[:-2000]

# Normalize pixels 0-255 -> 0-1.
x_train = x_train.astype("float32") / 255.0
x_val = x_val.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

assert x_train.shape == (10000, 32, 32, 3), x_train.shape
assert x_val.shape == (2000, 32, 32, 3), x_val.shape
assert x_test.shape == (3000, 32, 32, 3), x_test.shape


# --------------------------------------------------------------------------- #
# Model -- identical to make_baseline_model(dropout_rate=0.0)
# --------------------------------------------------------------------------- #
def make_baseline_model(dropout_rate=0.0):
    model = keras.Sequential([
        layers.Input(shape=(32, 32, 3)),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(dropout_rate),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


# --------------------------------------------------------------------------- #
# Device / environment
# --------------------------------------------------------------------------- #
GPUS = tf.config.list_physical_devices("GPU")
DEVICE = "GPU" if GPUS else "CPU"


# --------------------------------------------------------------------------- #
# 5 baseline runs
# --------------------------------------------------------------------------- #
run_rows = []          # one row per run for baseline_runs.csv
history_rows = []      # long-format per-epoch history
per_class_rows = []    # per-run per-class precision/recall/f1
histories = {}         # run_seed -> history.history (for the curves figure)
first_run_cm = None    # confusion matrix for run_seed 0
param_total = None
param_trainable = None

print("=" * 68)
print("BASELINE RUNNER -- %d runs, seeds %s" % (len(RUN_SEEDS), RUN_SEEDS))
print("TensorFlow %s | device %s | data source %s" % (tf.__version__, DEVICE, DATA_SOURCE))
print("=" * 68)

for run_seed in RUN_SEEDS:
    tf.keras.backend.clear_session()
    tf.random.set_seed(run_seed)
    model = make_baseline_model(0.0)  # rebuilt fresh each run

    if param_total is None:
        param_total = int(model.count_params())
        param_trainable = int(sum(int(np.prod(w.shape)) for w in model.trainable_weights))

    t0 = time.time()
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
    )
    wall_clock_sec = time.time() - t0
    histories[run_seed] = history.history

    # Test-set predictions (3000 images).
    probs = model.predict(x_test, verbose=0)
    pred = probs.argmax(axis=1)
    test_accuracy = accuracy_score(y_test, pred)
    macro_precision = precision_score(y_test, pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_test, pred, average="macro", zero_division=0)

    # Clean-image accuracy with FINAL weights, inference mode (fair gap number).
    _, clean_train_acc = model.evaluate(x_train, y_train, verbose=0)
    _, clean_val_acc = model.evaluate(x_val, y_val, verbose=0)
    gap = clean_train_acc - clean_val_acc

    run_rows.append({
        "run_seed": run_seed,
        "test_accuracy": test_accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "clean_train_acc": clean_train_acc,
        "clean_val_acc": clean_val_acc,
        "gap": gap,
        "wall_clock_sec": wall_clock_sec,
    })

    for epoch in range(EPOCHS):
        for metric in ("accuracy", "val_accuracy", "loss", "val_loss"):
            history_rows.append({
                "run_seed": run_seed,
                "epoch": epoch + 1,               # 1-indexed
                "metric": metric,
                "value": history.history[metric][epoch],
            })

    report = classification_report(
        y_test, pred, target_names=CLASS_NAMES,
        output_dict=True, zero_division=0,
    )
    for cname in CLASS_NAMES:
        per_class_rows.append({
            "run_seed": run_seed,
            "class_name": cname,
            "precision": report[cname]["precision"],
            "recall": report[cname]["recall"],
            "f1": report[cname]["f1-score"],
            "support": int(report[cname]["support"]),
        })

    if run_seed == RUN_SEEDS[0]:  # run_seed 0
        first_run_cm = confusion_matrix(y_test, pred)

    print("run_seed=%d  test_acc=%.4f  macro_P=%.4f  macro_R=%.4f  gap=%.4f  %.1fs"
          % (run_seed, test_accuracy, macro_precision, macro_recall, gap, wall_clock_sec))


# --------------------------------------------------------------------------- #
# Aggregate + write outputs
# --------------------------------------------------------------------------- #
runs_df = pd.DataFrame(run_rows)
runs_df.to_csv(os.path.join(RESULTS_DIR, "baseline_runs.csv"), index=False)

pd.DataFrame(history_rows).to_csv(
    os.path.join(RESULTS_DIR, "baseline_history.csv"), index=False)
per_class_df = pd.DataFrame(per_class_rows)
per_class_df.to_csv(os.path.join(RESULTS_DIR, "baseline_per_class.csv"), index=False)

# Summary JSON: mean / std / min / max of each scalar across the 5 runs.
scalar_cols = ["test_accuracy", "macro_precision", "macro_recall",
               "clean_train_acc", "clean_val_acc", "gap", "wall_clock_sec"]
summary = {"_meta": {"n_runs": len(RUN_SEEDS), "run_seeds": RUN_SEEDS,
                     "std_ddof": 1}}
for col in scalar_cols:
    summary[col] = {
        "mean": float(runs_df[col].mean()),
        "std": float(runs_df[col].std(ddof=1)),
        "min": float(runs_df[col].min()),
        "max": float(runs_df[col].max()),
    }
summary["noise_floor"] = float(runs_df["test_accuracy"].max()
                               - runs_df["test_accuracy"].min())
with open(os.path.join(RESULTS_DIR, "baseline_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

# Settings record.
with open(os.path.join(RESULTS_DIR, "baseline_settings.txt"), "w") as f:
    f.write("BASELINE SETTINGS RECORD\n")
    f.write("data_seed=%d\n" % DATA_SEED)
    f.write("run_seeds=%s\n" % RUN_SEEDS)
    f.write("train_size=%d  val_size=%d  test_size=%d\n"
            % (x_train.shape[0], x_val.shape[0], x_test.shape[0]))
    f.write("image_shape=%s\n" % (x_train.shape[1:],))
    f.write("class_count=%d\n" % len(CLASS_NAMES))
    f.write("epochs=%d  batch_size=%d\n" % (EPOCHS, BATCH_SIZE))
    f.write("optimizer=adam  loss=sparse_categorical_crossentropy\n")
    f.write("total_params=%d  trainable_params=%d\n" % (param_total, param_trainable))
    f.write("tensorflow=%s\n" % tf.__version__)
    f.write("device=%s\n" % DEVICE)
    f.write("data_source=%s\n" % DATA_SOURCE)

# Figure 1: train vs. validation accuracy and loss, all 5 runs faded + mean bold.
epochs_axis = list(range(1, EPOCHS + 1))
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for pair, ax, title in ((("accuracy", "val_accuracy"), axes[0], "accuracy"),
                        (("loss", "val_loss"), axes[1], "loss")):
    train_key, val_key = pair
    train_stack = np.array([histories[s][train_key] for s in RUN_SEEDS])
    val_stack = np.array([histories[s][val_key] for s in RUN_SEEDS])
    for s in RUN_SEEDS:
        ax.plot(epochs_axis, histories[s][train_key], color="C0", alpha=0.25, linewidth=1)
        ax.plot(epochs_axis, histories[s][val_key], color="C1", alpha=0.25, linewidth=1)
    ax.plot(epochs_axis, train_stack.mean(axis=0), color="C0", linewidth=2.5, label="train (mean)")
    ax.plot(epochs_axis, val_stack.mean(axis=0), color="C1", linewidth=2.5, label="validation (mean)")
    ax.set_title("Baseline: %s (5 runs)" % title)
    ax.set_xlabel("epoch")
    ax.set_ylabel(title)
    ax.set_xticks(epochs_axis)
    ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig_baseline_curves.png"), dpi=120)
plt.close(fig)

# Figure 2: confusion matrix for run_seed 0, annotated + labelled.
plt.figure(figsize=(9, 7))
sns.heatmap(first_run_cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Baseline confusion matrix (run_seed 0)")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "fig_baseline_confusion.png"), dpi=120)
plt.close()


# --------------------------------------------------------------------------- #
# Concise stdout summary (real numbers for the relay STAR reply)
# --------------------------------------------------------------------------- #
print("\n" + "=" * 68)
print("SUMMARY (5 runs)")
print("=" * 68)
print(runs_df.round(4).to_string(index=False))

acc = runs_df["test_accuracy"]
print("\ntest_accuracy per run : %s" % [round(v, 4) for v in acc.tolist()])
print("test_accuracy mean±std: %.4f ± %.4f" % (acc.mean(), acc.std(ddof=1)))
print("noise_floor (max-min) : %.4f" % (acc.max() - acc.min()))
print("clean_train_acc mean  : %.4f" % runs_df["clean_train_acc"].mean())
print("clean_val_acc   mean  : %.4f" % runs_df["clean_val_acc"].mean())
print("gap             mean  : %.4f" % runs_df["gap"].mean())
print("macro_precision mean  : %.4f" % runs_df["macro_precision"].mean())
print("macro_recall    mean  : %.4f" % runs_df["macro_recall"].mean())

mean_recall = (per_class_df.groupby("class_name")["recall"].mean()
               .sort_values())
print("\nmean recall per class (ascending):")
print(mean_recall.round(4).to_string())
print("\n3 weakest classes by mean recall:")
print(mean_recall.head(3).round(4).to_string())
print("\nWrote outputs to %s" % RESULTS_DIR)
