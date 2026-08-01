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
