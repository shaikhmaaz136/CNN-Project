# Pre-registered Hypotheses — Group 1 (Version 1: strict minimal)

> **Pre-registered before any augmented model was trained. Committed to git prior to execution.**

Two conditions only (baseline vs. one model with simple augmentation), paired by seed.

## Design

- **Independent variable:** augmentation — 2 levels (off / on, where "on" = horizontal flip + `RandomRotation(0.05)` + `RandomZoom(0.10)`, the notebook default).
- **Held constant:** architecture, Adam, 3 epochs, batch 64, `DATA_SEED = 42`, the 10,000 / 2,000 / 3,000 split.
- **Repeats:** both conditions on the same seeds 0–9, **paired by seed**.
- **Test statistic:** per-seed difference, `d_s = metric_augmented(s) − metric_baseline(s)`.

## Hypothesis pairs

**Pair 1 — Test accuracy** *(non-directional)*

> **H₀₁:** mean(d_s) = 0 — augmentation does not change test accuracy
> **H₁₁:** mean(d_s) ≠ 0 — augmentation changes test accuracy in some direction

**Pair 2 — Generalization gap**, where gap = clean train accuracy − validation accuracy, both from `evaluate()` in inference mode *(directional)*

> **H₀₂:** mean(d_s) ≥ 0 — augmentation does not reduce the gap
> **H₁₂:** mean(d_s) < 0 — augmentation reduces the gap

*Note the sign flip: smaller gap is better, so "augmentation wins" is `<` here and `≠` above.*

## Manipulation check — not a hypothesis

Training accuracy measured on augmented images should fall relative to baseline. If it does **not**, the augmentation layers did not fire and every other number is uninterpretable. This is a methods-section diagnostic, never a finding.

## Baseline reference values (5 seeds, already measured)

| Quantity | Value |
|---|---|
| Test accuracy | 51.35% (SD 1.62 pp, range 49.50–53.27) |
| Clean train accuracy | 56.18% |
| Validation accuracy | 51.13% |
| Generalization gap | 5.05 pp (SD 1.39) |
| **Noise floor (max − min)** | **3.77 pp** |

## What may be claimed afterwards

- ✅ "Across 10 paired seeds, augmentation changed test accuracy by X pp (95% CI …)."
- ✅ "The observed difference was smaller than the run-to-run variation of the baseline itself."
- ❌ "Augmentation does not improve CNN accuracy." — one architecture, one dataset, one training budget.
- ❌ Anything about fairness, robustness, or real-world readiness. (Roberts et al., 2021)

## What would prove us wrong (falsifiability)

- Test accuracy rises by clearly more than 3.77 pp under augmentation → the underfitting argument was wrong, and augmentation helps even in this regime.
- The generalization gap **widens** under augmentation → the regularization mechanism did not operate as theorised.

*(The flip-only vs. moderate decomposition is out of scope for this two-condition design.)*
