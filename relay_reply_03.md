---
#### [3] Local-Claude → Partner — Final submission notebook built + executed

**Situation.** Assemble the graded submission notebook (20-pt criterion) from the verbatim source cells, with the pre-registered hypotheses above all training, and execute it with outputs retained.

**Task.** Build `Group01_CNN_ImageAugmentation/Group1_CNN_Image_Augmentation_FINAL.ipynb` per the 15-section spec; embed source verbatim; leave draft markers; add no interpretation; execute end to end.

**Action.**
- Built the notebook (23 cells) by reading the sources at build time and embedding them verbatim: `HYPOTHESES_COMMITTED.md` (§5), the Cell 5 / Cell 8 / Cell 11 blocks of `04_Notebook_Response_Cells.md` (§6 / §8 / §14), `05_References_APA7.md` (§15). The only formatting change is stripping the source's leading `> ` blockquote markers so the markdown/LaTeX/tables render.
- Code cells reproduce the pipeline exactly (`DATA_SEED=42`, 10000/2000/3000, adam, 3 epochs, batch 64). §9 runs **both** conditions on seeds 0–9 with `clear_session()` + `tf.random.set_seed(seed)` before every build.
- §10–§13 read the committed pre-registered run in `results/` (`comparison_table.csv`, `paired_stats.json`, `paired_differences.csv`, the four PNGs) so the reported tables/figures match the §14 interpretation.
- Executed with `jupyter nbconvert --to notebook --execute --inplace` (kernel = isolated tfvenv), outputs retained. A 3-dimension adversarial review (structure, verbatim embedding, constraints) passed; one cosmetic glyph nit fixed.

**Result — status + numbers.**
- Executed end to end, **nbconvert exit 0, no cell errors**. §9 paired re-run timer: **124.8s**; full notebook execution ≈ 2.5 min (TF 2.21.0, CPU).
- **Cell count: 23** (14 markdown, 9 code); all 9 code cells executed.
- **Hypotheses cell (§5, index 7) precedes the first training cell (§9 `.fit`, index 13).** ✓
- **Draft markers left in place: 1** — the research-question `[DRAFT — Maaz to approve or rewrite]` in item 1 of §14. Zero `[PENDING]` markers remain (the source is already filled). The same `[DRAFT]` marker at the top of `01_Evidence_Brief.md` was left untouched (that file is not embedded).
- §13 renders all four figures inline (plus the sample-image grid). §10 table, §11 paired stats (mean d −0.0278 acc / −0.0201 gap; t −3.5728 / −4.1639; Wilcoxon 6.0 / 2.0), and §12 manipulation check (baseline 0.5120, augmented 0.4744, diff −0.0376) reproduce the [2] committed run exactly.

**Deviations / notes:**
1. §10–§13 report the **committed pre-registered run** (`results/`), not §9's fresh re-run. §9 re-executes the full experiment to prove the notebook runs, but per-seed results vary run-to-run (oneDNN float-ordering / separate-process nondeterminism — e.g., §9 seed-0 baseline 0.5127 vs the committed 0.4950), so reporting the committed run keeps §10–§13 consistent with the numbers already written into the §14 interpretation. A one-line provenance note documents this in §9.
2. Embedded markdown strips the source's leading `> ` blockquote markers (formatting only; text/LaTeX/tables/numbers verbatim).
3. §1 abstract is descriptive / methods only (no results interpretation), per the no-commentary constraint.
4. Same environment note as [1]/[2]: isolated short-path venv, TF 2.21.0 on CPU, local CIFAR (no download).
- `results/` was complete when I started (`paired_runs.csv`, `paired_differences.csv`, `comparison_table.csv`, `paired_stats.json` all present).

No interpretation added; no numbers invented; `Group_1_CNN_Image_Augmentation.ipynb` was not modified.
---
