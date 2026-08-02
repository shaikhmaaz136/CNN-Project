# CNN Learning Notes. Group 1: Image Augmentation

**A living document.** I add to this as we work through the project together. Read the module for whatever we're currently doing; skim the rest.

- **Project:** Statistics for AI Residency, Group 1 (CNN pathway)
- **Notebook:** `Group_1_CNN_Image_Augmentation.ipynb`
- **Assigned comparison:** baseline CNN (no augmentation) **vs.** same CNN with simple augmentation
- **Started:** 2026-08-01

**Status legend:** ✅ covered · 🔜 next up · ⬜ later

---

## Curriculum map

| # | Module | Why the project needs it | Status |
|---|---|---|---|
| 1 | Research question, hypothesis, variables | Graded directly; everything downstream aligns to it | ✅ |
| 1B | Hypothesis design, from scratch | You asked for it directly; graded under research question alignment | ✅ |
| 1C | Your run protocol: confirmatory vs. exploratory | Fixes the order you actually execute in; protects the research-foundation and reproducibility marks | ✅ |
| 1D | Running the baseline: what to capture | The starter notebook leaves per-class and baseline-confusion data uncollected; cheap now, a re-run later | ✅ |
| R | What our own results taught us | The statistics re-taught from the numbers you actually produced; feeds interpretation, limitations and next steps directly | ✅ |
| 2 | The dataset and preprocessing | Rubric: dataset + preprocessing | ✅ |
| 3 | What convolution actually does | Rubric: architecture; you must explain your own model | ✅ |
| 4 | Pooling, flatten, dense head | Same | ✅ |
| 5 | Activations: ReLU and softmax | Softmax is an approved "math foundation" | ✅ |
| 6 | Loss and optimization | Cross-entropy is an approved "math foundation" | ✅ |
| 7 | Training mechanics: epochs, batches, seeds | Reproducibility is 20 pts on the instructor rubric | ✅ |
| 8 | Overfitting and the generalization gap | The whole *reason* augmentation exists | ✅ |
| 9 | Augmentation in depth | Your assigned intervention | ✅ |
| 10 | Metrics: accuracy, precision, recall, macro | Required reporting | ✅ |
| 11 | Confusion matrix reading | Required visual | ✅ |
| 12 | Limitations and claim boundaries | Explicitly graded; easy points, easily lost | ✅ |

---

## Module 1: Research question, hypothesis, variables ✅

### 1.1 Three different things people call "the question"

These get muddled constantly. Keep them separate:

| Level | Example here | Who decides it |
|---|---|---|
| **Central question** (the course's) | How can a model use observed data to predict, and how do we evaluate what changes when something changes? | Instructor |
| **Pathway question** (your group's assignment) | How does basic image augmentation affect classification performance? | Instructor |
| **Focused research question** (yours to write) | Does horizontal-flip + small rotation/zoom augmentation change the test accuracy of a fixed 2-conv-layer CNN on a 10,000-image CIFAR-10 subset, relative to an identical unaugmented baseline? | **You** |
| **Hypothesis** (yours to write) | A directional, falsifiable prediction about that outcome, committed to *before* the run | **You** |

The focused question must name three things: the **outcome** you'll measure, the **intervention** you'll apply, and the **comparison** you'll make against. If a reader can't extract all three from one sentence, it isn't focused yet.

### 1.2 Variables

| Type | Meaning | In your project |
|---|---|---|
| **Independent** (what you manipulate) | The one thing you change | Whether training images are augmented: **off** vs. **on** |
| **Dependent** (what you measure) | The outcome | Test accuracy; also macro precision, macro recall, train–val gap |
| **Controlled** (held constant) | Everything else, so any difference is attributable to the IV | Architecture, optimizer (Adam), epochs (3), batch size (64), data split, seed (42) |

The controlled list is what makes this an **experiment** rather than an observation. Your notebook already handles it: both models are built from the same layer stack, both call `tf.random.set_seed(SEED)`, both train on the exact same `x_train`. That's not an accident, the starter code was written to make the comparison clean, and you should say so out loud in the presentation.

⚠️ One honest caveat about the current notebook: the baseline cell calls `tf.keras.backend.clear_session()` and re-seeds before building, but the augmented cell does not. That's a small reproducibility gap worth fixing before your final run, it means the two models don't start from identical random weight initialisations. We'll deal with it in Module 7.

### 1.3 What makes something a *hypothesis* and not just a guess

Three properties:

1. **Directional**, it says *which way*, not just "there will be a difference." "Augmentation will improve test accuracy" is a hypothesis; "augmentation will affect accuracy" is barely one.
2. **Falsifiable**, there is a specific result that would prove it wrong. If no possible outcome contradicts your statement, it isn't science.
3. **Pre-committed**, you write it down before you look at the augmented model's numbers. Writing it after is called HARKing (Hypothesising After Results are Known) and it's the thing that makes a finding untrustworthy.

### 1.4 H₀ and H₁: and why the null exists

- **H₀ (null hypothesis):** augmentation makes no difference. Formally, μ_augmented = μ_baseline.
- **H₁ (alternative hypothesis):** augmentation improves test accuracy. μ_augmented > μ_baseline.

The null feels like a strange thing to bother writing down, of *course* you don't believe it. But that's the point. In statistics you never "prove" H₁ directly; you assume H₀ (nothing is happening, any difference is noise) and ask how surprised you'd be to see your data under that assumption. The null is the boring explanation you're trying to rule out. Naming it forces you to admit that "the augmented model scored higher" and "augmentation works" are two different claims.

### 1.5 The mechanism: *why* would augmentation help?

This is the part that turns a guess into a research-grounded hypothesis, and it's the single most valuable idea in this module.

A CNN learns only from what it's shown. Suppose most horses in your 10,000 training images happen to face left. The model has no way to know that facing direction is irrelevant to "horseness", so it may quietly learn "horse = leftward-facing quadruped shape." Show it a right-facing horse at test time and it stumbles.

Horizontal flip fixes exactly that. It tells the model: *these two images are the same class.*

> **Augmentation is not really "more data." It is a way of injecting invariances you already know are true into the training process.**

That sentence is your presentation's best line. It also gives you the rule for choosing transforms: **only apply a transform whose invariance genuinely holds in your domain.**

- Horizontal flip on CIFAR-10 ✅, a mirrored car is still a car.
- **Vertical** flip on CIFAR-10 ❌, upside-down airplanes and cars essentially don't occur. You'd be teaching the model to expect something the world never shows it.
- Small rotation ✅, photos are handheld and slightly tilted all the time.
- Large rotation on 32×32 images ⚠️, at this tiny resolution, rotation crops corners and interpolation smears detail. You can destroy signal.

That last point is your justification hook when you choose intensity: the tradeoff is **added invariance vs. destroyed signal at low resolution.**

### 1.6 The trap: why augmentation might LOSE at 3 epochs

Predict this now and you look sharp instead of broken.

Augmentation makes the training task **harder**. Every epoch, the model sees a differently-flipped, differently-rotated version of each image, so it can never simply memorise. Two consequences:

1. **Training accuracy goes down**, or rises more slowly, versus the baseline. This is expected and is *not* a bug.
2. **The payoff arrives late.** Augmentation's benefit is reduced overfitting, but a model has to actually start overfitting before there's anything to reduce.

Your notebook trains for **3 epochs on 10,000 images**. That is very likely not enough for the baseline to overfit meaningfully. If the baseline hasn't overfit yet, augmentation has no problem to solve and mostly just slows learning down. **A null or slightly negative result is a realistic, defensible outcome here.**

The assignment brief explicitly backs you up: *"A failed experiment does not automatically mean a failed project... Grading emphasizes sound reasoning, reproducibility, comparison, and interpretation more than model performance."*

So the strong move is a **two-part hypothesis**:

- **H₁a (accuracy):** augmented test accuracy > baseline test accuracy.
- **H₁b (generalization gap):** the train-minus-validation accuracy gap is *smaller* for the augmented model than the baseline.

H₁b is far more likely to hold at 3 epochs than H₁a, and it measures augmentation's actual mechanism rather than a downstream side effect. If H₁a fails but H₁b holds, you have a genuinely interesting finding: *the intervention did what it was designed to do; the training budget was too short for that to convert into test-set gains.*

### 1.7 Candidate focused research questions

Pick one as *the* question. The others become material for the interpretation and "next step" sections.

| Option | Question | Best if |
|---|---|---|
| **A. Accuracy** | Does basic augmentation improve test accuracy vs. the unaugmented baseline? | You want the cleanest match to the assigned comparison |
| **B. Generalization** | Does augmentation reduce the train–validation accuracy gap, even if test accuracy barely moves? | You want the question that survives a null result gracefully |
| **C. Class-level** | Does augmentation help more for visually confusable classes (cat/dog, automobile/truck) than for distinctive ones? | You want to make real use of the required confusion matrix |
| **A+B combined** | Does augmentation improve test accuracy **and/or** reduce the generalization gap? | **Recommended**, one intervention, two outcomes, no extra runs needed |

A+B is not "changing two things." You still change exactly one condition (augmentation on/off). You just measure two outcomes from the same pair of runs, which is completely standard and costs you nothing.

### 1.8 What you can and cannot claim

You will run **one model per condition**. That means:

- ✅ You can say: "In this run, the augmented model scored X vs. the baseline's Y."
- ❌ You cannot say: "Augmentation significantly improves accuracy." With n=1 per condition there is no variance estimate, so no p-value, no significance, full stop.
- ✅ You can name this as your limitation and propose the fix as your next step: **re-run both conditions across 5–10 random seeds and compare the distributions**. That's a specific, credible, cheap next step, much better than a vague "train longer."

Neural network training is genuinely stochastic. Two runs of the *identical* config with different seeds can differ by a percentage point or more on a subset this small. So if your two conditions differ by 0.4%, the honest reading is "indistinguishable from run-to-run noise," not "augmentation helped slightly."

### 1.9 ✍️ Your turn: fill these in before running the augmented model

```
Focused research question:

Independent variable (and its two levels):

Primary dependent variable:

Secondary dependent variable(s):

H₀:

H₁:

Mechanism, why do we expect this, in one sentence:

What result would prove us wrong:
```

---

## Module 1B: Hypothesis design, from scratch ✅

You asked for this one directly, and you were right to. Module 1 handed you a finished hypothesis and told you it was a good one. This module builds a hypothesis up from parts, so that next time you can tell a good one from a bad one without me. None of it is CNN-specific, it's ordinary experimental design, but every example comes from your project so nothing stays abstract.

### 1B.1 A hypothesis is a claim about the world, not about your data

This is the distinction beginners miss most often, so we start here.

| Statement | What it actually is |
|---|---|
| "The augmented model scored 62% on the test set." | An **observation**, a fact about 3,000 specific images on one specific afternoon. |
| "Augmentation improves classification accuracy." | A **hypothesis**, a claim about how the world works, which reaches beyond your run. |

The observation is *evidence about* the hypothesis. It is not the hypothesis. You collect the first in order to argue about the second, and the whole apparatus of statistics exists to manage the gap between them.

A quick diagnostic you can apply to any sentence you've written: **if it could be settled by looking at your results table, it's an observation.** A hypothesis has to be the kind of thing your results table can only *support* or *undermine*. It should also be writable before the data exists, if you couldn't have written it last week, it's a description of your numbers wearing a hypothesis costume.

### 1B.2 Why a null hypothesis exists at all: the courtroom

The null feels like bureaucratic paperwork until you see where it comes from.

A criminal trial starts from the assumption that the defendant is innocent. The prosecution never proves guilt in a direct, positive way, there's no way to do that. Instead it presents evidence and argues that this evidence would be wildly improbable if the defendant were innocent. And notice the verdicts available to the jury: *guilty* and *not guilty*. There is no verdict of "proven innocent." "Not guilty" only ever means the evidence wasn't strong enough to overturn the starting assumption.

Statistical testing is that structure, moved into arithmetic:

| Courtroom | Your experiment |
|---|---|
| Presumption of innocence | **H₀:** augmentation changes nothing; any difference is noise |
| The prosecution's evidence | Your measured accuracies |
| "Would this evidence be absurd for an innocent person?" | "Would this difference be absurd if augmentation did nothing?" |
| Verdict: guilty | **Reject H₀** |
| Verdict: not guilty | **Fail to reject H₀** |
| No verdict of "proven innocent" exists | **No "accept H₀" exists** |

So the null is not a hypothesis you believe. It's the boring explanation you're trying to make untenable, and you need it written down explicitly, because you can only measure how surprising your data is *relative to some assumption*. Without a stated null there is nothing to be surprised against.

This is also why the vocabulary is so awkward. "Reject the null" is a double negative, and students always want to replace it with "prove my hypothesis." Resist that. The logic is **falsification, not confirmation**: you cannot prove your effect exists, you can only show that the data would be surprising if it didn't. Every careful sentence in a results section is shaped by that limitation.

### 1B.3 The two rules every H₀/H₁ pair must obey

1. **Mutually exclusive**, they cannot both be true.
2. **Exhaustive**, between them they cover every possible outcome.

Here is a pair that breaks the second rule, and it's the single most common error I see:

> H₀: μ_aug = μ_base
> H₁: μ_aug > μ_base

Mutually exclusive? Yes. Exhaustive? No. The outcome **μ_aug < μ_base**, the augmented model doing *worse*, is covered by neither statement. And that isn't a hypothetical for you: at 3 epochs it's a live possibility. If it happens, your framework has literally nothing to say about the result you got, which is a bad position to be in at 9am on presentation day.

Two repairs, both legitimate. Pick one; don't mix them.

| Repair | H₀ | H₁ | What you're now asking |
|---|---|---|---|
| **Make H₁ non-directional** | μ_aug = μ_base | μ_aug ≠ μ_base | "Is there a difference at all, in either direction?" |
| **Make H₀ inclusive** | μ_aug ≤ μ_base | μ_aug > μ_base | "Is augmentation better?", worse *and* equal both live inside H₀ |

The second repair is the one people find strange, so look at what it does: it bundles "augmentation is worse" together with "augmentation makes no difference" into a single null. That bundling is exactly right, because both outcomes mean the same thing for your claim, **you failed to show augmentation helps.** The null doesn't have to be a single tidy point. It just has to be everything that isn't your alternative.

### 1B.4 Directional vs. non-directional

|  | Non-directional (two-tailed) | Directional (one-tailed) |
|---|---|---|
| **H₀** | μ_aug = μ_base | μ_aug ≤ μ_base |
| **H₁** | μ_aug ≠ μ_base | μ_aug > μ_base |
| **The claim** | "Something changes" | "It goes up" |
| **Statistical power** | Lower, sensitivity is split across both tails | Higher, all of it is concentrated in one tail |
| **The price** | A vaguer conclusion | You must commit *before* seeing any data |
| **If the result goes the other way** | Still detected as a difference | You report "failed to reject," and that's all |

Directional is the stronger, more useful, more scientific claim, it says something the world could contradict, and you get better sensitivity for free, because you've told the test where to look. Module 1 already told you to prefer it.

But understand what you're buying it with. The price is a genuine commitment. If you predict augmented > baseline and the augmented model comes in *lower*, the honest write-up says: **we failed to reject H₀**, and then discusses the unexpected direction in your interpretation section, where such discussion belongs. What you may not do is quietly rotate H₁ around to "augmentation reduces accuracy at low epoch counts" and declare a win. That's mistake (e) in the next section, and it's the one that actually invalidates work.

### 1B.5 "Fail to reject" is not "accept"

You never accept H₀. The phrase is not in the vocabulary, and the reason is a single sentence worth memorising: **absence of evidence is not evidence of absence.**

Two very different worlds produce an identical null result:

1. There genuinely is no effect.
2. There is a real effect, and your study was too small, too short, or too noisy to detect it.

A null result cannot tell those apart. That is not a technicality you can wave away, it is the actual epistemic situation, and it constrains what you're allowed to write.

This matters more for your project than for most. Three epochs on 10,000 images is a small budget; §1.6 walks through why augmentation may well have no visible payoff in that window. So a null is a realistic outcome for you, and when it arrives the correct sentence is:

- ✅ "We did not detect a difference in test accuracy between the two conditions under a 3-epoch training budget."
- ❌ "Augmentation does not improve accuracy."

The second claims something your data cannot carry, it's a claim about augmentation in general, made from one short run. The first is honest, precise, and has the pleasant side effect of pointing straight at your limitations and next-steps sections, both of which are graded.

### 1B.6 The five mistakes beginners make

| # | Mistake | Wrong version | Why it fails | Fix |
|---|---|---|---|---|
| **a** | H₁ written as a question | "Does augmentation improve test accuracy?" | A hypothesis must be assertable, and therefore refutable. A question can't be false, so there's nothing to test. | "Augmentation improves test accuracy relative to the unaugmented baseline." |
| **b** | Claim about the sample, not the population | "The augmented model will score higher on these 3,000 test images." | That's settled by looking. There's no inference in it, you'd have made a prediction about a spreadsheet, not about the world. | Make it about the procedure: "A CNN trained with augmentation achieves higher test accuracy than an identical CNN trained without it." |
| **c** | Unfalsifiable or vague wording | "Augmentation will affect performance somehow." | No possible result contradicts it. Even identical numbers can be spun as "an effect." A claim that can't lose can't win either. | Name the metric *and* the direction: "…achieves higher test accuracy…" |
| **d** | Compound hypothesis | "Augmentation improves accuracy **and** reduces the generalization gap." | Two claims smuggled into one, so no single result cleanly tests it. If accuracy falls but the gap narrows, is the hypothesis true? "Half true" is not an available verdict. | Split it into two complete pairs, see 1B.7. |
| **e** | Deciding direction after the fact (**HARKing**) | Running both models, seeing the augmented one score lower, then writing "we hypothesised augmentation would reduce accuracy at low epoch counts." | It guarantees you're right, which means the test carried no information at all. Predicting the outcome you already observed is not a prediction. | Commit in writing first, put your hypotheses in a markdown cell *above* the training cells, before you execute them. The notebook itself becomes your timestamp. |

Mistake (b) is the subtlest and worth one extra beat. The population you're generalising to isn't "all images ever", it's something like *CIFAR-10-style 32×32 natural images, classified by a small CNN under this training budget.* Being explicit about that keeps you honest without making your claim so narrow it says nothing.

### 1B.7 Your direct question: **pairs**, not one null with two alternates

You asked whether to write one H₀ with two H₁s hanging off it. The answer is no, you write **two complete pairs**.

The reason is mechanical rather than stylistic. A null hypothesis is always a statement about *one specific quantity*: "for this quantity, nothing is happening." Test accuracy and the generalization gap are different quantities, computed from different data, meaning different things. "Nothing is happening to accuracy" and "nothing is happening to the gap" are two distinct claims, and each alternative needs its own matching statement of what boring would look like. A shared null would leave one of your alternatives with no defined null to be tested against.

So here are your two pairs, written out:

| Pair | Outcome variable | H₀ | H₁ |
|---|---|---|---|
| **Pair 1** | Test accuracy | **H₀₁:** acc_aug ≤ acc_base | **H₁₁:** acc_aug > acc_base |
| **Pair 2** | Generalization gap (train acc − validation acc) | **H₀₂:** gap_aug ≥ gap_base | **H₁₂:** gap_aug < gap_base |

**Watch the sign, it flips between the two pairs.** This trips people up constantly, usually because they copy the sign down from the row above. For accuracy, bigger is better, so "augmentation wins" is `>`. For the gap, *smaller* is better, a narrow gap means the model does about as well on data it has never seen as on data it trained on, so "augmentation wins" is `<`. Before writing any inequality, ask yourself: **which direction is good for this particular metric?** Then write the sign from that answer, not from habit.

(This also means the ✍️ block at the end of Module 1 wants two lines under H₀ and two under H₁. Fill it in as pairs.)

One piece of formal housekeeping worth a sentence in your writeup: testing more than one hypothesis on the same experiment inflates the **family-wise error rate**, run enough tests and something will look impressive by chance alone. The standard correction is **Bonferroni**: divide your alpha by the number of tests, so with two tests you'd use 0.05 / 2 = 0.025 for each. With only two tests this is a minor adjustment, but naming it shows you understand that multiple comparisons are a real statistical problem and not just extra rows in a results table.

⚠️ **And now the honest caveat.** You run **one model per condition**. One run gives you no variance estimate, and with no variance estimate there is no test statistic, no p-value, and no alpha to correct in the first place. So frame the pairs above formally, the writeup asks for it, and the discipline of writing them properly is most of the value, but when you report, be clear that you can describe the **observed direction** of each comparison, not test it. The fix is your next step and it's cheap: repeat both conditions across 5–10 seeds and compare the resulting distributions. That's the run where all the language above becomes literally applicable instead of aspirational.

### 1B.8 A menu of candidate hypotheses for this project

Every one of these comes out of the same single pair of runs, one changed condition, several readings of it. Choose from the menu; don't try to claim all five.

| Label | Hypothesis (directional statement) | Why it's plausible | Risk at 3 epochs |
|---|---|---|---|
| **H-A** | Augmented **test accuracy** > baseline test accuracy | Augmentation is a standard regularizer, and better test accuracy is its advertised benefit. | **HIGH.** At 3 epochs the baseline may not have overfit yet, so there's no overfitting for augmentation to fix, and augmentation slows early learning. It can genuinely finish behind. |
| **H-B** | Augmented **generalization gap** < baseline gap | Reducing memorisation *is* augmentation's mechanism. This measures the thing itself rather than a downstream side effect. | **LOW–MODERATE.** The most likely of the set to hold, but read 1B.9 before you rely on it, there's a measurement subtlety. |
| **H-C** | Augmented **training accuracy** < baseline training accuracy | Nearly certain. The augmented model faces a harder, noisier task every epoch, because it effectively never sees the same image twice. | **Very unlikely to fail**, which is precisely why it belongs as a **manipulation check**, not a headline finding (1B.10). |
| **H-D** | Augmentation improves **macro recall more for visually confusable classes** (cat/dog, automobile/truck) than for distinctive ones (ship, frog) | Flip and small rotation add pose invariance, which should matter most where classes are separated by fine shape detail rather than by colour or context. | **HIGH, noise.** 3,000 test images across 10 balanced classes is roughly **300 per class**, so a few points of per-class recall is a handful of images. Genuinely interesting, and it makes real use of the required confusion matrix, but don't hang your conclusion on it. |
| **H-E** | **No detectable difference** between the conditions | The honest H₀ outcome, and a real possibility at this budget. | **Not a risk, a legitimate result.** The brief states that grading emphasises sound reasoning and interpretation over model performance. A well-argued null beats an overclaimed win. |

### 1B.9 ⚠️ The caveat you must know about H-B: your augmentation is a *layer*

This one is subtle, it's specific to your notebook, and getting it right is a real differentiator.

Your augmentation isn't applied to the dataset before training. It's a stack of Keras preprocessing layers, `RandomFlip("horizontal")`, `RandomRotation(0.05)`, `RandomZoom(0.10)`, sitting at the top of the augmented model, ahead of the first Conv2D. Layers of that kind behave differently depending on the mode they're called in:

| Mode | When it happens | Augmentation layers |
|---|---|---|
| **Training** | inside `model.fit()` | **Active**, every image is randomly flipped, rotated, zoomed |
| **Inference** | `model.evaluate()`, `model.predict()`, and the per-epoch validation pass | **Inactive**, images pass straight through, untouched |

Follow the consequence through, because it lands directly on H-B. The training accuracy Keras prints for the augmented model is measured on **augmented, that is, harder, images**. Its validation and test accuracy are measured on **clean images**. The baseline model sees clean images throughout, in both modes.

So `train_acc − val_acc` is **not an apples-to-apples comparison between your two models.** The augmented model's gap will look smaller partly because its training number was pushed down by a harder task, that's a **measurement artifact**, and not purely because it generalizes better. If you report the gap straight off the Keras output and claim H-B held, a sharp examiner can take the finding apart.

The clean fix costs you one line. After training, evaluate the augmented model on the *unaugmented* training set:

```python
train_loss, train_acc = model_aug.evaluate(x_train, y_train, verbose=0)
```

`evaluate()` runs in inference mode, so the augmentation layers are skipped and you get the augmented model's accuracy on the same clean images the baseline was scored on. Compute the gap from *that* number. Do it for **both** models, so the comparison stays symmetric, and as a bonus, this also sidesteps a second smaller distortion: the training accuracy `fit()` reports is a running average taken across the epoch while the weights were still changing, whereas `evaluate()` is one clean pass with the final weights. A post-hoc evaluation is the fairer number for either model.

Raise this in your presentation. Noticing that your two models' training accuracies were measured on different things, and then fixing it, is exactly the kind of methodological care that separates a top mark from a merely competent one.

### 1B.10 What a manipulation check is

A term borrowed from experimental design generally, and a useful one to have. A **manipulation check** verifies that your intervention actually did something, kept entirely separate from the question of whether it *helped*. In a psychology experiment where participants are meant to be made anxious, the manipulation check is asking them afterwards whether they felt anxious. If they didn't, then a null result on the main outcome tells you nothing about anxiety, because you never induced any in the first place.

H-C is your manipulation check. If the augmented model's training accuracy did *not* drop relative to the baseline, that's a signal the augmentation layers weren't doing what you assume, wrong mode, layers left out of the model, a training path that bypassed them. You'd debug that before interpreting anything else, because every other number becomes uninterpretable if the intervention never fired.

Report it as a **diagnostic, not a finding.** "Training accuracy fell from X to Y, confirming the augmentation was applied as intended" is a sentence for your methods section. It is *not* evidence that augmentation works, taken alone, lower training accuracy is also what a broken model looks like. Its value is that it licenses everything else you go on to say.

### 1B.11 Recommendation

| Role | Hypothesis | Why |
|---|---|---|
| **Primary** | **H-B**, generalization gap | Measures augmentation's actual mechanism; most likely to hold at 3 epochs; computed with the 1B.9 correction applied |
| **Secondary** | **H-A**, test accuracy | The headline metric every reader expects, and the most direct match to your assigned comparison |
| **Manipulation check** | **H-C**, training accuracy | Confirms the intervention fired; a methods-section diagnostic, not a result |

One experiment, one changed condition, three defensible readings, and a story that survives whichever way the numbers fall.

If test accuracy improves, H-A carries the headline and H-B explains *why* it improved. If accuracy doesn't move but the gap narrows, you report that the intervention did exactly what it was designed to do and the 3-epoch budget was too short for that to convert into test-set gains, which is a more interesting finding than a lucky win, and it's yours to explain. And if nothing moves at all, H-C still demonstrates that the experiment ran correctly, and you write up a clean, honest null with a concrete next step attached.

There is no outcome here that leaves you with nothing to say. That's the point of designing hypotheses before you run, rather than after.

---

## Module 1C: Your run protocol: confirmatory vs. exploratory ✅

You described your plan back to me. Most of it is sound, one part of it will quietly cost you marks if it survives into the writeup, and two smaller things are framed backwards. This module is the order you actually execute in.

### 1C.1 What you got right

Baseline first is correct, and it matches the brief's own step order, you cannot describe a change without something to change *from*. Recording the settings and metrics at baseline is correct, and it's what makes the comparison auditable rather than anecdotal. And grounding your augmentation choice in a published paper isn't garnish: the instructor rubric allocates **15 points to "Research foundation."** Three good instincts. The corrections below are about sequencing, not about substance.

### 1C.2 The one real error: "we will keep adding hypothesis"

This is the part to fix. Adding hypotheses as you go means some of them will be written *after* you've seen the augmented results, which is HARKing arriving through the back door (mistake **(e)**, §1B.6). You wouldn't be doing it deliberately. It just happens: a number appears, it's interesting, and writing it up as "we hypothesised…" feels like the natural sentence.

The distinction that resolves it cleanly:

|  | Confirmatory | Exploratory |
|---|---|---|
| **When decided** | Before seeing the augmented result | After |
| **What it can do** | **Test** a claim | **Generate** a claim, for someone else to test |
| **How to report it** | "We hypothesised X and found…" | "We also observed X; this was not predicted in advance and would need a fresh experiment to test" |

Read the middle row carefully, because it carries the whole idea. Exploratory analysis is **not forbidden and not second-class**. Noticing that the augmented model's confusion matrix cleaned up cat/dog while making ship/airplane worse is a genuinely valuable observation, and you should report it. What is forbidden is *relabelling* it as confirmatory after the fact, presenting a claim you formed **from** the data as one you tested **against** the data. That's circular, because the same numbers can't both invent the claim and confirm it.

So the working rule: **the list of hypotheses is locked before the augmented run; the list of observations and next steps keeps growing forever.** Both are legitimate. They simply get different sentences in the writeup, and a reader must be able to tell at a glance which is which.

The mechanic costs you nothing. Write H-A, H-B and H-C into a markdown cell positioned **above** the training cells, before you execute them. The notebook's own top-to-bottom order then becomes your timestamp, a grader scrolling through can see the commitment preceded the result. This is a poor man's pre-registration, and it is the cheapest credibility you will ever buy.

### 1C.3 Three hypotheses is not three experiments

You wrote "experiment the three things." H-A, H-B and H-C are **three readings of one experiment**, not three experiments.

You still change exactly **one** condition, augmentation off vs. on, and you still run exactly **two** models in total. All three hypotheses are computed from that same pair of runs: H-A from test accuracy, H-B from the train−validation gap, H-C from training accuracy. No extra training, no extra runs.

The temptation worth naming, because it's the one people fall into: running **conservative / moderate / stronger augmentation as three variants**. Don't. That is three interventions rather than one, it changes more than one thing, and the brief is emphatic about varying a single condition. It would also destroy the clean attribution that makes this an experiment at all (the controlled-variable list in §1.2), with one baseline and three variants, a difference could come from the intervention *or* from the intensity, and at n=1 you cannot separate them.

Comparing augmentation intensities is an excellent **next step** to propose, and proposing it shows you understand what your design can and can't answer. Just don't run it as this experiment.

### 1C.4 The paper comes first, not as back-up

You called the paper "a back-up and citation." Flip that ordering, because it's doing real work in the wrong direction.

A source cited *after* a decision is decoration, and graders can spot it instantly, the giveaway is a citation that would sit equally comfortably next to any of the choices you might have made. A source cited *before* the decision earns its place: it is the reason the decision came out one way rather than another.

You have a live, undecided choice sitting in front of you right now:

| Option | Transforms |
|---|---|
| **Conservative** | horizontal flip only |
| **Moderate** | flip + `RandomRotation(0.05)` + `RandomZoom(0.10)`, the notebook default |
| **Stronger** | flip + `RandomRotation(0.10)` + `RandomZoom(0.15)` |

That is precisely the decision your method source should settle. The literature on augmentation for low-resolution images speaks directly to it, recall from §1.5 that your real tradeoff at 32×32 is **added invariance vs. destroyed signal**, and that is a question the field has actual evidence about. Read first, choose second, and your justification section writes itself instead of being reverse-engineered.

Notice that this is structurally the same error as HARKing, one level up: justification assembled after the fact rather than driving the decision. The brief agrees with me here, it puts research at **step 2**, before the baseline even runs.

### 1C.5 Two things missing from your plan

**(a) Capture the clean training accuracy at baseline, not just later.** When you run the baseline, also run:

```python
train_loss, train_acc = model_base.evaluate(x_train, y_train, verbose=0)
```

That's the §1B.9 correction. If you skip it now, you will have to re-run the baseline later to get a symmetric generalization gap, because the training number `fit()` printed isn't comparable to the augmented model's. One line now; a repeated run later.

**(b) Fix the reproducibility asymmetry *before* the augmented run.** Module 1 flagged this in §1.2 and deferred it to Module 7, but you need it sooner than that: the baseline cell calls `tf.keras.backend.clear_session()` and `tf.random.set_seed(SEED)` before building, and the augmented cell does not. Add those same two lines above the augmented model so both start from identical random initialisation. Do it before the augmented run, doing it afterwards means running everything again.

Both items are cheap now and irritating later, which is the whole reason they get their own subsection.

### 1C.6 Read the baseline curves before finalising which hypothesis leads

This is a legitimate move, and I want you to make it deliberately, because at first glance it looks like exactly the thing I just told you not to do.

It isn't, and the reason is precise. **The baseline is your control condition.** At that point you have not seen the augmented result, and, more importantly, you have not seen the *difference between conditions*, which is where the effect lives and where every one of your hypotheses stakes its claim. Using control-condition behaviour to calibrate expectations is standard practice; it's the same logic as running a pilot. The line that must not be crossed is committing hypotheses after seeing the **augmented** number.

The decision rule:

| What the baseline curves show by epoch 3 | What it means | Which hypothesis leads |
|---|---|---|
| Training accuracy pulling **clearly above** validation accuracy | The baseline is overfitting, so augmentation has a real problem to solve | **H-A** (test accuracy) is plausible and can lead |
| Training and validation still **tracking together** | Little overfitting yet, so there's nothing for augmentation to fix in this budget | Lead with **H-B** (generalization gap); frame H-A as the secondary that probably won't move |

One honest footnote, because you should know where the strict line sits. The strictest version of pre-registration commits *everything* upfront, including which hypothesis leads, and looks at nothing until all data is in. By that standard, even this is a small liberty. For your writeup, the line that actually matters is **committing before the augmented run**, and what makes either version defensible is saying plainly which one you did. Put it in your methods: *"Hypotheses were committed before the augmented run; the primary hypothesis was selected after inspecting baseline training curves."* A reader who knows exactly what you did has nothing left to object to.

### 1C.7 The corrected sequence

| Step | Action | What to record / produce |
|---|---|---|
| **1** | Find your three sources, **method source first** | Method source on CNNs/augmentation; dataset/application source; limitations-validity source. APA 7. |
| **2** | Choose augmentation intensity, justified by that method source | Conservative / moderate / stronger, plus the sentence saying why, written before anything runs |
| **3** | Write H-A, H-B, H-C into a markdown cell **above** the training cells | Each as a complete H₀/H₁ pair (§1B.7). This cell is your timestamp. |
| **4** | Run the baseline **exactly as shipped** | Test accuracy, macro precision, macro recall; train/val curves per epoch; the clean `evaluate(x_train, y_train)` number; and every setting, seed 42, 10,000 / 2,000 / 3,000 split, 32×32×3 RGB, 10 balanced classes, 3 epochs, batch 64, Adam, sparse categorical crossentropy |
| **5** | Inspect the baseline curves; confirm which hypothesis leads | One line in your notes recording the call and the reason (§1C.6) |
| **6** | Fix the `clear_session()` / `set_seed()` asymmetry | Both cells now identical in setup; note the fix for your reproducibility section |
| **7** | Run the augmented model, **once** | The same metric set as step 4, including the clean training-accuracy number |
| **8** | Build the comparison | Side-by-side metric table, confusion matrix, train/val curves for both models |
| **9** | Interpret | Finding, implication, limitation, next step, with confirmatory and exploratory claims **visibly separate** |

Step 9's last clause is what carries §1C.2 into the actual document, and it's easier than it sounds. Two headed paragraphs will do it: **"Hypothesis outcomes"** for what you committed to and what happened, then **"Additional observations"** for everything you noticed afterwards. Nobody can then mistake one for the other, which is the entire objective.

---

## Module 1D: Running the baseline: what to capture ✅

You're about to run. This module is the short list of things that are cheap to capture while the run is happening and expensive to reconstruct afterwards.

### 1D.1 Where to run it: Google Colab

Run it in Colab. Our Cowork cloud container can't execute this notebook. TensorFlow isn't installable there because PyPI isn't reachable from that sandbox, and the CIFAR-10 download host `cs.toronto.edu` is blocked, so `keras.datasets.cifar10.load_data()` would fail even if TensorFlow were present. This costs you nothing, because the starter notebook's own instructions say to use Colab with a GPU runtime, and the **graded artifact is the notebook with its outputs retained**. It has to be your run, in your Colab, saved with the outputs in place, a preview elsewhere would never have counted for marks anyway.

### 1D.2 Running the baseline first is consistent with the protocol

You asked whether you can run the baseline before finalising your sources and hypotheses. Yes, and the reasoning is worth having rather than just the permission.

The baseline is your **control condition**. It does not depend on the augmentation choice in any way: not on the intensity you pick, not on which paper you cite, not on which hypothesis ends up leading. None of those decisions enter the baseline at all, so nothing about the baseline run can be contaminated by a decision you haven't made yet. The line that matters, restated from §1C.2 and §1C.6, is that hypotheses are locked before the **augmented** run, not before the baseline.

So reorder §1C.7 for yourself like this:

1. **Run the baseline** (with the §1D.4 diagnostics, twice, see §1D.3)
2. Find your three sources, method source first
3. Choose augmentation intensity, justified by that source
4. Write H-A / H-B / H-C into a markdown cell above the training cells
5. Inspect the baseline curves; confirm which hypothesis leads
6. Fix the `clear_session()` / `set_seed()` asymmetry
7. Run the augmented model, once
8. Build the comparison
9. Interpret

The §1C.7 ordering is a **default, not a constraint**. The only hard requirement inside it is that steps 2–4 are finished before the augmented run. Moving the baseline to the front costs you nothing and gets you real numbers to think with sooner.

### 1D.3 Run it twice before you trust anything

Short subsection, disproportionate value.

Colab GPU training is **not deterministic**, even with `tf.random.set_seed(42)`. cuDNN selects kernels adaptively, and GPU reductions use atomics whose accumulation order isn't fixed, so two identical runs can land on slightly different numbers. Seeding removes the randomness *you* control (shuffling, weight initialisation); it does not remove the randomness the hardware introduces underneath you.

So run the baseline cell **twice** and write down both test accuracies.

The spread between those two numbers is your **practical noise floor**. Any baseline-vs-augmented difference smaller than that spread cannot be read as an effect, it's inside the range the same configuration produces on its own. That converts a vague limitation sentence into a specific one: *"two identical baseline runs differed by X percentage points; we therefore treat differences below that magnitude as indistinguishable from run-to-run variation."*

It doesn't give you a significance test, §1B.7's n=1 problem is still the n=1 problem, but it gives you a defensible floor to compare against, it costs about a minute of GPU time, and it is the single cheapest credibility upgrade available to you. Do it.

### 1D.4 The diagnostics cell

The starter notebook leaves three things uncollected that you will want later. Adding them is **measurement only**, it doesn't touch the model, the data, or the training loop, so your baseline is still "exactly as shipped." You're reading more off the same run, not changing the run.

| # | What's missing | Why you need it |
|---|---|---|
| **(a)** | No clean-image train/validation evaluation | Without it there is no fair generalization gap (§1B.9) |
| **(b)** | `classification_report` is imported in the first cell and never called | No per-class precision / recall / F1, exactly what H-D needs, and exactly what the macro averages smooth away |
| **(c)** | The confusion matrix is plotted only for the **augmented** model (`aug_pred`) | There is no baseline confusion matrix to compare against |

Paste this immediately after the baseline training cell:

```python
# ============================================================
# BASELINE DIAGNOSTICS, measurement only.
# Does not change the model, the data, or the training.
# Run immediately after the baseline training cell.
# ============================================================

# 1) Clean-image accuracy with FINAL weights, in inference mode.
#    This is the fair number for the generalization gap (see 1B.9).
base_train_loss, base_train_acc = baseline_model.evaluate(x_train, y_train, verbose=0)
base_val_loss,   base_val_acc   = baseline_model.evaluate(x_val,   y_val,   verbose=0)
base_gap = base_train_acc - base_val_acc

print(f'Baseline clean train accuracy : {base_train_acc:.4f}')
print(f'Baseline validation accuracy  : {base_val_acc:.4f}')
print(f'Baseline generalization gap   : {base_gap:.4f}')

# 2) Per-class precision / recall / F1.
#    classification_report is imported in cell 1 but never used.
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

# 4) Settings record, copy this straight into the evidence brief.
print('\nSETTINGS RECORD')
print(f'seed=42  train={x_train.shape[0]}  val={x_val.shape[0]}  test={x_test.shape[0]}')
print(f'image shape={x_train.shape[1:]}  classes={len(class_names)}')
print(f'epochs={len(baseline_history.history["accuracy"])}  batch_size=64  optimizer=adam')
print(f'trainable params={baseline_model.count_params():,}')
```

Every variable this uses, `baseline_model`, `baseline_pred`, `baseline_history`, `x_train`, `y_train`, `x_val`, `y_val`, `y_test`, `class_names`, already exists in the starter notebook, and `classification_report`, `confusion_matrix`, `plt` and `sns` are all imported in the first cell. So it needs no new imports and no installs; paste and run.

One deliberate detail worth noticing: `annot=True, fmt='d'` in the heatmap call. The notebook's existing confusion-matrix call omits both, which produces an unlabelled colour grid, it looks fine and a reader cannot extract a single number from it. The rubric asks for *readable* visuals, so print the counts.

### 1D.5 What to look at when it finishes

| Number | Where it comes from | What it tells you | What to do with it |
|---|---|---|---|
| **Test accuracy** | the notebook's test-set evaluation | Headline performance on data neither you nor the model has tuned against | Goes straight into the comparison table, this is H-A's quantity |
| **Macro precision / macro recall** | `classification_report`, or your metrics cell | Required reporting; each class weighted equally regardless of how it performs | Compare them against accuracy, if they diverge noticeably, some class is being handled badly, and the confusion matrix will tell you which |
| **Generalization gap** | diagnostics cell, `base_train_acc − base_val_acc` | How much the model is memorising versus generalising | Decides which hypothesis leads (§1C.6), this is H-B's quantity |
| **Train vs. validation curves** | `baseline_history` | The epoch-by-epoch visual version of the gap | Read the shape: lines diverging = overfitting has started; lines still parallel = it hasn't |
| **Per-class report** | diagnostics cell, part (b) | Which classes are weakest, most likely cat, dog and bird | Seeds H-D, and gives you something specific to say in interpretation instead of a single averaged number |
| **Confusion matrix** | diagnostics cell, part (c) | *Which pairs* get confused, and whether the confusion runs both ways | Symmetric confusion (cat↔dog in both directions) suggests genuinely similar classes; one-directional confusion suggests a bias toward predicting one class |
| **Run-to-run spread** | your two baseline runs (§1D.3) | Your noise floor | Any later baseline-vs-augmented difference smaller than this is not an effect; cite the number in limitations |

**The decision rule, once more:** if training accuracy is pulling clearly above validation accuracy by epoch 3, the baseline is overfitting and **H-A can lead**; if the two are still tracking together, there is little overfitting yet, so **H-B leads** and H-A probably won't move.

### 1D.6 Record this now

Copy these out while the run is fresh. Reconstructing any of it means running again:

- [ ] The `SETTINGS RECORD` block, verbatim
- [ ] Test accuracy, macro precision, macro recall
- [ ] Clean train accuracy, validation accuracy, and the gap
- [ ] **Both** runs' test accuracies (§1D.3)
- [ ] The per-class report output
- [ ] Saved copies or screenshots of both figures, the training curves and the baseline confusion matrix
- [ ] The Colab notebook itself, saved **with outputs retained**

Colab sessions expire and clear their state without warning. The saved notebook with its outputs is simultaneously your graded artifact and your only backup, so save it before you close the tab.

---

## Module R: What our own results taught us ✅

Every module before this one taught you something *before* you ran anything. This one runs the other way round: the experiments are finished, and the numbers turned out to be a better statistics teacher than I am. Almost everything below is a correction to an intuition that looked perfectly safe in Module 1 and did not survive contact with data.

Read it twice. This is the module your interpretation and limitations sections are made of.

### R.1 What actually happened

**Baseline, five seeds**, the data is fixed, only the weight-initialisation / training seed varies:

| Quantity | Value |
|---|---|
| Test accuracy | **51.35%**, SD **1.62 pp** |
| Clean train accuracy | 56.18% |
| Clean validation accuracy | 51.13% |
| Generalization gap | **5.05 pp** |
| Noise floor (max − min test accuracy) | **3.77 pp** |
| Wall clock | ~8 s per run, on CPU |

Two things to notice straight away. First, ~8 seconds a run means the n=1 limitation that §1.8 and §1B.7 called your single biggest weakness was never a real constraint. The multi-seed re-run those sections proposed as a *next step* cost about two minutes, so it stopped being a next step and became the experiment.

Second, and more important: **the baseline was underfitting, not overfitting.** Validation loss fell monotonically through epoch 3 in all five runs. A model that has started to overfit shows validation loss turning back *up* while training loss keeps falling, and nothing resembling that appears anywhere in the curves. So §1C.6's decision rule fires on its second row, train and validation still tracking together, and **H-B leads**, exactly as §1B.11 recommended. §1.6 called this trap in advance; it is satisfying to have predicted it, and you should say so in the writeup, because a prediction on record is worth more than the same sentence written afterwards.

One curiosity worth understanding properly, because it looks like a bug and isn't: for the first two epochs, **validation accuracy sat above training accuracy.** That does not mean the model did better on data it had never seen than on data it trained on. The training number `fit()` prints is a **running average across the whole epoch**, accumulated while the weights were still poor at the start and improving all the way through. The validation number is computed **once, at the end of the epoch, with the final weights**. You are comparing an average over a moving target against a single snapshot taken at that target's best moment. Early in training, when weights improve fast, the snapshot wins. This is the §1B.9 measurement asymmetry wearing a second costume, and it is one more argument for computing every gap from `evaluate()` on clean data rather than reading `fit()`'s output.

**The paired experiment, ten seeds, both conditions run at every seed:**

| Metric | Baseline | Augmented |
|---|---|---|
| Test accuracy | 0.5069 ± 0.0221 | 0.4791 ± 0.0159 |
| Clean train accuracy | 0.5428 | 0.4993 |
| Clean validation accuracy | 0.4986 | 0.4752 |
| Generalization gap | 0.0442 | 0.0241 |

**The two pre-registered tests:**

|  | Accuracy (H-A, non-directional) | Gap (H-B, directional) |
|---|---|---|
| Mean paired difference | **−2.78 pp** | **−2.01 pp** |
| 95% CI | [−4.54, −1.02] | [−3.10, −0.92] |
| Paired *t*(9) | −3.573 | −4.164 |
| *p* | 0.006 (two-tailed) | 0.0012 (one-tailed) |
| Wilcoxon *p* | 0.027 | 0.0029 |
| Cohen's *d* | −1.13 | −1.32 |
| Seeds in the negative direction | 7 of 10 | 9 of 10 |
| Verdict | **H₀ rejected** | **H₀ rejected** |

Both pre-registered nulls fell. Augmentation **changed** test accuracy, downward, and it **narrowed** the generalization gap, which is the direction H-B predicted. The manipulation check (§1B.10) confirms the intervention actually fired: `fit()`'s final training accuracy went 0.5120 → 0.4744, a drop of 3.76 pp. Every number below the headline is now interpretable, because that one is.

What follows is seven lessons. Each is something I would have got wrong if all I had to reason from was Module 1.

### R.2 A gap is not overfitting, and a shrinking gap is not automatically good

H-B held. The gap narrowed from 0.0442 to 0.0241, *p* = 0.0012, 9 of 10 seeds moving the predicted way. If you stopped there you would report a clean confirmatory win.

Do not stop there. Decompose it:

| Term | Baseline | Augmented | Change |
|---|---|---|---|
| Clean train accuracy | 0.5428 | 0.4993 | **−4.35 pp** |
| Clean validation accuracy | 0.4986 | 0.4752 | **−2.34 pp** |
| **Gap** (train − val) | 0.0442 | 0.0241 | **−2.01 pp** |

**Both terms fell. Training fell nearly twice as far.** The gap narrowed not because the model closed the distance between what it memorised and what it generalised, but because the whole model slid downwards and the top slid faster. Nothing generalised better. The augmented model is worse at the training set *and* worse at the validation set; it is merely worse at the training set by more.

Sit with how close you came to the opposite conclusion. "Augmentation significantly reduced the generalization gap (−2.01 pp, *p* = 0.0012)" is a **true sentence**. It is also, on its own, badly misleading, every reader will take it to mean the model generalises better, because that is what a narrowing gap normally signals. Reported alone, a true statistic would have carried a false finding.

So here is the general rule, and it is the most portable thing in this whole document:

> **Never report a difference without reporting both of its terms.**

A gap, a delta, an improvement, a ratio, a percentage change, every one of them is a single number standing in for two, and the two can move in ways the one cannot express. A gap of 2 pp describes a model at 95/93 and a model at 50/48 equally well. The difference tells you about *distance*; it is silent about *level*. Always print the level next to it.

Why did this happen? Because the regime was wrong for the intervention. Augmentation is regularization, and regularization helps a model that is fitting *too much*. Three epochs on 10,000 images leaves this model fitting too little (R.1), so augmentation had no overfitting to remove and simply made an already-hard task harder. Lin et al. (2024) frame augmentation as an implicit regularizer whose benefit depends on the fitting regime, which is precisely the distinction your run illustrates. §1.6 predicted the mechanism; your data supplies the evidence.

**The same rule, one level down.** Look at the per-class recalls:

| Class | Baseline recall | Augmented recall | Change |
|---|---|---|---|
| bird | 0.252 | 0.172 | −8.0 pp (**≈32% of the recall it had**) |
| deer | 0.380 | 0.280 | −10.0 pp |
| airplane | 0.646 | 0.549 | −9.7 pp |
| automobile | 0.643 | **0.705** | **+6.2 pp** |

Automobile is the trap. Recall rose, and if you reported per-class recall alone you would have a tidy little sub-finding: "augmentation improved automobile recognition." But automobile's **precision fell from 0.624 to 0.545** over the same comparison. Both facts together say something quite different: the augmented model simply predicts "automobile" more often. It therefore catches more of the real automobiles (recall up) and is wrong more often when it says automobile (precision down). That is not improved recognition; it is a shifted decision boundary. Recall alone would have invented a success story out of a redistribution.

And note bird, which teaches the companion habit: **always ask "relative to what."** An 8 pp drop sounds like the smallest number in the table. From a base of 25.2% it is roughly a third of everything the model had. Absolute and relative changes tell different stories and you need both, which is the same rule again, since a percentage change is also a difference hiding its terms.

### R.3 Effect size must be read against measured noise, not against zero

Here is the result that looks impossible. The accuracy effect is **2.78 pp**. Your measured noise floor is **3.77 pp**. The effect is *smaller than the noise*, and it is significant at *p* = 0.006.

That is not a contradiction, and understanding why is worth more than the finding itself.

§1D.3 taught you a rule: any baseline-vs-augmented difference smaller than the run-to-run spread cannot be read as an effect. That rule is **correct for a single comparison**, one baseline run against one augmented run. With n=1 per condition you genuinely cannot distinguish a 2.78 pp difference from the wobble the same configuration produces on its own. It is the **wrong** rule for ten paired comparisons, and the reason is the difference between a spread and the precision of a mean.

| Quantity | What it describes | Does it shrink with more runs? |
|---|---|---|
| **Noise floor / SD of a single run** | How much one run's outcome bounces around | **No.** It is a property of the procedure. Running more seeds measures it better; it does not make it smaller. |
| **Standard error of the mean difference** | How precisely you know the *average* effect | **Yes**, it falls roughly as 1/√n. Ten paired runs know the mean about three times more precisely than one. |

The test is not asking "is this one difference bigger than the wobble?" It is asking "**is the average of ten differences far enough from zero that random wobble is an implausible explanation?**" And the answer came from **consistency, not magnitude**: 7 of 10 seeds negative on accuracy, 9 of 10 on the gap. The Wilcoxon signed-rank test makes this vivid, it throws away the magnitudes entirely and looks only at signs and ranks, and it still rejects, at *p* = 0.027 and *p* = 0.0029. A test that cannot see how big your effect is still found it, purely from how reliably it repeated.

The coin analogy is exact. One flip tells you nothing about a coin's bias, because a single flip's "spread" covers the entire range of outcomes. Twenty flips landing 15 heads is persuasive, even though each individual flip is still completely uninformative. Repetition does not shrink the noise in any one measurement; it shrinks your uncertainty about the centre.

The corollary matters just as much, and it is the one that protects you from other people's results: **a large one-run difference that flips sign across seeds is not an effect.** Magnitude without consistency is nothing. Consistency without magnitude is a real, small effect.

Which brings the honest sentence. You have to report both facts, because each one bounds the other:

> The effect is statistically detectable and it is smaller than the run-to-run spread of a single configuration. A practitioner comparing these two setups on one run each could not reliably tell them apart; across ten paired runs the difference is systematic.

That sentence is more informative than either half, and it is what "without overclaiming" looks like in practice. Note also that Cohen's *d* = −1.13 is "large" by the usual convention, but *d* is standardised against the SD of the paired **differences**, not against accuracy itself, so quoting *d* alone commits the R.2 error all over again. Print *d*, the raw pp, and the noise floor together.

This is also a place to reach for the literature rather than argue from your own run alone: seed-only variation is a documented, non-negligible phenomenon on CIFAR-10-scale problems (Coakley & Gundersen, 2026), and leaning on a single seeded run is exactly the practice Åkesson et al. (2024) show to be unreliable. Your multi-seed design is the standard response to a known problem, not an improvisation, say that, because it converts a methods choice into a research-grounded one, which is where §1C.4's marks live.

### R.4 Why the non-directional hypothesis on accuracy was the right call

§1B.4 argued that directional hypotheses are stronger, more scientific and better-powered, and Module 1 told you to prefer them. H-A was nevertheless written **non-directional** (acc_aug ≠ acc_base). This is the case that shows why the weaker-looking choice was the right one.

Work through the three hypotheses you could have committed to:

| What you could have pre-registered | What the data did | What you would be allowed to report |
|---|---|---|
| **Directional, up**, acc_aug > acc_base (Module 1's default, §1.4) | Accuracy fell 2.78 pp | "**Failed to reject H₀.**" Nothing else. A clearly systematic effect, and your framework would have no sentence for it, the effect went through the tail you weren't looking at. |
| **Directional, down**, acc_aug < acc_base | Matched exactly | You would have been right, with better power. But this is the prediction that *looks* reverse-engineered, because §1.6 already had you expecting a decline. A reader cannot audit the order in which you wrote things; they can only judge whether the prediction looks suspiciously well-fitted to the outcome. |
| **Non-directional**, acc_aug ≠ acc_base (**what you did**) | Change detected | A significant **change**, *p* = 0.006 two-tailed, with the direction described afterwards as an observation. Clean hands. |

The middle row is the subtle one, so be clear about what is and isn't wrong with it. Predicting the decline would not have been HARKing, you had §1.6 on record before the augmented run, and that is a real timestamp. It would have been legitimate. But legitimacy you cannot *demonstrate* is worth less than legitimacy that needs no defending, and the non-directional version needs none: when you never claim a direction, describing the direction you observed cannot possibly be mistaken for having claimed it.

You paid a real price for that. Two-tailed tests split their sensitivity across both tails, so you had less power than a one-tailed test would have given you (§1B.4's table). You paid it and rejected anyway.

Now notice the design's real sophistication, which is that **the two hypotheses were not treated the same way**:

| Hypothesis | Form | Where the direction came from |
|---|---|---|
| **H-A** (accuracy) | Non-directional | Nowhere. §1.6 argued accuracy could genuinely go either way at 3 epochs, improvement from regularization, decline from a harder task. Real uncertainty. |
| **H-B** (gap) | Directional | The **mechanism** (§1.5). Reducing memorisation is what augmentation is *for*; a narrower gap is the definitional consequence of the intervention working. |

That asymmetry is not hedging. It is the design matching what you actually knew. **A directional hypothesis is defensible when a mechanism supplies the direction; a non-directional one is the honest choice when the direction is genuinely open.** Write that sentence into your methods, it turns what could look like an inconsistency into evidence that each hypothesis was reasoned separately.

### R.5 What a replication buys you: and why you must not swap to the stronger numbers

When the notebook was re-executed, the same configuration on the same ten seeds produced different numbers. Not a bug: TensorFlow's CPU kernels are not bit-deterministic across processes (R.6). So the re-run is not a reproduction of the committed numbers, it is a **second independent measurement**, which is to say a replication.

| Finding | Committed run | Replication |
|---|---|---|
| Accuracy difference | −2.78 pp, *t*(9) = −3.573, *p* = 0.006, *d* = −1.13, 7/10 seeds | **−3.75 pp**, 95% CI [−4.90, −2.60], *t*(9) = −7.367, *p* < 0.0001, *d* = −2.33, **10/10** seeds |
| Gap difference | −2.01 pp, *t*(9) = −4.164, *p* = 0.0012, 9/10 seeds | **−2.96 pp**, *t*(9) = −10.891, *p* < 0.0001, **10/10** seeds |

Both findings replicated, and more strongly on every dimension: larger effect, tighter interval, smaller *p*, unanimous seeds.

**What a replication actually buys you** is not a better *p*-value. It is a different kind of evidence. A single experiment's *p* answers "how surprising would this data be if nothing were happening?", a question asked entirely *inside* one dataset. A replication answers a question no *p*-value can reach: "does this come back when I do it again?" Your ten seeds could always have been ten unlucky draws. A second, independently-noisy execution landing the same way on both pre-registered metrics is worth far more than one execution with a smaller *p*, and it is the thing most student projects cannot offer at all.

**And now the discipline.** You must not report the replication's numbers as your headline. Not because they are wrong, they are as valid as the committed ones, but because **you saw both sets of *p*-values before choosing.** Selecting the run with the better statistics is selection on the outcome. It is the run-level version of §1B.6's mistake (e): instead of choosing the hypothesis after seeing the result, you would be choosing the *result* after seeing the results. Do it consistently and every *p*-value you publish is inflated by the choices you didn't disclose.

So the rule extends further than §1C.2 stated it:

> **Pre-registration binds the reported result, not just the hypothesis.** You commit to which run counts before you look at any of them.

Then do the thing that actually earns the credit: **say all of this out loud.** State in the writeup that a stronger replication exists, give its numbers in a robustness subsection, and state plainly that it was deliberately not substituted for the pre-registered run. A grader cannot see the analyses you discarded, nobody ever can, so disclosure is the *only* signal available that you had the opportunity and declined it. The fact that you could have swapped and did not is a claim about your process that no results table can carry, and it is the kind of sentence that separates work that is trustworthy from work that merely looks tidy.

### R.6 Seeds do not guarantee reproducibility

§1D.3 told you Colab GPU training is non-deterministic and blamed cuDNN's adaptive kernel selection and GPU atomics. That was half right, and the half it got wrong matters: you ran on **CPU**, and it was still not reproducible.

The measurement, comparing the two runs seed by seed under identical configuration, identical data and identical seeds:

| Statistic | Value |
|---|---|
| Mean per-seed difference in test accuracy | **1.57 pp** |
| Maximum per-seed difference | **3.63 pp** |

**The mechanism** is floating-point arithmetic, and it is worth knowing because it generalises to every framework you will ever use. Floating-point addition is **not associative**: in finite precision, (a + b) + c and a + (b + c) can give different answers, because each intermediate result is rounded. Training is built almost entirely out of large summations, over a batch, over a feature map, over channels, and those reductions are parallelised across threads. The order in which the partial sums are combined depends on thread scheduling, which depends on what the machine happened to be doing. Change the order, change the last bits of the result. Then feed that tiny difference into a gradient, into a weight update, into the next batch's gradients, and let it compound over three epochs.

So be precise about what a seed does:

| A seed fixes | A seed does not fix |
|---|---|
| Weight initialisation | The order floating-point reductions are combined in |
| Shuffling order | Thread scheduling |
| Which random flip/rotation/zoom each image gets | Library versions, BLAS backends, hardware |

Seeding removes the randomness *you* control. It does nothing about the randomness underneath you.

**The consequence for your noise floor is the real lesson.** The 3.77 pp figure was described in §1D.3 as run-to-run variability, and the analysis attributed all of it to seed variation. It is actually a sum of two distinct components:

| Component | What varies | How you measure it | Your value |
|---|---|---|---|
| **Between-seed** | Initialisation and data order | Different seeds, same process | The 5-seed spread, reported as 3.77 pp, but confounded with the row below |
| **Within-seed, across runs** | Floating-point reduction order across processes | The **same** seed, run twice in separate processes | Mean 1.57 pp, max 3.63 pp |

Look at those side by side. The maximum within-seed difference (3.63 pp) is almost the entire five-seed "noise floor" (3.77 pp). Whatever else is true, **3.77 pp was never a pure seed effect**, it always contained both, and the earlier framing that credited it all to seeds was wrong. Correct it in your limitations section; it costs you nothing and it is a genuinely sharp observation.

One more repair to §1D.3 while we are here. It proposed max − min as the noise floor, and your own data shows why a range is the worst available statistic for this job: **the baseline SD rose from 1.62 pp at 5 seeds to 2.21 pp at 10.** A range can only ever grow as you add runs, and it is determined entirely by two extreme observations, so a small-sample range systematically *understates* variability, exactly as predicted in §1.8's warning about small-sample thinking. Report the SD, which is stable and uses every observation, and give the range alongside it as a descriptive extra. Say which one you used for any comparison.

Practically: bit-reproducibility on this stack would require forcing deterministic operations and fixing thread counts, at a real cost in speed, and it still would not survive a library or hardware change. The honest position for a project like yours is not to promise bit-reproducibility at all. It is to publish **the seeds, the mean, the SD, and the code that regenerates them**, reproducibility of the *distribution* rather than of the digits. That is a stronger and more defensible claim, and it is what your artifacts already support.

### R.7 Why pairing helped less than we hoped

The design paired the two conditions by seed: at every seed, run baseline and augmented, then test the ten **differences**. The point of pairing is variance reduction. If both arms share the same seed-level luck, that luck cancels inside each difference, the differences are less variable than either arm on its own, and you get more power for free.

The arithmetic that governs it:

> SD(difference)² = SD_base² + SD_aug² − 2 · r · SD_base · SD_aug

where *r* is the correlation between the paired runs. When *r* is high the subtracted term is large and the differences are tight. When *r* = 0 there is nothing to subtract, and you land on the "as if independent" value, √(SD_base² + SD_aug²).

Put your numbers in. SD_base = 2.21 pp, SD_aug = 1.59 pp, so the independent expectation is √(2.21² + 1.59²) = **2.72 pp**. The SD of the differences you actually observed was **2.46 pp**. Solving back gives **r ≈ 0.19**.

That is almost no cancellation. The pairing bought you a little, and far less than the design assumed.

**Why?** Because "same seed" did not mean "same initial weights." The augmented model inserts three preprocessing layers ahead of the first Conv2D, and inserting them alters the layer-construction order and therefore the position in the random stream from which the convolution and dense kernels are drawn. Same seed, different draws, so the two arms never started from the same point, which is the one thing the pairing depended on. On top of that, the cross-process floating-point variation from R.6 enters both arms **independently** and cannot cancel by construction: it is noise the pairing has no mechanism to remove.

**The fix is cheap and specific**, which is what makes it a good next step rather than a vague one: build the model once, save its initial weights before any training (`get_weights()` / `save_weights()`), then load those identical weights into both arms at every seed. Then "paired" means paired on the thing that actually drives the variance, instead of paired on a seed value that only loosely determines it.

Now the interpretation, and be careful here because it cuts both ways. Weak pairing does **not** invalidate the test. A paired *t*-test is valid at any correlation, *r* only determines how much variance reduction you get, not whether the procedure is legitimate. What it means is that your test was running at closer to unpaired efficiency, with less power than the design promised. And you rejected both nulls anyway. A finding that survives a design operating below its intended efficiency is, if anything, more robust than one that needed the full design to squeak through. Say that, it is true, it is favourable, and you can only say it because you measured *r* instead of assuming it.

### R.8 Finding your own design flaw and reporting it is a strength

R.7 is a flaw in your own experiment. You found it yourself, after the results were in, and it makes your headline design look weaker than it did on paper. Report it anyway, and report it prominently.

This feels wrong to most students, so here is why it is right. A limitation you discovered yourself is **evidence that you understood your design well enough to interrogate it.** Nobody can find the flaw in R.7 without knowing what pairing is supposed to buy, computing what it actually bought, and being willing to look. A limitations section that lists only the generic caveats, small dataset, few epochs, future work should train longer, is indistinguishable from one written by somebody who never checked. The specific ones are the only ones that carry information about you.

The rubric asks you to interpret **without overclaiming**. That does not mean hedging everything, softening true statements, or refusing to commit to a finding. Under-claiming is its own failure: you rejected two pre-registered nulls and you should say so plainly, without apology. What it means is: **claim exactly what your evidence supports, and disclose everything that bounds it.** So state the findings, then state that the accuracy effect is smaller than the single-run noise floor (R.3); that the gap narrowed because both terms fell rather than because generalization improved (R.2); that the pairing achieved only *r* ≈ 0.19 (R.7); that seeding does not make the run bit-reproducible and the noise floor has two components (R.6); and that a stronger replication exists which you deliberately did not substitute (R.5).

Concretely, for the writeup:

| ✅ You can say | ❌ You cannot say |
|---|---|
| Under a 3-epoch budget on a 10,000-image CIFAR-10 subset, adding horizontal flip + rotation 0.05 + zoom 0.10 significantly reduced test accuracy (−2.78 pp, *p* = 0.006) and significantly narrowed the train−validation gap (−2.01 pp, *p* = 0.0012). | "Augmentation hurts CNNs." |
| The gap narrowed because training accuracy fell further than validation accuracy, both fell, so this is not evidence of improved generalization. | "Augmentation improved generalization." |
| The accuracy effect is significant across ten paired seeds and smaller than the 3.77 pp single-run noise floor. | "Augmentation costs about 3 points of accuracy." |
| A replication on the same seeds reproduced both findings more strongly; the pre-registered run is reported as the result. | Reporting the replication's numbers as the headline. |
| Bird recall fell by roughly a third of its baseline value, the largest per-class loss observed. | "Augmentation damages fine-grained classes", one experiment, one intensity, no sweep. |

Roberts et al. (2021) is the right citation for the shape of this argument: a systematic review in which study after study failed not because the models were bad but because the design and reporting around them did not support the claims made. Cite it where you say what your result does **not** license, not where you say what it does.

And the closing thought, which is really the thesis of every module so far. The most valuable sentence in your writeup will not be the one with the *p*-value in it. It will be the one where you explain why your significant gap reduction is not the good news it appears to be. Anyone can run a *t*-test, it is one line. Working out what your own significant result does not mean is the part that is genuinely hard, and it is what §1.8, §1B.5 and §1C.2 have been building toward since the first page.

---

## Module 2: The dataset and preprocessing ✅

Two rubric lines live here: **dataset** and **preprocessing**. Every choice in this module was made for you by the starter notebook, and that does not get you out of explaining them, "the starter code did it" is not an answer to "why is your data divided by 255?" It is also the module where the population your claims apply to gets defined, which is §1B.6's mistake (b) made concrete.

### 2.1 What CIFAR-10 is

60,000 colour images at 32×32 pixels, 10 mutually exclusive classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck), 6,000 per class, perfectly balanced. It ships pre-divided into 50,000 training and 10,000 test images, and those two pools are disjoint by construction. Standard benchmark, which is exactly why it's a defensible choice for a time-boxed project.

Get concrete about what a single image *is*, because everything in Module 3 depends on it:

| Property | Value | Why it matters later |
|---|---|---|
| Spatial size | 32 × 32 pixels | Fixes every shape in the network: 32 → 16 → 8 through two poolings |
| Channels | 3 (red, green, blue) | The first Conv2D filter is 3×3×**3**, not 3×3, it spans all channels at once |
| Numbers per image | 32 × 32 × 3 = **3,072** | Each one is an input feature; the raw storage type is an integer 0–255 |
| Label | one integer, 0–9 | Not a one-hot vector, which is why the loss is *sparse* categorical crossentropy |

Three properties are worth naming explicitly in the writeup, because each one licenses a downstream choice:

- **The classes are mutually exclusive.** Every image has exactly one label and no image is both a cat and a dog. That is what makes a **softmax** output layer meaningful (Module 5): softmax produces ten probabilities summing to 1, a shape that only makes sense when exactly one option is correct. On a multi-label problem, "contains a cat" and "contains a car" both true, softmax would be the wrong layer.
- **The classes are perfectly balanced.** 6,000 images each, no rare class. That is what makes plain **accuracy** a meaningful headline number. On imbalanced data it stops being one: a fraud detector that predicts "not fraud" every single time scores 99% and is useless. Because your classes are balanced, accuracy and macro recall should track each other closely, and in your baseline they do (51.35% vs. 51.37%), which is a small consistency check worth mentioning.
- **It is a benchmark, not ground truth.** Standardisation is the point, thousands of published results use the same images, so your numbers are comparable. But the labels are human-assigned and imperfect: benchmark test sets contain recoverable label errors (Northcutt et al., 2021). At ~51% accuracy that is nowhere near your limiting factor, so mention it as a boundary on the dataset rather than as an explanation for your results. Claiming label noise caused your numbers would be overclaiming in the other direction.

### 2.2 What your notebook actually does with it

| Step | Code | Numbers |
|---|---|---|
| Load full dataset | `keras.datasets.cifar10.load_data()` | 50,000 train / 10,000 test |
| Sample a subset | `rng = np.random.default_rng(SEED)`, then `rng.choice(..., 12000, replace=False)` and `rng.choice(..., 3000, replace=False)` | 12,000 / 3,000 |
| Carve out validation | `x_train[-2000:]` | 10,000 train / 2,000 val / 3,000 test |
| Normalize | `.astype('float32') / 255.0` | pixel range 0–255 → 0.0–1.0 |

**`DATA_SEED = 42` is doing more work than it looks.** Both `rng.choice` calls draw without replacement from a generator seeded at 42, so the *same* 12,000 training indices and the *same* 3,000 test indices come out on every run, in every condition, on every seed of the paired experiment. The run seed varies the weight initialisation; the data seed does not vary at all. That distinction is what makes this an experiment rather than two unrelated measurements: the augmented model is not merely trained with the same *settings* as the baseline, it is trained on the same *images*. This is §1.2's controlled-variable list, implemented in two lines.

**Why subset at all?** The notebook's own comment calls it a "time-safe subset," and the trade is explicit: 12,000 of 50,000 images buys you a run that finishes in about 8 seconds on CPU (§R.1). That number is the reason a ten-seed paired experiment across two conditions was affordable at all, the entire study is about two minutes of compute. Be honest about the cost in your limitations: less training data means a lower accuracy ceiling (yours sits near 51%) and more variance between runs, both of which shape everything in Module R.

**One detail about balance that catches people out.** The *full* dataset is exactly balanced. A random subset of it is only *approximately* balanced, because random sampling does not preserve exact proportions. Your baseline per-class report shows test supports running from 275 to 313 rather than a flat 300. So "roughly 300 test images per class" is the right way to say it, and that number is precisely why §1B.8 flagged H-D as risky: a few percentage points of per-class recall is a handful of images, well inside the range chance can produce. Note also that this is only "approximately balanced," not "biased", the sampling is unbiased, the deviations are just sampling noise.

There is **no resizing**, images stay 32×32 (§2.6 explains why that is correct, not an omission), and no per-channel standardisation (§2.3). Both are legitimate things to discuss as possible enhancements.

### 2.3 Why divide by 255?

Raw pixels arrive as integers from 0 to 255. One line converts them to floats in [0, 1]:

```python
x_train = x_train.astype('float32') / 255.0
```

The short answer is that **gradient descent behaves badly when inputs are large, or on mixed scales.** Two distinct problems hide inside that sentence and it's worth separating them.

**Problem one: magnitude.** A neuron computes a weighted sum of its inputs. Feed it values in the hundreds and the pre-activation sums land in the thousands, which pushes activations far from the range where they carry useful variation. Worse, the gradient of the loss with respect to a weight has the corresponding *input value* as a factor, so inputs in the hundreds produce gradients scaled up by the same factor. Multiply a large gradient by the learning rate and the update step overshoots the minimum it was aiming at. The loss then oscillates, or diverges outright, and none of it is the model's fault. Dividing by 255 shrinks inputs, activations and gradients together into a numerically comfortable band.

**Problem two: mixed scales.** When different input features live on different scales, the loss surface becomes a long narrow valley rather than a round bowl, and a single learning rate is simultaneously too large for the steep direction and too small for the shallow one. Your three colour channels already share the 0–255 scale, so this problem is milder here than in tabular data where one column is "age in years" and the next is "income in dollars", but it is the reason the step is universal rather than optional.

Three details to get right:

- **Apply it identically to train, validation and test.** The notebook does. If you normalised the training set and forgot the test set, the model would meet inputs at 255× the scale it was trained on and accuracy would collapse to near-chance. It is a classic bug and it looks like a modelling disaster rather than a preprocessing slip.
- **This is rescaling, not standardisation.** Per-channel standardisation, subtract the mean, divide by the standard deviation, per colour channel, is the common next step and a fair enhancement to propose. If you do propose it, say the important part: **compute those statistics on the training set only, then apply them unchanged to validation and test.** Computing them across all the data leaks test-set information into your preprocessing, which is the same sin §2.4 is about, just wearing a different hat.
- **Adam softens this, but doesn't remove it.** Adam adapts a step size per parameter from the recent gradient history, so it is less sensitive to input scaling than plain SGD would be. That's why an unnormalised run might limp along rather than explode. It is not a reason to skip the step; it's a reason the failure would be quiet rather than obvious.

### 2.4 Why three splits and not two

| Split | Size | Who learns from it | How often it is touched |
|---|---|---|---|
| **Train** | 10,000 | **The model**, weights update on it | Every batch of every epoch |
| **Validation** | 2,000 | **You**, the model never updates on it | Once per epoch |
| **Test** | 3,000 | **Nobody**, until the very end | Once |

The two-split version, train and test only, looks simpler, and it is what most people write first. It breaks the instant you make any decision by looking at the test number. And you *will* make decisions: which augmentation intensity, whether to add dropout, how many epochs, when to stop. Every decision made by consulting test accuracy moves a little test-set information into your model, not through the optimizer, through **you**. Do it enough times and the test set is no longer measuring "data nobody has seen." It is measuring "data I have been slowly, manually optimising against." The name is **leakage**; the symptom is an accuracy figure that looks great and does not reproduce anywhere else.

The validation set exists to absorb exactly those decisions. It is the set you are allowed to look at as often as you like, and the *reason* you're allowed is that you have quarantined a third set you never look at. The moment you use test performance to pick a setting, it stops being an honest generalization estimate, you have leaked information from it into your choices. Keeping it untouched is the whole point.

For this project that discipline had teeth, and you should point at the specific place it bit. §1C.6 told you to read the baseline curves and then decide which hypothesis leads. That is a real decision made from data, and it was made on **validation** behaviour, which is what validation is for. Had it been made from test accuracy, the entire pre-registration story in Module 1C would have collapsed, because the set you used to choose your hypothesis would have been the same set you later used to test it.

One honest wrinkle to state rather than gloss: at 3 epochs, with no early stopping and no hyperparameter search, your validation set is doing **less** work than it would in a typical project. It produced the per-epoch curves, supplied §1C.6's decision, and provided the clean generalization-gap term for H-B. That is real, but it is not a tuning loop. Describe what validation actually did here rather than implying a search you never ran, the reader can tell, and a precise account of a modest role reads far better than an inflated one.

### 2.5 The open question, answered: is the last 2,000 rows a random validation split?

The sketch left you this question: *your notebook takes the validation set as the last 2,000 rows of an already randomly-sampled subset. Is that a random validation split?*

**Yes, it is a valid random split.** But the reasoning is the whole value here, because the answer depends entirely on a step that happened earlier.

```python
train_idx = rng.choice(len(x_train_all), 12000, replace=False)
x_train, y_train = x_train_all[train_idx], y_train_all[train_idx]
...
x_val, y_val   = x_train[-2000:], y_train[-2000:]
x_train, y_train = x_train[:-2000], y_train[:-2000]
```

"Take the last 2,000 rows" is a **positional** operation, and slicing by position is only as random as the ordering it slices. Here the ordering was already randomised: `rng.choice(..., replace=False)` drew 12,000 indices at random from 50,000 and placed them in the order it drew them. Row 11,999 of `x_train` holds whichever image the random draw happened to deposit there, it has no relationship to class, to file order, or to anything else. So the last 2,000 rows are a uniformly random subset of the 12,000, exactly as random as shuffling and taking 2,000 from the middle. The split is valid, and your notebook is fine.

Now the condition to check, which is why the question was worth asking rather than assuming:

| Situation | Is `[-2000:]` a random split? | Why |
|---|---|---|
| Indices drawn by `rng.choice`, **your notebook** | ✅ **Yes** | Selection already randomised the row order, so position carries no information |
| Data taken **sequentially** from a class-ordered file | ❌ **No** | The last 2,000 rows would be the last classes. Validation might contain two classes and training the other eight, both sets worthless, and the failure would look like a modelling problem |
| Data sequential and ordered by capture time or collection batch | ❌ **No** | Validation is drawn from a different period than training, so the gap measures drift rather than generalization |

The rule that generalises out of this:

> **A positional split is random if and only if the row ordering is random.**

So the question to ask of any slice-based split is never "is slicing random?", it isn't, it never is, but "**what determined this row order?**" Answer that and you have answered the split. This is a habit worth carrying: the same reasoning catches leakage in time-series splits, grouped-subject data, and anything sorted before it was partitioned.

Two smaller observations while you are here, both of which are one-sentence additions to your methods:

- **The validation set is not stratified.** Nothing guarantees exactly 200 per class, random sampling gives you the right proportions *on average*, with sampling noise around them. At 2,000 images those deviations are small and harmless. On a much smaller validation set you would want stratified sampling, which forces the class proportions to match exactly.
- **Train and test cannot overlap.** The two `rng.choice` calls sample from `x_train_all` and `x_test_all`, which CIFAR-10 keeps as separate, disjoint pools, so no image can appear in both. Worth stating explicitly, because train/test contamination is the first thing a careful reader checks and the cheapest accusation to pre-empt.

### 2.6 Why nothing gets resized: and why 32×32 is the centre of your augmentation argument

The obvious question is why you don't resize *up*. Most well-known CNN architectures expect 224×224 inputs, so 32×32 feels undersized.

Three reasons not to:

- **There is no information to gain.** Upsampling 32×32 to 224×224 interpolates, it manufactures 49 times as many pixels from the 1,024 you actually had. The result looks smoother and contains not one additional bit of detail. What you pay is roughly 49× the compute per image and a far larger Flatten layer feeding the dense head.
- **Resizing is a requirement, not an improvement.** You resize when a pretrained backbone demands a fixed input size, or when your images arrive at inconsistent dimensions. Neither applies: your model is built from scratch (Module 3), and CIFAR-10 is uniformly 32×32 already.
- **Downsizing would destroy signal** at a resolution that has almost none to spare.

So "no resizing" is the correct choice, and the architecture is designed around it, 32 → 16 → 8 through two pooling layers, landing on the 8 × 8 × 64 that Module 3's shape table flattens into 4,096.

**Now the part that matters for your experiment.** 32×32 is *small*. One CIFAR-10 image holds 1,024 pixels per channel; a phone photograph holds millions. At that resolution a whole animal might occupy twenty pixels across, and every geometric transform you apply has consequences that would be invisible on a large image:

| Transform | What it does at high resolution | What it does at 32×32 |
|---|---|---|
| **Rotation** | Rotates smoothly; interpolation blur is imperceptible; corner loss is a thin sliver | Pixels land between grid positions, so every pixel is interpolated from its neighbours, visibly smearing the few edges the model has to work with. Corners have no source pixels and must be filled or cropped, and a meaningful fraction of a small object can leave the frame |
| **Zoom** | Crops or pads a large canvas with room to spare | Crops *and* resamples, compounding the same interpolation loss; zooming out pads a frame that had little context to begin with |
| **Horizontal flip** | Exact | **Exact**, a mirror is a pure index reversal, no interpolation, no loss. This is why flip is the safe transform at any resolution (§1.5) |

That table is the "**added invariance vs. destroyed signal**" tradeoff from §1.5, made specific. It is also exactly why §1C.4 insisted the method source had to come *before* the intensity choice rather than after it: this is a question the literature has evidence about. Rotation at low resolution destroys pixel information (Alomar et al., 2023); transforms severe enough to push a sample outside its class's natural variation stop being label-preserving altogether (Xu et al., 2023), a rotated, zoomed 32×32 bird can simply stop containing enough evidence of "bird"; and stacking several transforms together can underperform a more selective choice (Ottoni et al., 2023), which is a direct comment on the notebook's default of flip + rotation + zoom all at once.

Your own results are consistent with that reading. Bird recall fell from 0.252 to 0.172 under moderate augmentation, the largest per-class loss in the experiment (§R.1, §R.2), and bird is a small, thin-featured class with little margin for interpolation blur. Be careful how you phrase it, though: this is **consistent with** the resolution mechanism, not evidence for it, because you ran one intensity and no sweep. That gap between what you observed and what you can attribute is precisely what makes an **intensity sweep** (conservative / moderate / stronger, from §1C.4's table) the obvious next step, and §1C.3 already explains why it is a separate experiment rather than something to bolt onto this one.

Finally, state the resolution explicitly whenever you bound a claim. Your finding is about **32×32 natural images under a 3-epoch budget**, not about augmentation, and not about CNNs. How augmentation behaves at 224×224 with a long training schedule is a different question that your data says nothing about. That is §1B.6's mistake (b), name the population you are generalising to, applied to the dataset module, and it is the sentence that keeps §R.8's "without overclaiming" honest.

---

## Module 3: What convolution actually does ✅

Two rubric lines run through Modules 3 to 6: **architecture** and **mathematical foundation**. This module is where you stop describing your model as a list of layers you were handed and start explaining why each one is there. The professor's stack is fixed and that is fine. Being given an architecture does not excuse you from understanding it, and the most natural question a grader can ask is also the simplest: why a convolution at all?

### 3.1 The problem convolution exists to solve

Start with what a fully connected layer would have cost you.

A CIFAR-10 image is 32 × 32 × 3, which Module 2 counted as **3,072 numbers**. Suppose you wanted a first layer producing 32 filtered versions of that image at full resolution. That is exactly what your `Conv2D(32, 3×3, padding='same')` produces: an output of 32 × 32 × 32 = **32,768 values**. Wire it as a dense layer, where every output connects to every input, and you pay:

> 3,072 × 32,768 + 32,768 = **100,696,064 parameters**

Your actual convolutional layer produces those same 32,768 outputs from **896 parameters**. That is a ratio of roughly 112,000 to 1, and it comes from a single idea.

### 3.2 A kernel is a small window of shared weights

A convolutional filter (or kernel) is a small patch of weights. Yours is 3 × 3 and it spans all three colour channels at once, so it is really 3 × 3 × 3 = **27 weights, plus one bias, 28 numbers in total**.

The operation is this. Place the 3 × 3 × 3 window over the top-left corner of the image. Multiply each of the 27 weights by the pixel value underneath it, add the 27 products together, add the bias, and write that single number into an output grid. Slide the window one pixel to the right and do it again. Keep sliding until you have covered the image.

The word doing all the work is **slide**. The same 28 numbers are reused at every position. A dense layer would have learned a separate weight for the pixel at (0,0) and another for the pixel at (17,23), as though those were unrelated quantities. A convolution learns one small pattern detector and applies it everywhere.

This is why the cost of a filter is **independent of image size**. A 3 × 3 × 3 filter costs 28 parameters on a 32 × 32 image, on a 512 × 512 image, and on a satellite photograph. Only the amount of *computation* grows with resolution, not the number of things to learn. Say that sentence in your presentation; it is the cleanest one-line answer to "why not just use a dense network?"

### 3.3 Feature maps, stride and padding

The output grid a single filter produces is called a **feature map**. It is an image in its own right: bright where the filter's pattern was present, dark where it was not. Your first layer has 32 filters, so it produces 32 feature maps stacked into a 32 × 32 × 32 tensor. The third dimension is no longer "colour"; from here on it is "which feature."

**Stride** is how far the window moves between applications. Keras defaults to stride 1, which your notebook keeps, so the window shifts one pixel at a time and the output grid is (almost) as large as the input. Stride 2 would move two pixels at a time and halve the output resolution, which is an alternative way to downsample. Your model downsamples with pooling instead (Module 4).

**Padding** decides what happens at the edges. With no padding, a 3 × 3 window cannot be centred on the outermost row or column, because part of it would hang off the image. The output therefore shrinks:

> output size = (32 − 3) / 1 + 1 = **30**

`padding='same'` adds a one-pixel border of zeros around the image before sliding, so a 3 × 3 window can be centred on every original pixel and the output comes back at **32 × 32**. That is what the option name means: same spatial size in, same spatial size out.

It matters more than it sounds. Follow both versions through your architecture:

| Stage | With `padding='same'` (your model) | With `padding='valid'` |
|---|---|---|
| Input | 32 × 32 | 32 × 32 |
| Conv 1 | 32 × 32 | 30 × 30 |
| Pool 1 | 16 × 16 | 15 × 15 |
| Conv 2 | 16 × 16 | 13 × 13 |
| Pool 2 | **8 × 8** | 6 × 6 |
| Flatten | 8 × 8 × 64 = **4,096** | 6 × 6 × 64 = 2,304 |
| Dense(64) cost | 262,208 | 147,520 |

Two pixels lost per convolution does not sound like much until you notice that on a 32 × 32 image it is a 6% shave off each side, taken twice, and that everything downstream inherits it. `padding='same'` also protects border pixels from being seen fewer times than interior ones, which on an image this small is a real fraction of the picture.

### 3.4 Translation equivariance

This is the conceptual pay-off, and it is worth getting the word right because the near-synonym means something different.

**Equivariance:** shift the input, and the feature map shifts by the same amount. If your filter has learned to fire on a horizontal edge, and the edge moves five pixels right, the bright spot in the feature map moves five pixels right. The detection still happens, performed by *the same weights*.

**Invariance:** shift the input and the output does not change at all.

Convolution gives you equivariance, not invariance. That is still enormous. It means the network never has to learn "edge in the top-left corner" and "edge in the bottom-right corner" as two separate facts, which is precisely the redundancy a dense layer would be forced to pay for. A feature detected anywhere is detected by one shared set of 28 numbers.

Invariance is assembled later, in pieces: pooling contributes a small amount of local invariance (§4.2), and augmentation deliberately injects the invariances the architecture does not supply for free (§1.5, Module 9). Keeping those three mechanisms separate in your head is genuinely useful, because your whole experiment is about the third one.

### 3.5 The parameter arithmetic, layer by layer

The formula for a convolutional layer is:

> params = (kernel height × kernel width × input channels + 1) × number of filters

The `+ 1` is the bias, one per filter. The **input channels** term is the part people forget: a filter always spans the full depth of what it is looking at. Your first layer's filters are 3 × 3 × **3** because the input has three colour channels. Your second layer's filters are 3 × 3 × **32**, because they look at the 32 feature maps the first layer produced, not at an image.

| Layer | Output shape | Params | Arithmetic |
|---|---|---|---|
| Input | 32 × 32 × 3 | 0 | |
| Conv2D(32, 3×3, same) | 32 × 32 × 32 | 896 | (3 × 3 × 3 + 1) × 32 |
| MaxPooling2D | 16 × 16 × 32 | 0 | no weights |
| Conv2D(64, 3×3, same) | 16 × 16 × 64 | 18,496 | (3 × 3 × 32 + 1) × 64 |
| MaxPooling2D | 8 × 8 × 64 | 0 | no weights |
| Flatten | 4096 | 0 | 8 × 8 × 64 |
| Dense(64) | 64 | 262,208 | 4096 × 64 + 64 |
| Dropout(0.0) | 64 | 0 | no weights |
| Dense(10, softmax) | 10 | 650 | 64 × 10 + 10 |
| **Total** | | **282,250** | 896 + 18,496 + 262,208 + 650 |

Check the second convolution by hand, because it is the one that shows you understand the depth term: 3 × 3 × 32 = 288, plus 1 bias is 289, times 64 filters is **18,496**. If you had wrongly used 3 × 3 = 9 you would have got 640, and `model.summary()` would have caught you out.

**Now the observation worth building a slide around.** The two convolutional layers together hold 896 + 18,496 = **19,392 parameters, under 7% of the model**. Roughly **93% of all 282,250 parameters sit in the dense head**, and almost all of that in the single `Dense(64)` after the flatten. The part of the network that does the actual image processing is a rounding error next to the part that does not. Module 4 is about why that happens and what to do about it.

Parameters are also not the same thing as work. Layer 1 computes 32,768 output values, each from 27 multiply-adds, which is **884,736** multiply-accumulate operations from only 896 weights. Layer 2 computes 16,384 outputs at 288 multiply-adds each, about **4.7 million** operations from 18,496 weights. Convolutions are parameter-cheap and compute-hungry; dense layers are the reverse. Keep them apart when you talk about "model size."

### 3.6 Why the first layer learns edges and the second learns combinations

Nobody tells the filters what to detect. They are initialised randomly and shaped only by gradient descent on the loss. But the structure of the problem makes what they converge to fairly predictable.

A first-layer filter sees a 3 × 3 × 3 patch of raw pixels: nine locations, three colours. Nine pixels is not enough to contain a wheel or a beak. The only things expressible in that window are **local intensity and colour transitions**: a bright-to-dark step in some orientation, a corner, a blob of one colour against another. Those are edge detectors and colour-blob detectors, and they emerge because they are the most informative functions available at that receptive field size.

The second layer is in a completely different position. Its input is not pixels; it is 32 feature maps that already say "there is a vertical edge here," "there is a green-to-brown transition here." Its 3 × 3 × 32 filter therefore asks a question about **co-occurrence**: is there a vertical edge next to a horizontal edge in this arrangement, with this colour transition below it? Combinations of edges are corners, junctions, textures and simple part-like shapes.

The **receptive field** arithmetic makes this concrete. Track how much of the original image a single unit can see:

| After | Receptive field on the input |
|---|---|
| Conv 1 | 3 × 3 pixels |
| Pool 1 | 4 × 4 pixels |
| Conv 2 | **8 × 8 pixels** |
| Pool 2 | 10 × 10 pixels |

A unit in the second convolutional layer sees an 8 × 8 window, a quarter of the image's width. That is the honest ceiling on what your network can represent: no single unit anywhere in this model looks at a region larger than 10 × 10 out of 32 × 32 before the flatten collapses everything. A deeper stack would keep enlarging that window until units could see whole objects, which is one straightforward reason a two-block CNN plateaus near 51% on CIFAR-10 (§R.1) while deeper models do far better. That is a limitation of the architecture, not of your experiment, and Module 12 is where it belongs.

### 3.7 What this buys you in the submission

Three sentences you can now defend, and each of them is the kind of thing an examiner probes:

- Weight sharing is why a 3 × 3 × 3 filter costs 28 parameters regardless of image size, and why this CNN has 282,250 parameters where a comparable dense first layer alone would have needed over 100 million.
- `padding='same'` is what keeps the spatial chain at 32 → 16 → 8, which is where the 4,096-unit flatten comes from.
- The convolutions are 7% of the model and the dense head is 93%, which is the single most striking fact in `model.summary()` and the seed of your strongest future-work item.

---

## Module 4: Pooling, flatten and the dense head ✅

Module 3 covered the layers that do the seeing. This one covers the layers that throw information away and then commit to an answer. Both of your pooling layers and the flatten hold **zero parameters between them**, which makes them easy to skip over in a presentation and is exactly why they are worth a slide: they shape everything, and they cost nothing.

### 4.1 What MaxPooling2D actually does

`MaxPooling2D()` in Keras defaults to a 2 × 2 window, and the stride defaults to the window size, so the windows are **non-overlapping**. The image is tiled into 2 × 2 blocks and each block is replaced by its largest value. Four numbers become one.

Applied to your first feature-map stack: 32 × 32 × 32 becomes 16 × 16 × 32. Applied after the second convolution: 16 × 16 × 64 becomes 8 × 8 × 64. Note that pooling acts **per channel**. It halves height and width and leaves depth untouched, which is why the channel count survives unchanged through both poolings.

There are no weights. Taking a maximum is a fixed rule, not a learned one, so pooling contributes 0 to the 282,250.

### 4.2 Why downsampling is useful

Three reasons, in ascending order of how often they get stated correctly.

**Fewer activations downstream.** Count them:

| Stage | Activations | Running total effect |
|---|---|---|
| After Conv 1 | 32 × 32 × 32 = 32,768 | |
| After Pool 1 | 16 × 16 × 32 = **8,192** | 4 × reduction |
| After Conv 2 | 16 × 16 × 64 = 16,384 | |
| After Pool 2 | 8 × 8 × 64 = **4,096** | 4 × reduction |

Take the poolings out and the second convolution would run on a 32 × 32 grid, producing 32 × 32 × 64 = 65,536 values, and the flatten would hand 65,536 features to `Dense(64)`. That layer would then cost 65,536 × 64 + 64 = **4,194,368 parameters**, sixteen times its current 262,208. Notice what did *not* change: `Conv2D(64, 3×3)` still costs 18,496 either way, because convolutional cost depends on kernel size and channel count, never on spatial size. Pooling saves parameters only in the dense head. It saves computation everywhere.

**A larger receptive field for free.** §3.6 traced this: the pooling is what lifts the second convolution's view from 4 × 4 to 8 × 8 pixels. Without downsampling you would need many more layers to see any meaningful fraction of the image.

**A small amount of translation invariance.** This is the one people overstate, so be precise. If a feature moves by one pixel and stays inside the same 2 × 2 block, the maximum is unchanged and the pooled output is identical. If it moves across a block boundary, the output changes. So pooling buys **local** invariance, on the order of a pixel at a time, not the general "the object can be anywhere" invariance the phrase suggests. Module 3's distinction still holds: convolution gives equivariance, pooling adds a sliver of invariance, and augmentation is where you inject the rest deliberately (Module 9).

### 4.3 What pooling discards

Three of every four values, and the exact position of the survivor.

After the max is taken, the network knows a strong response occurred somewhere in that 2 × 2 block, and it no longer knows where, nor what the other three responses were. On a 224 × 224 image that is a negligible sacrifice. On a 32 × 32 image, where an entire animal may span twenty pixels (§2.6), two rounds of 2 × 2 pooling take you from 1,024 spatial positions to 64. You are making a real trade, and it is fair to name it as a limitation rather than presenting pooling as free.

Average pooling is the obvious alternative: take the mean of the four instead of the maximum. Max pooling asks "was this feature present?" and average pooling asks "how much of this feature was around?" Max is the conventional choice for detection-style features and it is what your notebook uses. It is a defensible design question to raise, not a flaw to fix.

### 4.4 Flatten: where the spatial structure dies

`Flatten()` takes the 8 × 8 × 64 tensor and reads it out into one long vector:

> 8 × 8 × 64 = **4,096**

No parameters, no arithmetic, no choice. It is a reshape.

What it costs is conceptual rather than computational. Up to this point the network has been **equivariant**: features carried their positions with them. After the flatten, feature 37 at position (2,5) is simply element 1,573 of a vector, indistinguishable in kind from any other element. The dense layer that follows learns a separate weight for "feature 37 in the top-left" and another for "feature 37 in the bottom-right," as two unrelated facts. Every property Module 3 spent a page establishing is discarded in one line.

That is not automatically wrong. Absolute position sometimes carries information: sky tends to be at the top, ground at the bottom. But it is where the parameter explosion comes from, and it is why the model has to relearn a great deal of what the convolutions already knew.

### 4.5 The dense head, and why 93% is lopsided

> 4,096 × 64 + 64 = **262,208**

One layer. 64 units. 262,208 parameters, because each of the 64 units needs its own weight for every one of the 4,096 flattened features.

| Block | Parameters | Share of model |
|---|---|---|
| Conv 1 | 896 | 0.3% |
| Conv 2 | 18,496 | 6.6% |
| **Convolutional total** | **19,392** | **6.9%** |
| Dense(64) | 262,208 | 92.9% |
| Dense(10) | 650 | 0.2% |
| **Dense head total** | **262,858** | **93.1%** |
| **Model** | **282,250** | 100% |

Say what this means rather than just reading the numbers out. **Roughly 93% of your model's learning capacity is spent on a layer that cannot see spatial structure at all**, while the 7% that can see it is doing the work that makes this a CNN. A model this shaped is mostly a dense classifier with a small feature extractor bolted to the front. It is also the part of the model most able to memorise the training set, since a dense layer with a quarter of a million weights on 10,000 images has ample room to fit noise, which is a point worth carrying into Module 8.

This shape is not unusual. It is what early CNN designs looked like, and the starter notebook is a faithful teaching example of that style. You are not criticising the professor by pointing it out; you are demonstrating that you read `model.summary()` and understood it.

### 4.6 GlobalAveragePooling2D: your strongest future-work item

The standard modern replacement for `Flatten()` is `GlobalAveragePooling2D()`. Instead of reading 8 × 8 × 64 out as 4,096 numbers, it **averages each 8 × 8 feature map down to a single number**, giving **64 features**: one per channel, each answering "how strongly was this feature present anywhere in the image?"

Run the arithmetic all the way through, because the comparison is dramatic:

| Head design | Features into the head | Dense(64) params | Dense(10) params | Head total | Model total |
|---|---|---|---|---|---|
| `Flatten` (your model) | 4,096 | 262,208 | 650 | 262,858 | **282,250** |
| `GlobalAveragePooling2D` | 64 | 64 × 64 + 64 = **4,160** | 650 | 4,810 | **24,202** |

The model shrinks by a factor of about **11.7**, and the head falls from 93.1% of the parameters to roughly 20%. The convolutional layers are untouched at 19,392, so the balance flips: the network becomes mostly feature extractor, which is what a convolutional network is supposed to be.

Global average pooling also restores full translation invariance, because an average over the whole map does not care where the activation was, and it acts as a structural regularizer: there are simply far fewer weights available to memorise with.

**Now be honest about what you can predict.** Global average pooling is a regularizing, capacity-reducing change, and §R.1 established that your baseline is **underfitting** at 3 epochs, not overfitting. Applying a regularizer to an underfitting model is exactly the mistake the augmentation result already documented (§R.2, Lin et al., 2024). So the defensible framing is architectural, not performance-based:

> Replacing `Flatten` with `GlobalAveragePooling2D` would cut the model from 282,250 to about 24,202 parameters and move it from 93% dense to roughly 80% convolutional. I would propose it as an architectural rebalancing that frees budget for a third convolutional block, rather than as a change I expect to raise accuracy on its own at a 3-epoch budget.

That sentence is worth more than "we would use global average pooling to improve the model," because it shows you know which regime you are in and refuses to promise something your own data argues against. It is the same discipline §R.8 asked for, applied forward instead of backward.

### 4.7 Dropout(0.0), while we are here

Your architecture contains `Dropout(0.0)`. Dropout randomly zeroes a fraction of activations during training, forcing the network not to depend on any single unit. At rate 0.0 it zeroes nothing: it is a **no-op**, a placeholder left in the stack with its dial turned to zero.

Mention it in the architecture description and say plainly that it is inactive. Two reasons. First, a reader looking at your layer list will otherwise assume you had dropout regularization and wonder why you needed augmentation. Second, it is another regularization dial sitting at zero on a model that is underfitting, which is entirely consistent with the picture Module 8 builds. It is also a controlled variable in the §1.2 sense: identical in both arms, so it cannot explain any difference you measured.

---

## Module 5: Activations, ReLU and softmax ✅

Softmax is one of the approved options for the assignment's **mathematical foundation** requirement, so this module has to be more than intuition. You need the formula, what it does, and what it does not do. ReLU comes first because softmax only makes sense once you understand why non-linearity is in the network at all.

### 5.1 Why any non-linearity is required

This is the argument to have ready, because it is short, it is provable in one line, and it is the reason activations exist.

A dense layer computes W·x + b. Stack two of them with nothing in between:

> layer 2 output = W₂(W₁x + b₁) + b₂ = (W₂W₁)x + (W₂b₁ + b₂)

W₂W₁ is just another matrix, call it W'. W₂b₁ + b₂ is just another vector, call it b'. So two stacked linear layers compute W'x + b', which is **one linear layer**. The same argument extends to any depth, and convolution is a linear operation too, so it applies to your convolutional layers as well.

Strip the ReLUs out of your model and all 282,250 parameters would collapse to a single linear map from 3,072 inputs to 10 outputs. That is multinomial logistic regression with extra steps. Depth would buy you literally nothing.

**Non-linearity is what makes depth mean something.** Each ReLU bends the function, and stacking bent functions produces shapes a single linear map cannot express. That is the whole reason a two-layer CNN can do anything a linear model cannot.

### 5.2 ReLU

> ReLU(x) = max(0, x)

Negative in, zero out. Positive in, unchanged out. That is the entire definition, and its derivative is just as simple: 1 where x > 0, 0 where x < 0, undefined exactly at 0 where frameworks conventionally use 0.

Your model applies it in three places: after each convolution, and after `Dense(64)`. The output layer does not use it, because softmax goes there instead.

### 5.3 Why ReLU replaced sigmoid and tanh

For years the default activations were the sigmoid, σ(x) = 1 / (1 + e⁻ˣ), and tanh. ReLU displaced both, for two reasons.

**Cost.** Sigmoid and tanh require an exponential per activation. ReLU requires a comparison against zero. Your first convolutional layer alone produces 32,768 activations per image, times 64 images per batch, times 157 batches, times 3 epochs. Small per-element savings compound.

**Vanishing gradients, which is the real reason.** Backpropagation multiplies derivatives together layer by layer. The sigmoid's derivative is σ(x)(1 − σ(x)), which peaks at **0.25** when x = 0 and falls toward zero as the input moves away from the origin in either direction. So every sigmoid layer multiplies the gradient by at most 0.25, and usually far less:

| Depth | Best-case gradient factor through sigmoids | Through ReLUs (positive inputs) |
|---|---|---|
| 2 layers | 0.25² = 0.0625 | 1 |
| 4 layers | 0.25⁴ ≈ 0.0039 | 1 |
| 8 layers | 0.25⁸ ≈ 0.000015 | 1 |

The early layers of a deep sigmoid network receive gradients so small that they barely move, and the network effectively refuses to train its first layers at all. ReLU's derivative is exactly **1** for every positive input, so the gradient passes through undiminished. That single property is most of why deep networks became trainable.

Your model is only four weight layers deep, so you would probably survive sigmoids here. The argument is still the right one to give, because it explains why ReLU is the default you inherited.

### 5.4 The dying ReLU

Every activation has a failure mode and this is ReLU's. If a unit's pre-activation is negative for **every** input in the dataset, its output is always zero, its derivative is always zero, and therefore its gradient is always zero. No gradient means no update. The unit is stuck at zero permanently, and it stays stuck: a dead unit cannot revive itself, because the mechanism that would revive it is the gradient it no longer receives.

It usually starts with an oversized update. A large learning rate, or a large gradient from unnormalised inputs (§2.3), can drive a unit's bias sharply negative in a single step, and if it lands far enough below zero the unit never comes back. A network can quietly lose a substantial fraction of its capacity this way while still training, just worse.

Mitigations: a smaller learning rate; Leaky ReLU, which uses a small negative slope such as 0.01x for x < 0 so the gradient never becomes exactly zero; and sensible initialisation.

For your project this is a named failure mode rather than a diagnosis. Your inputs are normalised to 0–1, you use Adam at its default learning rate, and you train for 471 updates, which is not many opportunities to kill a unit. If you wanted to check, you would look at the fraction of activations that are exactly zero for every image in a batch. It is worth mentioning as something you considered and had no evidence of, which is different from ignoring it.

### 5.5 Softmax

Your final layer produces 10 raw numbers, one per class. They are called **logits**. They can be any real value, positive or negative, they do not sum to anything in particular, and on their own they mean nothing except that bigger is better. Softmax converts them into a probability distribution:

> p_i = exp(z_i) / Σⱼ exp(z_j)

Take the exponential of every logit, then divide each by the total. Two properties follow immediately: every p_i is positive, because exp is always positive; and they sum to exactly 1, because you divided by their sum.

A worked example, illustrative rather than measured. Take three logits z = [2.0, 1.0, 0.1]:

| Step | Class A | Class B | Class C | Sum |
|---|---|---|---|---|
| Logit z | 2.0 | 1.0 | 0.1 | |
| exp(z) | 7.389 | 2.718 | 1.105 | 11.212 |
| p = exp(z) / sum | **0.659** | 0.242 | 0.099 | 1.000 |

Notice what the exponential did. The logit gap between A and B is 1.0, and the probability ratio is 7.389 / 2.718 ≈ 2.72. Softmax turns *additive* differences in logits into *multiplicative* differences in probability, which is why a model that is only slightly more confident in one class can end up assigning it most of the mass.

### 5.6 Two things softmax does not do

**It never changes which class wins.** exp is strictly increasing and every logit is divided by the same denominator, so softmax is **monotonic**: the largest logit always produces the largest probability. Your predicted label is `argmax` of the logits and `argmax` of the probabilities, always, identically. Softmax therefore has no effect whatsoever on your accuracy, precision, recall, or confusion matrix. If you deleted it and took the argmax of the raw logits, every prediction in your report would be unchanged.

That is a genuinely surprising fact and a good thing to say out loud, because it forces the follow-up question: then what is it for?

**It does not give you calibrated confidence.** A softmax output of 0.95 does not mean the model is right 95% of the time when it says that. Neural networks are routinely overconfident, and calibration is a separate property that has to be measured, not assumed. Your model averages 51.35% accuracy (§R.1), so treat its probabilities as scores rather than as trustworthy confidence estimates, and do not write sentences like "the model was 90% certain."

### 5.7 What softmax is actually for

**It makes the output compatible with cross-entropy.** Module 6's loss is L = −log(p_c), the negative log of the probability assigned to the correct class. That expression requires a probability: a positive number no greater than 1. A raw logit of −3.2 cannot be fed into it. Softmax is the bridge between "the network produces ten arbitrary real numbers" and "the loss function needs a probability distribution."

Two supporting reasons follow from that. Because the outputs must sum to 1, raising the probability of one class necessarily lowers the others, so the ten classes **compete**, which is the right structure for a mutually exclusive problem (§2.1). And the gradient of cross-entropy composed with softmax simplifies to (p − y), the predicted distribution minus the one-hot truth, which is unusually clean and well-behaved to optimise.

Two details worth knowing:

- **Shift invariance.** softmax(z + c) = softmax(z) for any constant c, because the added exp(c) cancels between numerator and denominator. Implementations exploit this by subtracting the maximum logit before exponentiating, which prevents overflow on large logits. It is a numerical trick with no effect on the result.
- **Softmax then cross-entropy is slightly less stable than the fused version.** Your notebook puts `softmax` in the last layer and uses `sparse_categorical_crossentropy` with its default `from_logits=False`. The alternative is a linear output layer plus `from_logits=True`, which computes the log and the softmax together and avoids taking a logarithm of a number that has already been rounded. Mathematically identical, numerically a little safer. It is a fair one-line observation for a technical-detail slide, not a bug.

### 5.8 For the submission

If you pick softmax as your mathematical foundation, the four things to cover are: the formula; that it converts ten unbounded logits into a distribution summing to 1; that it is monotonic and therefore never changes the predicted class; and that its real purpose is to make the output compatible with cross-entropy. The third point is the one most students miss, and it pairs naturally with Module 6.

---

## Module 6: Loss and optimization ✅

Cross-entropy is the other approved **mathematical foundation** option, and it pairs so naturally with softmax that covering both costs you almost nothing extra. Module 5 ended with ten probabilities. This module is about turning them into a single number the optimizer can push downhill, and about the machinery that does the pushing.

### 6.1 What a loss function is for

Training needs one scalar. Not ten probabilities, not an accuracy, not a confusion matrix: one number that gets smaller as the model gets better, and that can be differentiated with respect to all 282,250 parameters.

Accuracy cannot do this job. It is a count of correct predictions, so it changes in steps as predictions flip from wrong to right, and its derivative is zero almost everywhere. There is no downhill direction in it. A model that moved the correct class's probability from 0.20 to 0.45 has improved enormously and its accuracy has not changed at all. The loss is what notices.

### 6.2 Cross-entropy

For a single example whose correct class is c:

> L = −log(p_c)

Only the probability assigned to the **true** class enters the formula. The other nine are ignored, which surprises people until they remember softmax made them sum to 1: pushing p_c up automatically pushes the rest down, so the other classes are penalised implicitly.

The behaviour of −log is the whole story:

| p assigned to the correct class | Loss −ln(p) | Reading |
|---|---|---|
| 0.99 | 0.010 | confident and right, almost no penalty |
| 0.90 | 0.105 | |
| 0.70 | 0.357 | |
| 0.50 | 0.693 | hedging |
| 0.10 | 2.303 | this is chance on 10 classes |
| 0.01 | 4.605 | confident and wrong, heavily punished |
| 0.001 | 6.908 | |

The curve is gentle on the right and unbounded on the left. As p_c approaches 0 the loss approaches infinity. **Cross-entropy does not just want you to be right; it wants you to not be confidently wrong.** A model that hedges at 0.50 on everything pays 0.693 per example. A model that is brilliant on 90 examples and catastrophically certain on 10 pays far more. That asymmetry is deliberate, and it is why cross-entropy is the standard loss for classification.

**A free diagnostic falls out of the table.** Before any training, an untrained model has no reason to prefer any class, so it should assign roughly 1/10 to each, giving a loss of −ln(0.1) = **2.303**. If your first printed training loss is close to 2.3, your output layer and loss are wired correctly. If it starts at 15, or at 0.4, something is wrong before you have even begun: wrong number of output units, labels misaligned, data unnormalised. It costs nothing to check and it catches real bugs.

### 6.3 What "sparse" means

`sparse_categorical_crossentropy` and `categorical_crossentropy` compute the same quantity. The only difference is the format of the labels.

| Loss | Label format for "class 3" | What the loss does |
|---|---|---|
| `categorical_crossentropy` | one-hot: [0,0,0,1,0,0,0,0,0,0] | dot product of the one-hot vector with log p, which selects element 3 |
| `sparse_categorical_crossentropy` | integer: 3 | indexes element 3 directly |

That is it. Multiplying by a vector of nine zeros and one 1 in order to pick out one element is a wasteful way to index an array, so the sparse version skips it. It saves memory (one integer per label instead of ten floats) and removes a preprocessing step.

Your notebook uses the sparse version because CIFAR-10 labels arrive as integers 0–9 (§2.1) and are never one-hot encoded. It is a data-format decision, not a modelling one. A classic bug is mismatching them, and it presents as a shape error rather than a silent wrong answer, which is a small mercy.

### 6.4 Gradient descent

Picture the loss as a surface. Every one of the 282,250 parameters is an axis, and the height at any point is the loss the model would achieve with those parameter values. Training is a search for a low point on a 282,250-dimensional landscape you cannot draw.

The gradient ∂L/∂w is the direction of **steepest increase**. So you step the other way:

> w ← w − η · ∂L/∂w

where η is the **learning rate**, the size of the step. That single line, applied to every parameter, repeated 471 times (Module 7), is your entire training procedure. Backpropagation is the chain rule organised efficiently so that all 282,250 partial derivatives are computed in one backward pass rather than one at a time.

The learning rate is the setting everything hinges on:

| η | What happens |
|---|---|
| Too small | Loss falls, very slowly. With only 471 updates you run out of budget before you arrive. |
| About right | Steady descent |
| Too large | You overshoot the minimum, the loss oscillates or diverges, and units can die (§5.4) |

There is no way to know the right value in advance. It depends on the architecture, the data scale, and the optimizer, which is exactly the problem the next section addresses.

### 6.5 Adam

Plain gradient descent uses one learning rate for every parameter and every step. Adam improves on that in two ways at once.

**Momentum, the first moment.** Instead of stepping along the current gradient, Adam keeps a running average of recent gradients and steps along that. Consistent directions accumulate and get faster; directions that flip sign from batch to batch cancel out. It smooths the path through the narrow-valley loss surfaces §2.3 described.

**Per-parameter adaptive scaling, the second moment.** Adam also keeps a running average of each gradient's squared magnitude and divides that parameter's step by its square root. A parameter that consistently receives large gradients gets small steps; a parameter receiving tiny gradients gets proportionally larger ones. Every parameter effectively gets its own learning rate, adjusted continuously.

There is also a **bias correction** term, because both running averages start at zero and are therefore biased toward zero early on. That detail matters more for you than for most projects: bias correction has its largest effect in the first few dozen steps, and your entire training run is 471 steps.

Why it is the sensible default: it works acceptably across a wide range of problems without tuning, its default learning rate of 0.001 is a reasonable starting point almost everywhere, and it is far more forgiving of input scaling than plain SGD (§2.3). For a time-boxed project where nobody is going to run a learning-rate sweep, "Adam at defaults" is the choice that gets you a working model on the first try.

### 6.6 The honest note about the learning rate

**Your group did not tune the learning rate.** Say so, and say it in the right way, because there are two ways to describe this and only one is accurate.

It is a **held-constant control variable**, not an oversight. Module 1's §1.2 list is what makes this an experiment: architecture, optimizer, epochs, batch size, data split and seed are all identical across the two arms, so any difference in outcome is attributable to the augmentation. The learning rate belongs on that list. It was 0.001 in the baseline and 0.001 in the augmented model, so it cannot possibly explain the −2.78 pp accuracy difference or the −2.01 pp gap difference (§R.1). Holding something constant is a design decision. Failing to mention it is the oversight.

What it does do is **bound the claim**. Your finding is about this architecture, trained with Adam at its default learning rate, for 3 epochs. A different learning rate would change how far the model gets in 471 updates, and §R.1 established that the budget is the binding constraint on everything. It is entirely possible that a higher learning rate would push the baseline further into the fitting regime where augmentation actually helps. You have no evidence either way, which is precisely why it is a limitation (Module 12) and a next step, not a result.

And it is a **separate experiment**, for the reason §1C.3 already gave: varying learning rate and augmentation together would confound them at n = 1 per cell. If you propose a learning-rate sweep, propose it as its own study.

### 6.7 What to put in the writeup

If you choose cross-entropy as your mathematical foundation: the formula L = −log(p_c); the loss table showing that confident-and-wrong is unbounded while confident-and-right is nearly free; the ln(10) = 2.303 initialisation check; and the one-sentence explanation that "sparse" refers to integer labels rather than one-hot. Then one paragraph on Adam, ending with the control-variable framing from §6.6. That combination covers the mathematical foundation line and feeds Module 12's limitations at the same time.

---

## Module 7: Training mechanics, epochs, batches and seeds ✅

Reproducibility is worth **20 points** on the instructor rubric, which makes this the highest-value-per-word module in the document. It is also the module where your own data has already taught you something the textbooks state too confidently.

### 7.1 Three words that get used interchangeably and should not be

| Term | Definition | Your project |
|---|---|---|
| **Batch** | The group of examples processed before one weight update | 64 images |
| **Iteration** (or step, or update) | One forward pass, one backward pass, one application of w ← w − η·∂L/∂w | 471 in total |
| **Epoch** | One complete pass over the entire training set | 3 |

The relationship is:

> iterations per epoch = ceil(training set size / batch size)
> total updates = iterations per epoch × epochs

The word people misuse is **epoch**, usually by treating it as a unit of learning. It is not. An epoch is a unit of *data exposure*. How much learning happens in one epoch depends entirely on how many updates it contains, which depends on the batch size. That is §7.3's point and it is the one worth internalising.

### 7.2 The arithmetic for your run

> 10,000 training images / 64 per batch = 156.25

You cannot have a quarter of a batch, so Keras runs **157** batches: 156 full batches of 64, which is 9,984 images, plus one final partial batch of the remaining 16.

> 157 batches × 3 epochs = **471 weight updates**

That is the entire training budget. Every one of your 282,250 parameters is adjusted 471 times, and then the model is finished and evaluated. Four hundred and seventy-one steps down a 282,250-dimensional surface.

Put that next to the wall clock from §R.1: about **8 seconds per run on CPU**. The whole ten-seed paired experiment across both conditions is a couple of minutes of compute. The smallness of the budget and the smallness of the runtime are the same fact, and between them they explain both why §R.1 found underfitting and why the multi-seed design that §1.8 called an aspirational next step turned out to be affordable as the actual experiment.

### 7.3 Batch size is a tradeoff, and it silently changes the update count

Hold the training set at 10,000 and the epochs at 3, and vary only the batch size:

| Batch size | Batches per epoch | Total updates in 3 epochs |
|---|---|---|
| 32 | 313 | 939 |
| **64** (yours) | **157** | **471** |
| 128 | 79 | 237 |
| 256 | 40 | 120 |

Read that table carefully, because it contains the trap. "3 epochs" sounds like a fixed amount of training. It is not. Doubling the batch size at fixed epochs **halves the number of learning steps**. If you ever compare two configurations at different batch sizes and equal epochs, you are comparing different amounts of training as well, and you will not be able to say which caused what. Batch size is on §1.2's controlled list for exactly this reason.

The genuine tradeoff, separate from the update-count effect:

| | Small batch | Large batch |
|---|---|---|
| **Gradient quality** | Noisy. Each gradient is estimated from few examples, so it is a poor approximation of the true gradient over the full dataset. | Smoother. Averaging over more examples reduces the estimate's variance. |
| **Updates per epoch** | Many | Few |
| **Hardware use** | Poor. Small matrices underuse parallel hardware. | Good, up to the point where memory runs out. |
| **Side effect** | The noise itself is mildly regularizing: it makes it harder to settle into sharp, brittle minima. | Less of that noise, so slightly more prone to memorising. |

64 is a conventional middle. It gives enough examples per gradient to be a reasonable estimate, uses hardware sensibly, and leaves a usable number of updates in a short run. It was not tuned, and like the learning rate it is a control variable, held identical across both arms.

### 7.4 What a seed fixes

`tf.random.set_seed(SEED)` sets the state of TensorFlow's random number generator. Everything downstream that draws from it becomes reproducible **within a single process**:

| A seed does fix | A seed does not fix |
|---|---|
| Weight initialisation for every layer | The order floating-point reductions are combined in |
| The shuffling order of the training data each epoch | Thread scheduling and CPU core count |
| Which random flip, rotation and zoom each image receives | Library versions, BLAS backend, hardware |
| Dropout masks (irrelevant here at rate 0.0) | Anything happening outside TensorFlow's generator |

Note also that your project uses **two** independent seeds, and keeping them apart matters (§2.2). `DATA_SEED = 42` controls which 12,000 training images and which 3,000 test images are drawn, and it never varies: every run in every condition on every seed trains on the same images. The run seed varies the weight initialisation and the shuffling. That separation is what makes the paired comparison a comparison of *conditions* rather than of datasets.

### 7.5 What a seed does not fix, with your own evidence

§1D.3 told you Colab GPU training is non-deterministic and blamed cuDNN kernel selection and GPU atomics. §R.6 corrected it: you ran on **CPU**, and it was still not reproducible.

The measurement, same configuration, same data, same seeds, two separate executions:

| Statistic | Value |
|---|---|
| Mean per-seed difference in test accuracy | **1.57 pp** |
| Maximum per-seed difference | **3.63 pp** |

The mechanism is floating-point non-associativity. In finite precision (a + b) + c and a + (b + c) can differ, because each intermediate result is rounded. Training is built from large parallel summations, and the order in which partial sums are combined depends on thread scheduling, which depends on what the machine was doing. Change the order, change the last bits, feed that into a gradient, into a weight update, into the next batch's gradients, and let it compound over 471 steps.

**The consequence you must carry into your limitations section** is §R.6's decomposition. Your 3.77 pp "noise floor" was described as seed variation. It is not purely that:

| Component | What varies | Measured how | Value |
|---|---|---|---|
| Between-seed | Initialisation and data order | Different seeds, same process | Reported as 3.77 pp, but confounded with the row below |
| Within-seed, across runs | Floating-point reduction order across processes | The **same** seed, run twice in separate processes | Mean 1.57 pp, max 3.63 pp |

The maximum within-seed difference (3.63 pp) is almost the whole five-seed spread (3.77 pp). Whatever else is true, that figure was never a pure seed effect. Correcting your own earlier framing costs nothing and is a sharper observation than most limitations sections contain.

For external calibration: Coakley & Gundersen (2026) report CIFAR-10 seed ranges of **1.48–1.71 pp** across 100 seeds on well-trained models, and Åkesson et al. (2024) show that seed variation alone can manufacture apparently significant differences between runs of an identical algorithm. Your spread is larger than theirs, which is what a 10,000-image subset at 51% accuracy after 3 epochs should produce. It is expected, not a mistake.

### 7.6 What the 20-point reproducibility criterion actually wants

Not bit-identical numbers. Nobody can deliver those across machines, and promising them would be the wrong claim. What you can deliver, and what earns the marks:

1. **The complete settings record.** Seed values, split sizes, image shape, class count, epochs, batch size, optimizer, loss, parameter count. §1D.4's `SETTINGS RECORD` block prints all of it.
2. **The code that regenerates the result**, with outputs retained in the saved notebook (§1D.1).
3. **The distribution, not a point.** Report the mean, the SD and the number of seeds, and list the seeds themselves. §R.6's phrasing is the one to use: reproducibility of the *distribution* rather than of the digits.
4. **An explicit statement of the non-determinism you measured**, with the 1.57 pp / 3.63 pp figures. Disclosing a limitation you quantified reads as control; discovering it in the viva does not.
5. **The symmetry fix.** §1C.5(b) flagged that the baseline cell calls `clear_session()` and `set_seed()` before building while the augmented cell does not. Fix it, and note the fix. Without it the two arms do not even start from the same random state, which is also part of why the pairing only achieved r ≈ 0.19 (§R.7).

Item 5 deserves one more beat, because §R.7 found something subtler. Even with both cells re-seeded, "same seed" still did not mean "same initial weights," since inserting three augmentation layers ahead of the first convolution shifts the position in the random stream from which the convolutional and dense kernels are drawn. The genuine fix is to build the model once, save the initial weights, and load those identical weights into both arms at every seed. That is a specific, cheap, well-motivated next step, and it is far better than "future work should improve reproducibility."

---

## Module 8: Overfitting and the generalization gap ✅

Augmentation is a regularizer, and regularization is a treatment for overfitting. So the question "was this model overfitting?" is not a diagnostic detail in your project. It is the question that determines whether the intervention had anything to do at all, and §R.1 already answered it in a way that reframed the entire experiment.

### 8.1 Overfitting is a trajectory, not a level

The definition that matters:

> **Overfitting is validation performance getting worse while training performance keeps getting better.**

Every word is load-bearing. It is about **change over time**, about **two curves moving in opposite directions**. You cannot diagnose it from a single number, from a single epoch, or from the size of any difference. You need the shape of the curves.

The model has stopped learning things that transfer and started learning things specific to the 10,000 images it was shown: the exact noise pattern in one photograph, an incidental correlation between a background colour and a label. Those help on the training set by construction and hurt everywhere else.

### 8.2 A gap is not overfitting

The **generalization gap** is training accuracy minus validation accuracy. It is a difference in *level* at a moment in time, and it is almost always somewhat positive, for a reason that has nothing to do with pathology: the model updated its weights on the training data and did not update them on the validation data. Some advantage is guaranteed.

| | Gap | Overfitting |
|---|---|---|
| What it is | A difference in level | A divergence in direction over time |
| Measured from | One point in time | The shape of the curves |
| Normal value | Somewhat positive, always | Not applicable, it is a behaviour |
| Diagnostic on its own | **No** | Yes |

So "the model has a 5 pp gap" is not a finding. A 5 pp gap that is stable across epochs while validation accuracy climbs is a healthy model. The same 5 pp gap opening up because validation accuracy has turned downward is overfitting. Identical number, opposite conclusions.

This is §R.2's rule in another costume: never report a difference without reporting both of its terms, and never report a level without reporting its trajectory.

### 8.3 The three regimes

| Regime | Training curve | Validation curve | Gap | What it means | What helps |
|---|---|---|---|---|---|
| **Underfitting** | Still rising | Still rising, tracking training | Small | The model has not extracted what is available yet: too little capacity, too little training, or too strong a constraint | More epochs, more capacity, less regularization, higher learning rate |
| **Healthy** | Rising, flattening | Rising, flattening near its peak | Moderate and stable | The model has learned what generalizes and has not started on what does not | Stop here |
| **Overfitting** | Still rising, often toward 100% | Flat, then falling | Growing | The model is now memorising | Regularization: augmentation, dropout, weight decay, early stopping, more data |

The transition is gradual, and the useful early-warning sign is that **validation loss turns upward before validation accuracy turns downward**. Loss is sensitive to confidence, so it registers the model becoming more certain about its mistakes before enough predictions flip to move the accuracy count. If you only plot accuracy you will see overfitting late. Plot both.

### 8.4 Your model's diagnosis

Three pieces of evidence from §R.1, and they agree:

**Validation loss fell monotonically through epoch 3, in all five baseline runs.** This is the decisive one. Overfitting has a signature, validation loss turning back up while training loss keeps falling, and nothing resembling it appears anywhere in the curves. The model had not reached the point where further training would start to hurt.

**Validation accuracy sat above training accuracy for the first two epochs.** This looks impossible and it is an artifact, worth understanding properly because it recurs in every Keras project you will ever run. The training accuracy `fit()` prints is a **running average accumulated across the whole epoch**, while the weights were still poor at the start and improving throughout. The validation accuracy is computed **once, at the end of the epoch, with the final weights**. You are comparing an average over a moving target against a single snapshot taken at that target's best moment. Early in training, when weights improve rapidly within an epoch, the snapshot wins. It is not evidence of anything about generalization, and the fix is the one §1B.9 already gave: compute every gap from `evaluate()` on clean data with final weights, for both models.

**The clean-evaluated baseline gap was 5.05 pp**, from a clean training accuracy of 56.18% and validation of 51.13%. Modest, and by §8.2 not diagnostic on its own, but consistent with the curves.

**Conclusion: the baseline was underfitting at 3 epochs, not overfitting.** §1.6 predicted this before the run, which is worth stating in the writeup, because a prediction on record is worth more than the same sentence written afterwards.

### 8.5 Why that reframes the whole experiment

Augmentation is regularization. Regularization helps a model that is fitting *too much*. Your model was fitting *too little*, so the intervention had no overfitting to remove and simply made an already-hard task harder. Lin et al. (2024) give the peer-reviewed version of this argument: augmentation acts as an implicit regularizer whose effect depends on the fitting regime, and it can help or hurt accordingly.

Your numbers show exactly that, and §R.2's decomposition is the part to keep in front of you:

| Term | Baseline | Augmented | Change |
|---|---|---|---|
| Clean train accuracy | 0.5428 | 0.4993 | −4.35 pp |
| Clean validation accuracy | 0.4986 | 0.4752 | −2.34 pp |
| **Gap** | **0.0442** | **0.0241** | **−2.01 pp** |

The gap narrowed and **both terms fell**. Nothing generalised better. The whole model slid down and the top slid faster. "Augmentation significantly reduced the generalization gap, p = 0.0012" is a true sentence that, on its own, carries a false finding, and this is the clearest illustration in the entire project of why levels must be reported next to differences.

### 8.6 How to diagnose which regime you are in

A working procedure, in order:

1. **Plot training and validation loss on the same axes, every epoch.** Loss first, accuracy second.
2. **Look at the validation curve's direction, not the distance between the curves.** Still falling means you have not finished learning. Turned upward means you have gone too far.
3. **Only then look at the gap**, and read it as a level alongside the trajectory rather than as a verdict.
4. **Compute both terms with `evaluate()` on clean data and final weights**, so the numbers are comparable across models (§1B.9). This is doubly necessary for the augmented model, whose `fit()` training accuracy is measured on augmented images while its validation accuracy is measured on clean ones (§9.5).
5. **Ask whether the diagnosis is even stable.** With 3 epochs you have three points per curve, which is very little to read a trend from, and with a 3.77 pp noise floor (§7.5) the last epoch's numbers wobble. State the diagnosis with the confidence three points support, which is: consistent across five runs, monotone in the right direction, and short of definitive.

### 8.7 What would have produced overfitting

Useful for your next-steps section, and each is a specific, testable proposal:

- **More epochs.** The most direct route. At 471 updates the model has not had the opportunity. Twenty or thirty epochs on this subset would very likely open a real gap, and that is the run in which augmentation would have something to regularize.
- **Less data.** 10,000 images is already a subset. 2,000 would overfit faster.
- **More capacity.** The dense head already holds 262,208 parameters against 10,000 images (§4.5), which is ample room to memorise. It has simply not had time to use it.

Note the shape of that list. Every route runs through the **training budget**, which is the setting the assignment holds fixed. That is not a complaint; it is the honest boundary on your finding, and it is the sentence Module 12 wants: your result is about augmentation *under a 3-epoch budget*, and it says nothing about augmentation under a budget long enough for overfitting to appear.

One more detail that fits here. Your architecture contains `Dropout(0.0)` (§4.7), a regularization layer switched off. On a model that is underfitting, a disabled regularizer is the right setting. It is also a small piece of evidence that the starter notebook was designed as a fast teaching example rather than as a model tuned for accuracy, which is exactly how you should describe it.

---

## Module 9: Augmentation in depth ✅

This is your assigned intervention, so this module has to be the most precise in the document. Modules 1 and 2 built the framing; §R.2 and §R.7 reported what happened. Here we go transform by transform and mechanism by mechanism, because "we applied standard augmentation" is not an explanation and a grader will ask for one.

### 9.1 The framing, restated

> **Augmentation is not really "more data." It is a way of injecting invariances you already know are true into the training process.**

Everything else follows from that sentence, including the rule for choosing transforms: **apply a transform only if the invariance it encodes genuinely holds in your domain.** A transform that preserves the label teaches the model something true. A transform that does not preserve the label teaches it something false, and the model has no way to tell which it received.

There is a second reason the "more data" phrasing misleads, and it is specific to your setup. Augmentation does **not** increase the number of weight updates. You get 471 either way (§7.2). It changes what each of those 471 updates sees, not how many there are. So augmentation cannot compensate for a short budget; if anything it competes with it, because a harder, more variable training signal needs *more* steps to converge, not the same number. That is a clean mechanistic explanation for your negative accuracy result, and it is stronger than the vague "augmentation needs longer to pay off."

### 9.2 Transform by transform, for CIFAR-10 at 32 × 32

| Transform | Is the invariance true for CIFAR-10? | Pixel cost | Verdict |
|---|---|---|---|
| **Horizontal flip** | Yes. A mirrored car is a car, a mirrored horse is a horse. Nothing in these ten classes has a meaningful left/right handedness. | **None.** A horizontal flip is an exact index reversal. No interpolation, no invented pixels, no loss. | The safe transform, at any resolution |
| **Vertical flip** | **No.** Upside-down airplanes, cars, ships and horses essentially do not occur in these photographs. | None, technically | Do not use. You would be teaching the model to expect something the world never shows it |
| **Small rotation** | Yes, in principle. Photographs are handheld and slightly tilted all the time. | **Real.** Rotated pixels land between grid positions, so every output pixel is interpolated from its neighbours, and the corners have no source pixels at all. | Legitimate but not free, and the cost is worst at low resolution |
| **Zoom** | Partly. Objects genuinely appear at different scales. | **Real.** Zooming in crops and resamples; zooming out pads a frame that had little context to begin with. Interpolation again. | Same caveat as rotation |

The asymmetry between row 1 and rows 3 and 4 is the heart of the argument. **Horizontal flip is pixel-lossless; rotation and zoom are not.** On a 224 × 224 photograph the interpolation blur from an 18 degree rotation is imperceptible and the lost corners are thin slivers. On 32 × 32, where one image holds 1,024 pixels per channel and an animal may span twenty of them across (§2.6), you are smearing the few edges the model has to work with and discarding a meaningful fraction of a small object.

Alomar et al. (2023) is the closest published setting to yours: reduced CIFAR-10 subsets at the same resolution. They describe whole-image rotation producing boundary artifacts that do not reflect the original data, and a loss of significant pixel information, and they warn directly that excessive augmentation can degrade performance. Note that Keras's `RandomRotation` does not leave black corners by default; it fills them by reflecting the image across its border, which avoids the black patches but manufactures mirrored content that was never in the photograph. Either way you are adding pixels that carry no evidence about the label. (Worth confirming the default fill mode in your own Keras version, since that is a version-level detail.)

### 9.3 What `RandomRotation(0.05)` actually means

This is the single most commonly misread argument in the whole notebook, and getting it right is a cheap way to look like you read the documentation.

The argument is **a fraction of a full turn**, not degrees and not radians. It is applied symmetrically, so a scalar factor f means a uniform random rotation in the range −f to +f of 2π.

> 0.05 × 360° = 18°, so `RandomRotation(0.05)` rotates by up to **±18°**
> 0.10 × 360° = 36°, so `RandomRotation(0.10)` rotates by up to **±36°**

Eighteen degrees is a plausible camera tilt. Thirty-six degrees is not a tilt; it is a substantially reoriented image, and at 32 × 32 the corner loss and interpolation blur roughly double with it. If anyone in your group assumed 0.05 meant five degrees, the moderate setting is more aggressive than they think, and the stronger setting is considerably more so.

`RandomZoom(0.10)` follows the same fractional convention: up to a 10% change in scale.

### 9.4 The intensity options, and what your evidence says

§1C.4 left this as a live decision, and it is worth recording where it landed and why:

| Option | Transforms | Angle range |
|---|---|---|
| Conservative | horizontal flip only | none |
| **Moderate** (the notebook default, what you ran) | flip + `RandomRotation(0.05)` + `RandomZoom(0.10)` | ±18° |
| Stronger | flip + `RandomRotation(0.10)` + `RandomZoom(0.15)` | ±36° |

Xu et al. (2023) give the cleanest vocabulary for why an optimum exists at all. Augmentation works by expanding the **vicinity distribution** around each training point: the neighbourhood of inputs that are treated as sharing the original's label. Mild transforms stay inside that neighbourhood, so the label remains true and the model learns a genuine invariance. Severe transforms push the sample outside it, and the label becomes a lie: a rotated, zoomed, interpolated 32 × 32 bird can simply stop containing enough evidence of "bird," while still being trained on as one.

That is the mechanism behind the tradeoff §1.5 named as **added invariance versus destroyed signal**, and it is why intensity is not monotonically good. Ottoni et al. (2023) supply the complementary point: they treat individual transforms as separately estimable factors and find that selective tuning can beat the conventional "apply everything" configuration. Stacking flip, rotation and zoom together is a decision that needs evidence, not a default.

### 9.5 The mode distinction, and why it contaminates the gap

Your augmentation is not applied to the dataset. It is a stack of Keras preprocessing layers sitting at the top of the augmented model, ahead of the first convolution. Layers of this kind behave differently depending on the mode they are called in:

| Mode | When | Augmentation layers |
|---|---|---|
| **Training** | inside `model.fit()` | **Active.** Every image is randomly flipped, rotated and zoomed, differently each epoch |
| **Inference** | `model.evaluate()`, `model.predict()`, and the per-epoch validation pass | **Inactive.** Images pass straight through, untouched |

**That single fact is why the augmented model's `fit()` training accuracy is measured on harder images than its validation accuracy.** The baseline sees clean images in both modes. The augmented model sees distorted images while training and clean images while being validated. So `train_acc − val_acc` read straight off the Keras output is not an apples-to-apples comparison between your two models: the augmented model's gap looks smaller partly because its training number was pushed down by a harder task, which is a **measurement artifact**, not evidence of better generalization.

The fix is one line per model, and §1C.5(a) had you capture it at baseline so you would not have to re-run:

```python
train_loss, train_acc = model_aug.evaluate(x_train, y_train, verbose=0)
```

`evaluate()` runs in inference mode, so the augmentation layers are skipped and the augmented model is scored on the same clean images as the baseline. It also sidesteps the running-average distortion from §8.4, since `evaluate()` is a single clean pass with the final weights. Every gap in §R.1's tables is computed this way, which is why they are trustworthy.

The mode distinction also gives you your **manipulation check** (§1B.10). `fit()`'s final training accuracy fell from 0.5120 to 0.4744, a drop of 3.76 pp. That is the augmentation demonstrably firing. Report it as a methods-section diagnostic, not as a finding: lower training accuracy is also what a broken model looks like, and its value is that it licenses everything else you say.

### 9.6 What actually happened to your classes

The aggregate result is in §R.1: accuracy −2.78 pp (p = 0.006), gap −2.01 pp (p = 0.0012), with both gap terms falling. The per-class picture is more informative:

| Class | Baseline recall | Augmented recall | Change |
|---|---|---|---|
| bird | 0.252 | 0.172 | −8.0 pp, roughly a third of the recall it had |
| deer | 0.380 | 0.280 | −10.0 pp |
| airplane | 0.646 | 0.549 | −9.7 pp |
| automobile | 0.643 | **0.705** | **+6.2 pp** |

Automobile is the trap, and §R.2 dismantled it: its **precision fell from 0.624 to 0.545** over the same comparison. The augmented model simply predicts "automobile" more often, so it catches more real automobiles and is wrong more often when it says the word. That is a shifted decision boundary, not improved recognition. Module 11 reads the same pattern off the confusion matrix.

Bird's fall is consistent with the resolution mechanism: a small, thin-featured class with little margin for interpolation blur, losing the most under the transform that interpolates. Phrase it carefully, though. **Consistent with is not evidence for**, because you ran one intensity and no sweep. That gap between what you observed and what you can attribute is precisely what makes an intensity sweep the obvious next step, and §1C.3 already explains why it is a separate experiment rather than something to bolt onto this one.

### 9.7 The sentences that survive scrutiny

- Under a 3-epoch budget on a 10,000-image CIFAR-10 subset, flip + rotation 0.05 + zoom 0.10 significantly reduced test accuracy (−2.78 pp, p = 0.006) and significantly narrowed the train minus validation gap (−2.01 pp, p = 0.0012), with both gap terms falling.
- The augmentation demonstrably fired: `fit()` training accuracy dropped 3.76 pp, and all gaps were recomputed with `evaluate()` in inference mode so the two models were scored on identical clean images.
- The transforms were chosen for whether their invariance holds on CIFAR-10, not for being conventional: horizontal flip is exact and label-preserving; vertical flip was excluded as a false invariance; rotation at 0.05 is ±18°, chosen as a plausible camera tilt rather than a reorientation.
- What this does **not** license: any claim that augmentation hurts CNNs, any claim about a different intensity, and any claim about a longer budget.

---

## Module 10: Metrics, accuracy, precision, recall and macro ✅

Your assignment requires accuracy, precision and recall. Reporting three numbers is easy. Explaining what their *combination* tells you about the model's behaviour is where the marks are, and your truck class is a small, complete case study in exactly that.

### 10.1 The four counts everything is built from

Every one of these metrics is assembled from four counts. In a ten-class problem you compute them **one class at a time**, treating that class as positive and the other nine as negative. For "truck":

| | Model said truck | Model said something else |
|---|---|---|
| **Actually a truck** | **TP** (true positive) | **FN** (false negative), a truck the model missed |
| **Actually not a truck** | **FP** (false positive), something called a truck that was not | **TN** (true negative) |

Two of these are errors and they are not interchangeable. An **FN** is a miss: the truck was there and the model did not find it. An **FP** is a false alarm: the model shouted truck at something else. Which of the two is worse depends entirely on the application, which is why a single accuracy number can never be the whole report.

### 10.2 Accuracy

> Accuracy = (TP + TN) / (TP + TN + FP + FN)

For a multiclass problem it is easier to read as: **the number of correct predictions divided by the total**, which on a confusion matrix is the sum of the diagonal divided by the sum of everything.

Your baseline: **51.35%**, SD 1.62 pp across five seeds (§R.1). On 3,000 test images that is roughly 1,541 correct.

Accuracy is a meaningful headline here for one specific reason: **your classes are balanced** (§2.1). On imbalanced data it stops being meaningful, and the standard example is worth keeping in mind: a fraud detector that predicts "not fraud" for every transaction scores 99% on a 1-in-100 problem and is worthless. Your data has roughly 300 test images per class, so chance is 10% and 51.35% is genuinely five times chance.

What accuracy cannot tell you is **which** classes it got right, and that is the entire reason the other metrics exist.

### 10.3 Precision

> Precision = TP / (TP + FP)

Plain reading: **when the model says "truck," how often is it right?** The denominator is everything the model *called* a truck. Precision is a property of the model's claims.

### 10.4 Recall

> Recall = TP / (TP + FN)

Plain reading: **of all the actual trucks, how many did the model find?** The denominator is everything that *was* a truck. Recall is a property of the model's coverage.

The two are easy to confuse and the fix is to look at the denominator. Precision divides by what the model *said*; recall divides by what *was true*. They also trade off: a model that predicts a class more freely will catch more of them (recall up) and be wrong more often when it does (precision down). That is not a theoretical point in your project, it is what automobile did under augmentation.

### 10.5 F1

> F1 = 2 × (precision × recall) / (precision + recall)

The **harmonic** mean, not the arithmetic mean, and the choice is deliberate. The harmonic mean is dominated by the smaller of the two numbers, so it refuses to reward a model that games one metric. A classifier that predicts "truck" for every single image gets recall 1.00 and precision 0.10, an arithmetic mean of 0.55 which looks respectable, and an F1 of 0.18 which does not. F1 is the single number to use when you need one and both errors matter.

### 10.6 The truck example, worked

Baseline truck: **recall 0.64, precision 0.51**, on approximately 300 test trucks (§2.2 notes that a random subset gives roughly, not exactly, 300 per class, with supports running 275–313). Approximate arithmetic:

| Quantity | Working | Value |
|---|---|---|
| True trucks | class support | ≈ 300 |
| TP | 0.64 × 300 | ≈ 192 |
| FN (trucks missed) | 300 − 192 | ≈ 108 |
| Total truck predictions | TP / precision = 192 / 0.51 | ≈ **377** |
| FP (non-trucks called truck) | 377 − 192 | ≈ **185** |
| F1 | 2 × 0.51 × 0.64 / (0.51 + 0.64) | **0.57** |

Now read it. **The model said "truck" about 377 times when only about 300 trucks exist.** It over-predicts the class by roughly a quarter. Of everything it labelled truck, about 185 were something else, most plausibly automobiles, which share every low-level cue trucks have: straight edges, road context, manufactured surfaces.

That is a real behavioural finding, and notice that neither number produces it alone. Recall 0.64 was one of your three best classes and looks like a success. Precision 0.51 says half of its truck calls were wrong. **Truck is not a class the model understands well; it is a class the model reaches for.** High recall with low precision is the signature of over-prediction, and §11.6 shows you the same thing geometrically on the confusion matrix.

Contrast with frog, also at 0.64 recall. Same recall, and until you look at the precision you cannot tell whether that reflects genuine competence or the same over-prediction. **Always report precision and recall together.** Reporting one is §R.2's error, since each is a ratio hiding its terms.

### 10.7 Macro, micro and weighted

You have ten per-class values for each metric and one line in a results table. Three ways to collapse them:

| Averaging | How it works | Weights classes by |
|---|---|---|
| **Macro** | Compute the metric per class, then take the plain mean of the ten values | Equally |
| **Micro** | Pool TP, FP and FN across all classes, then compute the metric once from the pooled counts | Effectively by frequency, since larger classes contribute more counts |
| **Weighted** | Compute per class, then average weighted by each class's support | Frequency |

**Macro** gives your worst class the same vote as your best. Bird, at 0.32 recall, counts exactly as much as truck at 0.64. That makes it the right choice when every class matters equally, which for a ten-way balanced benchmark it does. It is also the sensitive one: macro recall drops as soon as any single class collapses, which is what you want a summary to do.

**Micro** has a property worth knowing because it explains a common confusion. In a single-label multiclass problem where every example gets exactly one prediction, every error is simultaneously a false positive for one class and a false negative for another. The pooled counts therefore make micro precision, micro recall, micro F1 and accuracy all **the same number**. If someone reports "micro F1" on your problem, they have reported accuracy.

**Weighted** matters on imbalanced data, where it stops a tiny class from dragging the average down out of proportion to its importance. On your data it will sit very close to macro, since the supports are all near 300.

### 10.8 Why comparing accuracy with macro recall is informative

On a **perfectly** balanced test set, accuracy and macro recall are mathematically identical. Accuracy is total correct over total; macro recall is the mean of per-class correct-over-support; when every support is the same number, those are the same calculation.

Your test set is approximately balanced, not exactly (supports 275–313, §2.2), so they come out close but not equal. §2.1 records the check: baseline accuracy **51.35%** against macro recall **51.37%**. A difference of 0.02 pp.

That is not a coincidence and it is not padding. It is a **consistency check** that tells you two useful things at once: your test set really is close to balanced, and no class is so tiny that its recall is distorting the average. If those two numbers had diverged by several points on data you believed was balanced, something would be wrong, either with the balance or with how the metrics were computed. Cheap to run, worth a sentence, and it demonstrates that you know why they should agree rather than just observing that they do.

The pair that genuinely can diverge on balanced data is **macro precision against macro recall**. They differ when the model distributes its predictions unevenly across classes: over-predicting some, under-predicting others. That is exactly the truck and automobile behaviour, and it is the aggregate signal that sends you to the confusion matrix.

### 10.9 What to report

| Metric | Why it is in your table |
|---|---|
| Test accuracy | The headline, meaningful because the classes are balanced, and H-A's quantity |
| Macro precision | Required, and the counterweight that catches over-prediction |
| Macro recall | Required, and the consistency check against accuracy |
| Per-class precision, recall and F1 | Where the actual behaviour lives; `classification_report` prints all of it (§1D.4) |
| Generalization gap | H-B's quantity, computed from `evaluate()` on clean data for both arms |

And the caution §1B.8 attached to per-class work, restated here because it belongs with the metrics: at roughly 300 test images per class, **one image is about 0.33 pp of recall**. A 3 pp per-class difference is about nine images. Report per-class changes as observations with that scale attached, not as findings with confidence attached.

---

## Module 11: Reading a confusion matrix ✅

The confusion matrix is a required visual, and it is the only figure in your submission that shows the model's actual behaviour rather than a summary of it. Every metric in Module 10 is computable from it. It is also the figure most often printed and then not read.

### 11.1 The layout convention

`sklearn.metrics.confusion_matrix(y_true, y_pred)` produces a 10 × 10 grid with the convention:

> **Rows are actual classes. Columns are predicted classes. The diagonal is correct.**

Cell C[i][j] is the number of images whose true class was i and which the model called j. The diagonal, where i = j, holds the correct predictions. Everything off the diagonal is an error, and *which* off-diagonal cell it lands in tells you what kind.

Two practical points before you read anything. First, the convention is not universal: some tools transpose it. State yours in the caption. Second, §1D.4's note about `annot=True, fmt='d'` matters here, because the notebook's existing heatmap call omits both and produces an unlabelled colour grid from which a reader cannot extract a single number. The rubric asks for readable visuals. Print the counts.

### 11.2 Reading a row

A row is **one true class, distributed across the model's guesses**. The row for "bird" tells you what happened to the roughly 300 actual birds: how many were called bird, how many cat, how many airplane.

> Row sum = the class's support, roughly 300 here
> Diagonal cell / row sum = **recall** for that class

So a row with a dim diagonal and bright cells elsewhere is a class the model keeps missing, and the bright cells name what it mistakes the class *for*.

### 11.3 Reading a column

A column is **one predicted label, and everything that received it**. The column for "truck" tells you what the model called a truck.

> Column sum = the number of times the model predicted that class
> Diagonal cell / column sum = **precision** for that class

The comparison that makes the matrix worth reading is between a class's row sum and its column sum. Row sum is how many there really were. Column sum is how many the model claimed. A column heavier than its row is **over-prediction**; a column lighter than its row is under-prediction. That one comparison, done by eye across ten classes, tells you more about the model's character than the accuracy figure does.

### 11.4 What a bright off-diagonal cell means

A large value at C[i][j] means class i is being systematically absorbed into class j. Not randomly scattered errors: a specific, repeated confusion. The interesting question is always what is in cell C[j][i], the mirror.

| Pattern | What it looks like | What it usually means | What to do about it |
|---|---|---|---|
| **Symmetric confusion** | C[i][j] and C[j][i] are both large and roughly comparable | The two classes are genuinely visually similar at this resolution, and the model cannot separate them in either direction. Cat and dog is the archetype: similar size, similar fur texture, similar poses, similar backgrounds, 32 × 32. | More resolution, more capacity, more training. This is a representational limit, not a bias. |
| **Asymmetric confusion** | C[i][j] is large and C[j][i] is small | Class j is **absorbing** class i. The model has a prediction bias toward j: it reaches for that label when uncertain. | Look at j's precision. If it has fallen, the model is over-predicting j and the "improvement" in j's recall is redistribution, not learning. |

The distinction matters because the two have different remedies and different stories. Symmetric confusion is the model honestly failing to tell two things apart. Asymmetric confusion is the model developing a habit.

### 11.5 The truck column, geometrically

§10.6 computed truck's counts from its precision and recall. Read them back onto the matrix:

> Truck row sum ≈ 300, of which about 192 land on the diagonal (recall 0.64)
> Truck **column** sum ≈ 377, of which the same 192 are on the diagonal (precision 0.51)

The column is roughly a quarter heavier than the row. About 185 non-trucks sit in that column, contributed by other rows, most plausibly the automobile row given how much the two classes share at 32 × 32. You would confirm that by looking at C[automobile][truck] specifically, and then at its mirror C[truck][automobile] to decide whether it is symmetric confusion or absorption.

This is the point of the figure. "Truck recall 0.64" is a number. "The model claims truck about 377 times when 300 exist, and most of the surplus comes from one neighbouring class" is a description of behaviour, and it came from reading a column against a row.

### 11.6 What your two matrices show

**The baseline splits cleanly along a vehicle/animal line.**

| Weakest baseline recall | | Strongest baseline recall | |
|---|---|---|---|
| bird | 0.32 | truck | 0.64 (precision only 0.51) |
| cat | 0.34 | frog | 0.64 |
| deer | 0.37 | airplane | 0.62 |

The bottom three are animals; two of the top three are vehicles. That is not arbitrary. Vehicles are rigid, high-contrast, straight-edged objects that photograph against consistent backgrounds. Animals are deformable, textured, and appear in poses and settings that vary enormously, and at 32 × 32 a deer and a horse and a dog reduce to a brown quadruped shape against foliage. Expect the animal classes to cluster at the bottom of the report and to confuse with each other in the matrix.

Note the caveat sitting inside that table: **truck's precision is only 0.51**, so its top-three recall is bought by over-predicting. That is the §10.6 finding, and it means the vehicle/animal split is slightly less flattering to the vehicles than the recall column alone suggests.

**Augmentation made the split worse, not better.** From §9.6:

| Class | Recall change | Precision |
|---|---|---|
| bird | 0.252 → 0.172 | |
| deer | 0.380 → 0.280 | |
| airplane | 0.646 → 0.549 | |
| automobile | 0.643 → **0.705** | 0.624 → **0.545** |

Three classes lost substantial recall, one gained, and the one that gained lost precision at the same time. That combination has a name and a mechanism.

### 11.7 The signature of a model retreating to easy classes

Here is the reading, and it is the most interesting sentence you can write about this experiment.

Augmentation made every training image harder and gave the model no extra updates to cope with it (§9.1). A model under that pressure becomes less certain, and a less certain classifier does not distribute its uncertainty evenly. It falls back on whatever cues survive the distortion. Rotation, zoom and interpolation destroy fine texture and thin structures first, which is exactly what bird and deer are made of, while leaving gross shape and colour blocks largely intact, which is what automobile is made of.

So predictions migrate. The classes whose evidence was destroyed lose recall. The classes whose evidence survived absorb the surplus, gaining recall while **losing precision**, because much of what they gained was not theirs. That is the automobile row and column exactly: recall 0.643 → 0.705, precision 0.624 → 0.545.

> **This is the signature of a model retreating to easy, distinctive classes.** Total accuracy falls, and the composition of the remaining accuracy shifts toward whichever classes are cheapest to identify.

On the matrix it appears as the automobile **column** growing while several animal **rows** thin out along their diagonals. On the metrics it appears as macro recall and macro precision diverging (§10.8). Both views describe the same event.

The honest qualifier, because §9.6 already flagged it: this is **consistent with** the resolution mechanism, not evidence for it. You ran one intensity and no sweep, so you cannot separate "interpolation destroyed fine texture" from other explanations. And at roughly 300 images per class, automobile's 6.2 pp gain is about 19 images and bird's 8.0 pp loss is about 24. Real, repeated across ten seeds at the aggregate level, and small in absolute terms. Report the mechanism as an interpretation, clearly labelled as **exploratory** in §1C.2's sense, since it was formed from the data rather than tested against it.

### 11.8 Reading procedure

1. **Print the counts.** `annot=True, fmt='d'` (§1D.4).
2. **Scan the diagonal** for the weakest and strongest classes. That is per-class recall by eye.
3. **Compare each column sum with its row sum.** Heavier column means over-prediction, and go check that class's precision.
4. **Find the brightest off-diagonal cells** and look at each one's mirror. Symmetric or asymmetric (§11.4)?
5. **Ask whether the confusions make physical sense** at 32 × 32. Cat with dog, automobile with truck, deer with horse: those are believable. A bright ship-with-frog cell would suggest a bug rather than a visual similarity.
6. **Convert every difference into images** before you interpret it. At ~300 per class, 1 pp ≈ 3 images.
7. **Compare the two matrices** for the direction of migration, not for individual cells.

On that last point: plotting baseline and augmented side by side is the right presentation choice. A difference matrix (augmented minus baseline) looks appealing and is harder to read honestly, because a cell can change by three images and produce a vivid colour. If you show one, show the raw pair next to it.

---

## Module 12: Limitations and claim boundaries ✅

This is explicitly graded, it is the section most often written in the last twenty minutes, and it is the one where your project is unusually strong. You have measured your own noise floor, quantified your own pairing failure, and declined to substitute a more flattering replication. Most submissions cannot say any of that. This module is about turning it into the right sentences.

### 12.1 Two kinds of validity

| | **Internal validity** | **External validity** |
|---|---|---|
| The question | Did the comparison isolate what it claims to isolate? | Does the result transfer beyond the conditions you ran? |
| Threatened by | Confounds, measurement asymmetry, leakage, uncontrolled variation | Narrow settings, one dataset, one architecture, one budget |
| Your version | Are the two arms genuinely identical except for augmentation, and were both scored the same way? | Does this say anything about augmentation outside 32 × 32 CIFAR-10 at 3 epochs? |
| Fixed by | Design and controls | Replication across conditions, which you cannot do here |

Keep them separate, because they fail independently and a submission can be strong on one and weak on the other. **Yours is strong internally and narrow externally**, and saying so precisely is better than hedging everything uniformly.

### 12.2 Internal validity: what your design actually controlled

The controls are real and worth listing plainly. Both arms used the same architecture, optimizer, learning rate, epochs, batch size, loss and data split. `DATA_SEED = 42` meant they trained on the identical 10,000 images, not merely on comparable samples (§2.2). The comparison ran across ten seeds with both conditions at every seed. Test data was never used to make a decision; §1C.6's choice of leading hypothesis was made from validation curves (§2.4). Hypotheses were committed in a markdown cell above the training cells before the augmented run (§1C.2).

And now the four internal threats you found yourself. This list is the strongest part of your limitations section:

| Threat | What it is | Status |
|---|---|---|
| **Measurement asymmetry** | The augmented model's `fit()` training accuracy is measured on augmented images and its validation accuracy on clean ones, so the raw gap is not comparable across arms (§9.5) | **Fixed.** All gaps recomputed with `evaluate()` in inference mode, both arms |
| **Initialisation asymmetry** | The baseline cell re-seeds before building and the augmented cell did not, so the arms did not start from the same random state (§1C.5b) | Fix and disclose |
| **Weak pairing** | The design assumed same-seed pairing would cancel seed-level luck. Measured correlation between arms was **r ≈ 0.19**, so almost nothing cancelled (§R.7) | **Disclosed.** Does not invalidate the paired t-test; it means the test ran near unpaired efficiency and rejected anyway |
| **Cross-process non-determinism** | Same seed, same config, two executions: mean 1.57 pp, max 3.63 pp difference (§R.6, §7.5) | Disclosed, with the noise-floor decomposition |

Notice how the third and fourth read. Weak pairing means your test had **less** power than the design promised, and both nulls fell regardless. A finding that survives a design operating below its intended efficiency is more robust than one that needed the full design to squeak through. You can only say that because you computed r instead of assuming it.

### 12.3 External validity: the boundaries

Your result holds for: **one architecture** (this two-block CNN, 282,250 parameters, 93% of them in the dense head); **one budget** (3 epochs, 471 weight updates, about 8 seconds); **one dataset and resolution** (a 10,000-image CIFAR-10 subset at 32 × 32); **one augmentation intensity** (flip + rotation 0.05 + zoom 0.10, so ±18°); and **one optimizer setting** (Adam at its default learning rate, untuned).

Each of those is a dimension you did not vary, and therefore a dimension along which you have no information. Three deserve particular emphasis:

- **The budget is the binding constraint.** §R.1 established the model was underfitting. Augmentation is a treatment for overfitting. Your finding is at least partly a finding about applying the right tool in the wrong regime (Lin et al., 2024), and it says very little about what a 30-epoch run would show.
- **The resolution drives the mechanism.** §2.6 and §9.2 argue that rotation and zoom cost more at 32 × 32 than they would at 224 × 224. How augmentation behaves at higher resolution is a different question your data does not address.
- **One intensity is not a dose-response curve.** Xu et al. (2023) frame augmentation as expanding the vicinity distribution, which implies an optimum rather than a monotone. You sampled one point on that curve. You cannot say where the optimum is, or whether you were past it.

### 12.4 A vocabulary for hedging honestly

Hedging is not softening everything. It is matching each verb to the evidence behind it.

| Instead of | Write | Because |
|---|---|---|
| "Augmentation reduces CNN accuracy" | "Under a 3-epoch budget on a 10,000-image CIFAR-10 subset, augmentation significantly reduced test accuracy (−2.78 pp, p = 0.006)" | Name the conditions. The general claim needs experiments you did not run |
| "Augmentation improved generalization" | "The gap narrowed by 2.01 pp, but both terms fell and training fell nearly twice as far, so this is not evidence of improved generalization" | §R.2. Never report a difference without both terms |
| "Augmentation costs about 3 points of accuracy" | "The effect is statistically detectable across ten paired seeds and smaller than the 3.77 pp single-run noise floor" | §R.3. The effect size and the noise floor bound each other |
| "The model performs well on trucks" | "Truck recall is 0.64 with precision 0.51, so the model over-predicts the class rather than recognising it reliably" | §10.6. One ratio hides its terms |
| "Augmentation damages fine-grained classes" | "Bird recall fell by roughly a third of its baseline value, the largest per-class loss observed; this is consistent with the low-resolution interpolation mechanism but was not tested" | §9.6. Consistent with is not evidence for |
| "Our results are reproducible" | "We publish the seeds, the code and the distribution. The same seed re-executed in a separate process differs by a mean of 1.57 pp, so we claim reproducibility of the distribution, not of the digits" | §R.6 |
| "Augmentation does not help" | "We did not detect a benefit under this budget" | §1B.5. Absence of evidence is not evidence of absence |

The general test: **could a reader take your sentence out of context and be misled?** If yes, the conditions belong inside the sentence, not in a caveat three paragraphs later.

### 12.5 The benchmark's labels are not ground truth

Your accuracy figures measure **agreement with CIFAR-10's labels**, which are human-assigned and imperfect. Northcutt et al. (2021) show that widely used benchmark test sets contain recoverable label errors, which means a fraction of your "errors" are cases where the model was right and the label was wrong.

State it, and state it in the right register. At 51.35% accuracy, label noise is nowhere near your limiting factor: your model is wrong about half the time and the dataset is not. So this is a **boundary on what the metric means**, not an explanation for your numbers. Claiming label noise caused your results would be overclaiming in the opposite direction, and §2.1 already warned about it. The place it genuinely bites is small differences: a 1–2 pp gap between two models may sit inside the dataset's own annotation error, which is one more reason your 2.78 pp effect is reported alongside a 3.77 pp noise floor rather than on its own.

CIFAR-10 itself is a research dataset assembled for exactly this kind of small-scale experimentation (Krizhevsky, 2009), which is what makes it a defensible choice for a time-boxed project and simultaneously what bounds the claims you can make from it.

### 12.6 Accuracy establishes nothing about fairness or readiness

The assignment asks for this explicitly, and Roberts et al. (2021) is the right citation because it is the strongest available demonstration of the point.

They systematically reviewed machine learning models for COVID-19 diagnosis and prognosis from chest radiographs and CT: 2,212 studies screened, 415 retained, **62 quality-screened in detail**, and their conclusion was that **none had potential clinical utility**, because of methodological flaws and underlying biases. An entire literature of papers reporting high accuracy, filtered for quality, produced zero deployable models.

Read the implication carefully, because it is not "accuracy is useless." It is that **accuracy is a measurement of one thing, on one held-out split, and deployment readiness is a different property that has to be established separately.** A model can score well and still fail on a subgroup, fail under distribution shift, fail on data from a different source, or have learned a shortcut that happens to correlate with the label in this dataset.

Your model is far weaker evidence than any of those 62 studies. It is one architecture, trained for 8 seconds, on a benchmark subset, evaluated on a single held-out split of 3,000 images, with no fairness analysis, no subgroup breakdown and no external validation. The correct sentence is roughly:

> Test accuracy on a held-out CIFAR-10 split measures agreement with this dataset's labels under this training budget. It does not establish fairness across populations, reliability under distribution shift, or real-world readiness, and no claim of that kind is made here.

Use Roberts et al. where you say what your result does **not** license, not where you say what it does.

### 12.7 Seed variation as a claim boundary

Two citations do different jobs here.

Coakley & Gundersen (2026) ran 186 experiments at 100 seeds each and report CIFAR-10 seed ranges of **1.48–1.71 pp** on well-trained models, concluding that model performance is a distribution rather than a single value and that mean performance alone should not guide model selection. That makes your 3.77 pp spread, on a 10,000-image subset at ~51% accuracy after 3 epochs, entirely expected rather than a defect, and it is the published warrant for reporting distributions.

Åkesson et al. (2024) go further: across 50 seeds of a single algorithm, the best seed statistically significantly outperformed a substantial fraction of the others, from the identical algorithm. Their conclusion is that a statistically significant difference in performance is a weak and unreliable indicator of a true difference between two learning algorithms. Their domain is medical image segmentation, not CIFAR-10 accuracy, so argue the relevance on the **mechanism** (seed-only variation), not on the metric.

Together they support the boundary you actually need: a single run per condition could not have supported your finding, ten paired runs can support a claim about the mean difference, and even that claim is about **this configuration on this data**, not about augmentation.

### 12.8 Underclaiming is also a failure

Be careful not to over-correct. You rejected two pre-registered nulls, on ten paired seeds, with a replication that reproduced both findings more strongly (§R.5). Say that plainly, without apology. "Interpret without overclaiming" does not mean hedging every sentence into vapour; it means **claiming exactly what your evidence supports and disclosing everything that bounds it**. A limitations section that undermines a real finding is as inaccurate as one that inflates it.

The structure that gets this right is two clearly separated passages, which is also §1C.7's step 9: **"Hypothesis outcomes"** for what you committed to and what happened, then **"Additional observations"** for everything you noticed afterwards. A reader must be able to tell at a glance which claims were tested and which were generated.

### 12.9 Why reporting your own weakness strengthens the submission

The last idea in this document, and the one worth ending on.

There is an asymmetry in what a reader can see. A grader can read the limitations you listed. They can never see the limitations you did not check, the analyses you discarded, or the more flattering numbers you had available. **Disclosure is the only signal that exists** for the parts of your process that leave no trace.

So consider what your disclosures actually demonstrate:

- The r ≈ 0.19 pairing failure (§R.7) is only findable by someone who knows what pairing is supposed to buy, computes what it actually bought, and is willing to look. Nobody stumbles into that.
- The seed non-determinism (§R.6) required deliberately re-running an identical configuration in a separate process for the purpose of measuring your own irreproducibility.
- The noise-floor decomposition corrected an earlier claim in your own notes, in a direction that made your effect look less impressive.
- The replication (§R.5) produced better statistics on both metrics and you kept the pre-registered run as the headline anyway.

None of those make the work weaker. They are all evidence that the work was interrogated rather than assembled. A generic limitations section (small dataset, few epochs, future work should train longer) is indistinguishable from one written by somebody who never checked. Specific ones are the only ones that carry information about the person who wrote them.

And the thesis the whole document has been building toward: the most valuable sentence in your writeup will not be the one with the p-value in it. It will be the one where you explain why your significant gap reduction is not the good news it appears to be. Running a t-test is one line. Working out what your own significant result does not mean is the part that is genuinely hard, and it is what §1.8, §1B.5, §1C.2 and §R.8 have all been pointing at since the first page.

---

## Glossary

| Term | Plain meaning |
|---|---|
| **Epoch** | One full pass over the training data |
| **Batch** | The group of images processed before one weight update (64 here) |
| **Seed** | A number fixing the random generator so runs are repeatable |
| **Feature map** | The output image produced by one filter |
| **Overfitting** | Learning the training set's quirks instead of general patterns |
| **Generalization gap** | Training accuracy minus validation accuracy |
| **Regularization** | Any technique that trades training fit for better generalization |
| **Invariance** | A change to the input that should *not* change the prediction |
| **Macro average** | Compute a metric per class, then average with equal class weight |
| **Logit** | A raw pre-softmax score |

---

## Running question log

Things to raise with your instructor or resolve as a group:

1. Is a null result (augmentation ≈ baseline) acceptable, given 3 epochs? *(The brief says yes, but worth confirming.)*
2. May we increase epochs, or must training settings stay exactly as shipped? *(This materially affects whether H₁a can hold.)*
3. May we re-run across multiple seeds, or is one run per condition the expectation?

---

## Sources still needed (APA 7, three required)

- [ ] **Method source**, scholarly, on CNNs or data augmentation
- [ ] **Dataset/application source**, on CIFAR-10 or image classification
- [ ] **Limitations/fairness/validity source**

---

*Last updated: 2026-08-02. All twelve modules complete. Module R is written from our own finished experiments; read it before drafting the interpretation and limitations sections. Modules 3 to 6 cover the architecture and mathematical-foundation rubric lines, Module 7 covers reproducibility, and Modules 10 to 12 cover reporting, the confusion matrix and claim boundaries.*
