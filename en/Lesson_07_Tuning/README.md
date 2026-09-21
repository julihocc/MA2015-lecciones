# Lesson 07 — Parameter Tuning

The book's laboratory chapter, rebuilt on Lesson 06's instruments. The three
knobs — crossover probability, mutation probability, population size — are
turned one at a time and then together, every setting measured over hundreds
of runs and priced against blind search at equal budget. The finding the
lesson is built around: on this landscape every knob is secretly a budget
knob, and the bell curve the textbooks promise exists only in the margin
over blind search, not in the raw success rate.

- **Source in the book:** Gridin, *Learning Genetic Algorithms with Python*,
  Chapter 7 (*Parameter Tuning*) — 3 sections, ~4,200 words, **42 figures**,
  248 lines of code in 4 files (`03-referencias/…/Chapter07/`).
- **What the refactor does to it:** the book tunes by watching one run per
  setting, each knob on a different landscape, across 30 blocking
  `plt.show()` windows. The refactor keeps the book's protocol as the
  *specimen* (step 1, and it fails on camera), then re-asks the same
  questions with Lesson 06's machinery: success rates over 500 runs,
  standard errors, evaluation counting, and a blind-search benchmark. One
  landscape for the whole lesson, so the knobs can actually be compared —
  the 42 figures become six.
- **Target duration:** 79 min by book weight (Chapter 7 is 12.1% of the
  text, the second-heaviest chapter). Six steps: the two sweep steps and the
  grid carry the time; steps 1 and 5 are fast.
- **Prerequisites:** Lesson 06 (the success-rate instrument, the evaluation
  counter, the blind benchmark — all reused, none re-derived); Lessons
  01–05 (the algorithm under study).

## Mathematical model

Settings share seed indices, so comparisons are paired. Define
\(d_i=I_{i,B}-I_{i,A}\in\{-1,0,1\}\); the estimated rate gap is \(\bar d\)
with standard error \(s_d/\sqrt n\). This replaces an independent-samples
calculation that ignores pairing. Individual rates retain Wilson intervals,
including the 500/500 observation. The 36-cell map is exploratory for this
problem and these seeds, not a universal parameter optimum or a confirmatory
multiple-comparison study.

## What the student leaves with

1. Tuning from single runs is **reading noise**: the same protocol on 8
   seeds splits two settings by 0.000004 on one seed and by 0.6284 on
   another.
2. A knob turned up raises the success rate **and the budget** — the two
   effects are confounded until evaluations are counted.
3. The honest currency is the **margin over blind search at equal budget**;
   in that currency the crossover margin shrinks as the knob opens, and the
   mutation curve bends into the textbook bell (peak near 0.2, negative at
   both extremes).
4. Population is the budget knob with no disguise: a perfect 100% success
   score can simply be bought, at a five-fold collapse in yield.
5. Every tuning verdict is a verdict **about one problem**. A fixed setting
   is a compromise — which is why the adaptive GA (Lesson 12) exists.

## The running example

Lesson 01's sine landscape, `f(x) = sin(x) - 0.2|x|` on `[-10, 10]`, optimum
+0.705908 known by brute force, success meaning within 0.01 of it. One
landscape for the whole lesson, on purpose: the book turns each knob on a
different function, so its chapter never compares anything to anything. The
operators are the course's own (tournament 3, blend alpha 1.0, gaussian
mutation), so every number continues Lessons 01–06. The price of this
choice is stated out loud in step 6: the target here is *wide* (1.43% of
the box), blind money goes far, and the verdicts are verdicts about this
problem.

## The six steps

| # | Script | What it adds | What its output proves |
|---|---|---|---|
| 1 | `tuning_01_single_runs.py` | the book's protocol, faithfully (pop 16, mutation 0.2, seed 63) | One run per setting crowns pc = 0.7 by 0.0021; over 8 seeds the win counts go 0/3/5, the tightest split is 0.000004 (seed 3) and the widest 0.6284 (seed 1) — the ranking is unstable |
| 2 | `tuning_02_the_instrument.py` | `standard_error()`, paired outcomes in `measure()`, knobs as arguments, RUNS = 500 | The observed crossover rate rises 55.6% → 83.0%; the 27.4-point endpoint gap is reported with the SE of paired 0/1 differences, while the table still lacks cost |
| 3 | `tuning_03_the_budget_knob.py` | the evaluation counter, `blind_success()`, the margin column | Crossover 0.0 spends 20 evals/run, 1.0 spends 120; the margin over blind falls +30.6 → +0.8 points; the 100 extra evaluations buy the GA 27.4 points where 100 blind draws buy 57.2 — the knob is a budget knob |
| 4 | `tuning_04_mutation.py` | the second knob (7 settings, crossover fixed 0.8) | The raw rate is monotone to mutation 1.0 (61.8% → 89.8%) — no bell curve; the margin bends: −10.9% at 0.0 (worse than dice), peak +5.9% near 0.2, −3.7% at 1.0 — the bell curve lives in the margin |
| 5 | `tuning_05_population.py` | the third knob (the book's sizes 6, 10, 20, 50) | Population 50 records 500/500 successes (95% Wilson interval 99.2%–100%) at 550 evals/run; yield collapses 9.25 → 1.82 successes per 1000 evals; the margin peaks at population 10 (+5.9%) and vanishes at 50 — "which is best?" has no answer without a currency |
| 6 | `tuning_06_the_grid.py` | the 6×6 grid, both knobs together, RUNS = 100 | The (0, 0) corner recovers blind search (9.0% ± 2.9% vs 13.4% for 10 blind draws — the instrument is calibrated); the best cell (1.0, 0.5) scores 90.0% at 160 evals with a margin of −0.01%; only 20 of 36 cells beat their own budget — the operators earn their keep in the cheap cells |

**These numbers are verified against actual output.** Any slide, handout or
translation that states a figure must state one of these, not a plausible
substitute. Re-run the script rather than trusting this table if the code has
changed since.

## How the code is marked up

Two devices, both mandatory course-wide, both applied
here:

- **The recipe** — every script after the first opens with a `CHANGES FROM …`
  block. Steps 4 and 5 are one-item recipes on purpose: the instrument is
  built once (steps 2–3) and each later step only re-aims it, marked with a
  single `# --- CHANGED ---` block at the study-design constants.
- **The bands** — each recipe item has its own `# --- NEW (n) name ---` band
  at the site of the change. Step 3's counter has two sites (the `Individual`
  class and the collection lines inside `measure()`); the band sits on the
  class and the collection lines carry a pointer comment.

Run i uses seed i throughout, so every setting and every grid cell faces the
same dice and all comparisons are paired.

## Running it

From the repository root, with `uv` handling the environment:

```bash
uv run en/Lesson_07_Tuning/src/tuning_01_single_runs.py
```

Every script is self-contained: no imports across steps, no imports across
lessons, no command-line arguments. Figures are **saved** to `../figures/`,
never shown, so the whole sequence runs unattended — about 15 seconds for
all six steps.

## Folder contents

```
Lesson_07_Tuning/
├── README.md      this file — the lesson's contract
├── src/           the six numbered scripts
├── figures/       generated plots (never edited by hand, never committed by hand)
```

## Status

| Piece | State |
|---|---|
| Code | Done. Six scripts, all run clean, all claims verified against output. |
| Figures | Done. Each step saves one figure, named after its own script. |
| Slides | Not started, and blocked on the course Beamer template. |
| Spanish mirror | Not started. The Spanish folder still holds the pre-refactor monolith — see TODO T2. |

## Guided notebooks

Student entry point: [`notebooks/README.md`](notebooks/README.md). This lesson has 1 independently runnable sequence: [`lesson_07_tuning.ipynb`](notebooks/lesson_07_tuning.ipynb). The retained outputs were validated on date omitted against the frozen scripts in three fresh-kernel path modes; see the notebook README for environment details and the explicit hosted-Colab gap.

