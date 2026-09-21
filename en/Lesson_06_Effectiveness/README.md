# Lesson 06 — Measuring Effectiveness

Every lesson so far ended by reading a single run; this one builds the
machinery for reading a thousand. A genetic algorithm is a random variable,
so a success rate has to be defined (what counts as success?), measured (how
many runs?), and priced (what did the evaluations cost?) — and each of the
three questions gets its own instrument and its own surprise.

- **Source in the book:** Gridin, *Learning Genetic Algorithms with Python*,
  Chapter 6 (*Effectiveness of Genetic Algorithms*) — 4 sections, ~2,100
  words, 8 figures, 301 lines of code in 5 files
  (`03-referencias/…/Chapter06/`).
- **What the refactor does to it:** the chapter's three experiments (single-run
  statistics, the 1000-run histogram, the population-vs-evaluations scatter)
  become the spine of **four numbered steps**. Step 2 inserts a step the book
  does not have — defining "success" before counting it — because the book's
  own 2-D landscape makes the naive definition measurably meaningless.
- **Target duration:** 39 min by book weight (Chapter 6 is 6.0% of the text).
  Four steps rather than the brief's usual five to eight: each diff is one
  idea sized like Lesson 01's, and the budget cannot hold more. Flagged as a
  deliberate choice, not an oversight.
- **Prerequisites:** Lessons 01–05 (the algorithm is reused unchanged);
  Lesson 02 step 5 (the brute-force grid, reused as the verdict instrument);
  Lesson 05 (the "optimum with no volume" finding, which returns in two
  dimensions).

## Mathematical model

For \(n\) independent seeded runs with \(s\) successes, the observed rate is
\(\hat p=s/n\) and its plug-in standard error is
\(\sqrt{\hat p(1-\hat p)/n}\). This is one estimated standard deviation, not a
confidence interval; the scripts also report a 95% Wilson interval, which stays
non-degenerate at 0 or \(n\) successes. The optimum and target share are
dense-grid approximations. If a blind draw hits with probability \(q\), a run
spending \(E\) evaluations succeeds with probability \(1-(1-q)^E\); variable
budgets are averaged run by run.

## What the student leaves with

1. A GA is a **random variable**: the same configuration on 8 seeds spreads
   its answers wider (0.8946) than one run's whole improvement (0.5822).
2. A success rate needs a **verdict from outside the run** — a brute-forced
   optimum and a stated tolerance — and the verdict is only meaningful if the
   target has **volume**: a measurable share of the search space.
3. A rate measured on 8 runs has a plug-in standard error of ±17 points; on
   1000 runs it is ±1.3. This is one estimated standard deviation, not a
   confidence interval; precision improves only with the square root of runs.
4. Effectiveness and **efficiency** pull in opposite directions: the most
   reliable population is the least profitable per evaluation.
5. On a wide target the GA barely beats blind search at equal budget — a
   property of the problem, not a defect of the algorithm.

## The running examples

Two landscapes, on purpose:

- **The book's 2-D function** `f(x, y) = sin(x)cos(x) - (|(x+50)(y-10)|/10)^0.1`
  on `[-100, 100]²` — steps 1 and 2. Kept as the *specimen*: its optimum is
  real (+0.500000) but has no volume, which is exactly what makes it the
  perfect counterexample for defining success.
- **Lesson 01's sine landscape** `f(x) = sin(x) - 0.2|x|` on `[-10, 10]` —
  steps 2, 3 and 4. Chosen because its optimum is a smooth peak with a
  measurable width (1.43% of the box), so a success rate there means
  something — and because it connects the lesson back to Lesson 01's
  local-optimum failure, which step 3 measures at scale.

The operators are the course's own, unchanged: tournament of 3, blend
crossover with alpha = 1.0, gaussian mutation behind a per-individual coin.
Every number in this lesson continues the earlier lessons instead of
restarting them.

## The four steps

| # | Script | What it adds | What its output proves |
|---|---|---|---|
| 1 | `success_rate_01_one_run.py` | the course's GA on the book's 2-D landscape, run 8 times | Spread across seeds is 0.8946 — 1.5× the tracked run's whole gain (+0.5822); seed 3 reports -0.0129, seed 4 reports -0.9076, and both reports are true |
| 2 | `success_rate_02_the_verdict.py` | `brute_force_optimum()`, `TOLERANCE`, `verdict()`, `within_tolerance_share()`, the move to sine 1-D | The 2-D optimum is real (+0.500000) but occupies 0.003197% of the grid — all 128 winning points sit exactly on y = 10 — so 0 of 8 runs "succeed"; on sine 1-D the target is 1.4299% of the box (0.286 wide) and 5 of 8 runs find it: 62.5% ± 17.1% |
| 3 | `success_rate_03_a_thousand_runs.py` | `success_rate()`, `standard_error()`, `wilson_interval()`, `RUNS = 1000` | The observed rate is 79.4%, one plug-in SE is 1.3%, and the 95% Wilson interval is reported separately; the 8-run estimate differs by 16.9 points, while 71/1000 runs end on Lesson 01's local peak at x = -4.51 |
| 4 | `success_rate_04_the_budget.py` | the evaluation counter, the population sweep, `blind_success()` | Population 10→30: success 80.0%→99.4%, cost 100→300 evaluations, yield falls 8.00→3.31 successes per 1000 evaluations; blind draws at equal budget reach 76.2% and 98.7% — the GA's margin is +3.8 and +0.7 points |

**These numbers are verified against actual output.** Any slide, handout or
translation that states a figure must state one of these, not a plausible
substitute. Re-run the script rather than trusting this table if the code has
changed since.

## How the code is marked up

Two devices, both mandatory course-wide, both applied
here:

- **The recipe** — every script after the first opens with a `CHANGES FROM …`
  block: the ordered changes that turn the previous script into this one.
  Steps 3 and 4 also carry a `REMOVED FROM …` note, because each drops the
  previous step's one-off specimen code; the note is unnumbered on purpose,
  so the recipe keeps its one-number-per-band mechanics.
- **The bands** — each recipe item has its own `# --- NEW (n) name ---` band
  at the site of the change. A changed value is marked on the line instead:
  `RUNS = 1000   # --- CHANGED ---` in step 3, `RUNS = 500` in step 4.

Run i uses seed i throughout the lesson, so any subset of runs is
reproducible on its own — and the first 8 of step 3's thousand runs *are*
step 2's experiment, seed for seed, which is what makes the comparison
honest.

## Running it

From the repository root, with `uv` handling the environment:

```bash
uv run en/Lesson_06_Effectiveness/src/success_rate_01_one_run.py
```

Every script is self-contained: no imports across steps, no imports across
lessons, no command-line arguments. Figures are **saved** to `../figures/`,
never shown, so the whole sequence runs unattended — about 8 seconds for all
four steps.

## Folder contents

```
Lesson_06_Effectiveness/
├── README.md      this file — the lesson's contract
├── src/           the four numbered scripts
├── figures/       generated plots (never edited by hand, never committed by hand)
```

## Status

| Piece | State |
|---|---|
| Code | Done. Four scripts, all run clean, all claims verified against output. |
| Figures | Done. Each step saves one figure, named after its own script. |
| Slides | Not started, and blocked on the course Beamer template. |
| Spanish mirror | Not started. The Spanish folder still holds the pre-refactor monolith — see TODO T2. |

## Guided notebooks

Student entry point: [`notebooks/README.md`](notebooks/README.md). This lesson has 1 independently runnable sequence: [`lesson_06_success_rate.ipynb`](notebooks/lesson_06_success_rate.ipynb). The retained outputs were validated on date omitted against the frozen scripts in three fresh-kernel path modes; see the notebook README for environment details and the explicit hosted-Colab gap.

