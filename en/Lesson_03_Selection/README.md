# Lesson 03 — Selection

Five selection methods, built one on top of the other on the same ten
individuals, and measured with the same instrument — so that the lesson ends not
with a list of five techniques but with one scale on which all five sit, and one
trade-off nobody escapes.

- **Source in the book:** Gridin, *Learning Genetic Algorithms with Python*,
  Chapter 3 (*Selection*) — 5 sections, ~2,600 words, 6 figures, 225 lines across
  15 files.
- **Budget:** 48 min (Chapter 3 is 7.4% of the book).
- **Prerequisites:** Lesson 01. This lesson uses the *same* population — same
  objective, same `SEED = 52` — so the numbers continue rather than restart, and
  tournament selection in step 6 reproduces Lesson 01 step 3 exactly.

## Mathematical model

For non-negative weights \(w_i\), proportional selection uses
\(p_i=w_i/\sum_jw_j\) and individual \(i\) receives \(Np_i\) copies in
expectation. A free shift floor changes those probabilities. Rank selection
uses order-based weights; SUS keeps the same expectation while reducing
sampling spread. With replacement and rank (r=1) best, a size-(k) tournament
selects rank (r) with probability
( ((N-r+1)/N)^k-((N-r)/N)^k ); these probabilities sum to one. Elitism preserves the
current best only with deterministic, stationary fitness and unchanged copying.

## What the student leaves with

1. Selection invents nothing. It takes N individuals and returns N, all copies of
   what was already there — so the only thing to measure is the bookkeeping: who
   got copied, how many times, who disappeared.
2. **Proportional selection hides a parameter.** The population's fitness is
   negative almost everywhere, a roulette sector cannot be negative, and the
   usual repair — shift everything up — silently introduces a floor that controls
   the pressure more strongly than the fitness values do.
3. Rank selection removes that scale dependence; elitism removes the risk of a
   generation being worse than its parent; SUS removes the sampling noise;
   tournament puts the pressure in one integer you set on purpose.
4. **Pressure and diversity are one trade, not two knobs.** Across 13
   configurations: pressure vs mean fitness correlates **+0.95**, pressure vs gene
   spread **−0.92**. No method here buys fitness without paying in diversity.

## The running example

The ten individuals of Lesson 01 on `f(x) = sin(x) - 0.2·|x|`, `x ∈ [-10, 10]`,
and one measuring function (`pressure_report`) applied to every method over
**2,000 draws** from a fixed seed — so the rows of the comparison tables are
comparable by construction rather than by hope.

The prefix is `selection_pressure_`, because pressure is what the lesson is
actually about.

## The six steps

| # | Script | What it adds | What its output proves |
|---|---|---|---|
| 1 | `selection_pressure_01_the_population.py` | the measuring apparatus, and the do-nothing baseline | Copying everyone once: 0 extinct, diversity 6.665, 1 copy of the best — the zero of every later measurement. Fitness is negative almost everywhere, so no wheel can be built yet |
| 2 | `selection_pressure_02_proportional.py` | `shift_to_positive()`, `select_proportional()`, `pressure_report()` | The floor *is* the pressure: the best expects **2.35** copies at floor 0.001 but **1.30** at floor 3.0 — same population, same wheel |
| 3 | `selection_pressure_03_rank.py` | `select_rank()` | Order alone fixes the pressure at one definite value, immune to rescaling — but the best individual is still lost in **13.6%** of draws |
| 4 | `selection_pressure_04_elitism.py` | `select_rank_with_elite()` | One line takes the loss of the best from 13.6% to **0.0%**, raises pressure (1.80 → 2.64 copies) and costs diversity (5.448 → 5.210) |
| 5 | `selection_pressure_05_sus.py` | `select_stochastic_universal_sampling()` | Same wheel, same expectation, spread collapses. And it does **not** fix the floor: 2.35 vs 1.31 copies, the same split step 2 found |
| 6 | `selection_pressure_06_tournament.py` | `select_tournament()`, and the whole comparison | One integer sets the pressure: k=3 → 2.74 copies, k=5 → 4.13, k=10 → 6.52. Across all 13 configurations, pressure vs quality **+0.95**, pressure vs diversity **−0.92** |

**These numbers are verified against actual output.** Any slide, handout or
translation that states a figure must state one of these. Re-run the script
rather than trusting this table if the code has changed.

## How the code is marked up

Two devices, mandatory course-wide:

- **The recipe** — each script after the first opens with a `CHANGES FROM …`
  block listing, in order, the changes that turn the previous script into this
  one. That order is the order to type in class and the order the slides follow.
- **The bands** — each recipe item has its own `# --- NEW (n) name ---` band at
  the site of the change, numbered and named to match. Everything outside a band
  is code the students already have.

## Running it

```bash
uv run en/Lesson_03_Selection/src/selection_pressure_01_the_population.py
```

Each script is self-contained, takes no arguments, and saves its figure to
`../figures/` under its own name. The whole sequence runs in about **16 s**
(2,000 draws per measurement is what buys the stable averages).

## Folder contents

```
Lesson_03_Selection/
├── README.md      this file — the lesson's contract
├── src/           the six numbered scripts
├── figures/       one figure per script
```

## Paths and status, September 25, 2026

- Shared scripts: [`src/`](src/), 6 steps; reproducible figures in [`figures/`](figures/).
- Independent study: [`student/lesson_03.pdf`](student/lesson_03.pdf) and [`student/lesson_03_selection_pressure.ipynb`](student/lesson_03_selection_pressure.ipynb), with TeX source beside the PDF.
- Instructor use: [`instructor/presentation/lesson_03_compact.pdf`](instructor/presentation/lesson_03_compact.pdf) and [`instructor/notebooks/lesson_03_compact.ipynb`](instructor/notebooks/lesson_03_compact.ipynb). The compact deck has 9 slides; the notebook has explanatory comments next to its code.

The extensive route retains the complete progression; the compact pair is self-contained for roughly 40 minutes. Examples are synthetic. Transfer to the Planta Física challenge is a modeling choice, not a partner-validated outcome. Student distribution remains blocked until rights for Gridin-derived material are documented. Canvas and hosted Colab were not changed or tested.
