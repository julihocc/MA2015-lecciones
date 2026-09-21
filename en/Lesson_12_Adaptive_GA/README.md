# Lesson 12 — The Adaptive Genetic Algorithm

Every lesson so far has written the parameters at the top of the file as
constants, and Lesson 07 tuned those constants by grid search. This lesson asks
whether they should be constants at all: it builds a sensor that tells a run
whether it is still improving, wires that sensor to the crossover and mutation
probabilities and to the population size, and then measures — twelve paired
runs at an equal evaluation budget — whether the machinery is worth having.
Against the untuned setting it inherited it wins every run; against four *tuned*
fixed settings it is **behind on the mean in three of the four and cannot be
told apart from the best**. That negative result is the lesson.

- **Source in the book:** Gridin, *Learning Genetic Algorithms with Python*,
  Chapter 12 (*Adaptive Genetic Algorithm*) — 6 sections, ~2,299 words, 8
  figures, 424 lines in 6 files (`03-referencias/…/Chapter12/`).
- **What the refactor does to it:** the book demonstrates the mechanism on one
  seed and then compares adaptive against "classical" over 100 runs in which the
  two algorithms do **not** spend the same number of evaluations. The refactor
  keeps the mechanism exactly (the same sensor, the same 1.1 / 0.99 multipliers,
  the same immigrants-and-culls population rule) and replaces the comparison
  with a paired, equal-budget one in the currency Lesson 07 established. It then
  adds the step the book does not have: the adaptive algorithm against a *tuned*
  fixed algorithm.
- **Target duration:** 43 min by book weight (Chapter 12 is 6.6% of the text).
  Six steps: 1–4 are one run each and go quickly; steps 5 and 6 are the lesson.
- **Prerequisites:** Lesson 10 (the travelling-salesperson problem, ordered
  crossover, inversion mutation — restated here, not imported), Lesson 07
  (tuning, and evaluations as the currency of comparison), Lesson 06 (why one
  run proves nothing).

## Mathematical model

Let \(m_t\) be the population-mean route length and \(\bar m\) the preceding
ten-generation mean. The sensor calls the run improving when
\(m_t<(1-0.001)\bar m\). Probabilities are multiplied by 0.99 while improving
and 1.1 while stalled, then clipped; resizing removes one individual for free
or adds evaluated immigrants. Comparisons use paired differences. The displayed
two-standard-error interval is approximate, not a significance test, and the
best fixed setting means best of four on these same twelve seeds.

## What the student leaves with

1. A fixed parameter is a compromise between two phases of a run, and the
   compromise can be measured: half the budget of step 1's run buys 85% of its
   total improvement, and the setting that bought that half is still in force
   for the second.
2. A run can sense its own progress without knowing the optimum — compare the
   population mean against its own trailing average — and that crude signal is
   the only sensor the adaptive algorithm has.
3. An adaptive rule must be *audited*, not assumed: in step 3 the "raise the
   probabilities when stalled" branch fires **0 times in 99 generations**. What
   is advertised as adaptation runs as a one-way decay schedule.
4. Adaptation beats an untuned setting decisively (−8,050 route length over 12
   paired runs, 12 wins out of 12) and does **not** beat a tuned one
   (+1,274 ± 727 against the best of four fixed settings, 4 wins out of 12).
5. Therefore an adaptive scheme is a cheap insurance against a badly chosen
   constant, not a replacement for Lesson 07.

## The running example

Lesson 10's travelling-salesperson instance: the 48 US state capitals of
`att48_xy.txt`, the same ordered crossover, the same inversion mutation, the
same tournament selection and elitism. This is the book's own choice — chapter
12 is the one place where Gridin imports another chapter's problem — and it is
kept because the comparison is then against a problem the students have already
seen searched, with a nearest-neighbour baseline (**40,526**) they already know.
The no-imports rule still holds: everything is restated inside each script and
the data file sits in this lesson's own `src/`.

Two things change from Lesson 10. The crossover probability becomes an explicit
knob (Lesson 10 always crossed), and **the stopping rule is an evaluation
budget of 12,000, not a generation count** — the adaptive algorithm resizes its
own population, so generations stop being comparable units, while evaluations
stay comparable. Every table prints the evaluations actually spent so the reader
can check the budgets really are equal.

## The six steps

| # | Script | What it adds | What its output proves |
|---|---|---|---|
| 1 | `tsp_01_fixed_parameters.py` | the whole program: Lesson 10's search under a 12,000-evaluation budget, crossover 0.90 / mutation 0.25, population 120 | Seed 1 spends 11,901 evaluations in 99 generations and returns a legal route of **57,076**, 40.8% longer than the nearest-neighbour baseline of 40,526 |
| 2 | `tsp_02_the_stall.py` | `average()`, `is_improving()`, the stall trend, `shade_stalls()` | The signal calls **14 of 99** generations stalled, the first at generation 74, and 14% of the budget is spent inside them; the first half of the budget buys **85%** of the run's total improvement, the second half 15% |
| 3 | `tsp_03_adaptive_probabilities.py` | `adapt_probabilities()`, and a `run()` that can use it | Same seed, same budget: adaptive **48,830** against fixed **57,076** (−14.4%). But the stalled branch fires **0 of 99** times — crossover walks 0.90 → 0.33 and mutation 0.25 → 0.09 and never comes back up. The rule is behaving as a decay schedule, not as adaptation |
| 4 | `tsp_04_adaptive_population.py` | `resize_population()`, and a second switch in `run()` | Resizing alone: 35 stalls, 70 immigrants, 103 culls, population 68–120, **138** generations for the same budget and **49,374**. With the probabilities too: 41 stalls, population down to 40, 155 generations, crossover ranging 0.39–1.00 — resizing brings back the stalls step 3 had made disappear — and **52,295**, worse on this seed than resizing alone |
| 5 | `tsp_05_twelve_runs.py` | `RUNS`, `measure()`, `paired_summary()`, the table and box plot | Over 12 paired runs (run *i* on seed *i*): fixed **58,195**, probabilities **50,145** (−8,050 ± 1,022, wins 12/12), resize **53,555** (−4,640 ± 926, 10/12), both **52,622** (−5,573 ± 1,127, 11/12). Every regime spends 11,901–11,946 evaluations, so the budgets are equal. The approximate mean-difference ± two-SE interval excludes zero for all three comparisons with *this* fixed setting; this is descriptive, not a significance test |
| 6 | `tsp_06_a_tuned_rival.py` | `run()` takes its starting rates as arguments, `FIXED_SETTINGS`, the verdict | Against four fixed settings at the same budget: 0.90/0.25 → 58,195 (adaptive wins 12/12), **0.60/0.05 → 48,870** (adaptive wins 4/12, **+1,274 ± 727**), 0.40/0.15 → 49,675 (6/12, +469 ± 480), 0.20/0.30 → 49,626 (5/12, +519 ± 835). The adaptive regime is behind 3 of the 4 on the mean; its approximate two-SE interval against the best includes zero. The best fixed setting means best among these four on these same 12 seeds, not independently validated tuning |

**These numbers are verified against actual output.** Any slide, handout or
translation that states a figure must state one of these, not a plausible
substitute. Re-run the script rather than trusting this table if the code has
changed since.

### The result the lesson was not planned around

The sequence was designed expecting step 3 to show the sensor opening the
operators up on a stall. It never does: with the probability rule on, keeping
the operators calm keeps the population mean improving, which keeps the sensor
satisfied, which keeps the operators closing. The step was rewritten around
that measurement, and step 4 then shows the stalls coming back as soon as the
population is allowed to shrink. Step 6 was planned as "the adaptive scheme
loses to a tuned rival" and the honest reading of its output is weaker than
that: it loses on the mean to three settings out of four, but the gap against
the best one is 1.75 standard errors, which at twelve runs is *not* a
distinguishable difference. The script says so in those words. Settling it

## How the code is marked up

Two devices, both mandatory course-wide:

- **The recipe** — every script after the first opens with a `CHANGES FROM …`
  block: the ordered list of changes that turn the previous script into this
  one. That order is the order to type them in class.
- **The bands** — each recipe item has its own `# --- NEW (n) name ---` band at
  the site of the change, numbered and named to match the recipe. Bands do not
  carry forward: step 4 shows step 4's three changes only. Where one change has
  a second site (the two probability variables read inside the generation loop
  in step 3, the rates travelling through `measure()` in step 6), the second
  site carries a plain `# (n)` pointer comment rather than a second band.

Run *i* uses seed *i* throughout steps 5 and 6, so every regime and every fixed
setting faces the same twelve starting populations and all comparisons are
paired. Steps 1–4 use seed 1, which is also run 1 of the tables.

## Running it

From the repository root, with `uv` handling the environment:

```bash
uv run en/Lesson_12_Adaptive_GA/src/tsp_01_fixed_parameters.py
```

Every script is self-contained: no imports across steps, no imports across
lessons, no command-line arguments. `att48_xy.txt` lives beside the scripts and
is read by all six. Figures are **saved** to `../figures/`, never shown.

**Runtime: about 42 s for the whole sequence** — steps 1–4 take under 2.5 s
each, steps 5 and 6 take about 16 s each because they are 48 and 60 runs of the
GA. What was cut to get there: the budget is 12,000 evaluations rather than the
book's run-until-it-stops rule, the study is 12 runs
rather than the book's 100, and the sweep in step 6 is four fixed settings
rather than a full grid. If the sequence still does not fit a slot, cut `RUNS`
in steps 5 and 6 from 12 to 8 — but say out loud that the standard errors grow
with it, because at 8 runs step 6's comparison is worth nothing.

Two more things the book has and this lesson does not: its fitness-driven
operators (a crossover that keeps the best two of four candidates, a mutation
that retries up to three times) and its stopping rule (run until the best
fitness stops improving). Both were dropped on purpose. The operators hide
extra evaluations inside themselves, which makes an equal-budget comparison
impossible to state honestly, and the stopping rule gives the two algorithms
different budgets — which is exactly the flaw Lesson 07 spent its time on.

## Folder contents

```
Lesson_12_Adaptive_GA/
├── README.md      this file — the lesson's contract
├── EXERCISE.md    the 4x4 crossover x mutation grid (not a seventh script)
├── src/           the six numbered scripts and att48_xy.txt
├── figures/       generated plots (never edited by hand, never committed by hand)
├── slides/        the Beamer deck
```

## Status

| Piece | State |
|---|---|
| Code | Done. Six scripts, all run clean, all claims computed from their own output. |
| Figures | Done. One figure per script, named after its producing script. |
| Slides | Done. |
| Spanish mirror | Scripts exist under `es/Leccion_12_AG_Adaptativo/`. Remaining work is polish (the lesson workflow), not a rewrite. |
| The 12-run verdict | Kept. T3 decided against the 100-run study. The 4x4 grid is the exercise in `EXERCISE.md`. |

## Guided notebooks

Student entry point: [`notebooks/README.md`](notebooks/README.md). This lesson has 1 independently runnable sequence: [`lesson_12_adaptive_ga.ipynb`](notebooks/lesson_12_adaptive_ga.ipynb). The retained outputs were validated on date omitted against the frozen scripts in three fresh-kernel path modes; see the notebook README for environment details and the explicit hosted-Colab gap.

