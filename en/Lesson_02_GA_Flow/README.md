# Lesson 02 — The genetic algorithm flow

Lesson 01 ended with a genetic algorithm that worked but whose loop body was a
run of unnamed comment blocks. This lesson turns that loop into a flow you can
talk about: named phases, a problem that arrives as an argument instead of being
welded in, four numbers per generation instead of one, and a rule for when to
stop — which the last step then catches telling a comfortable lie.

- **Source in the book:** Gridin, *Learning Genetic Algorithms with Python*,
  Chapter 2 (*Genetic Algorithm Flow*) — 7 sections, ~2,350 words, 8 figures, 188
  lines across 8 files (`individual.py`, `fitness.py`, `selection.py`,
  `crossover.py`, `mutate.py`, `population.py`, `settings.py`,
  `genetic_algorithm_flow.py`). The chapter's file split is what the steps follow.
- **Budget:** 44 min (Chapter 2 is 6.7% of the book).
- **Prerequisites:** Lesson 01. Step 1 reproduces Lesson 01 step 6 to the digit
  with the same seed, and says so in its own output — that identity is the proof
  that naming the flow changed nothing.

## Mathematical model

Write the population at generation \(t\) as \(P_t\). One generation is the
random transition
\[
P_{t+1}=R\!\left(M\!\left(C\!\left(S(P_t)\right)\right)\right).
\]
The reported maximum, mean and spread describe the finite population; the
best-so-far is separate state. A patience rule detects
\(b_t-b_{t-k}<\varepsilon\); it does not establish convergence or optimality.
The staircase is (r\lfloor(T-s|x-x_0|)/h\rfloor). Each ordinary band has
one-sided width (h/s=0.5); with (T=10.5,h=1,s=2), its top is
(|x-4|\le0.25), hence 0.5 wide. Its nonzero width matters as much as height.

## What the student leaves with

1. A genetic algorithm has five named phases, and one generation of evolution is
   one function. You cannot discuss a flow that has no boundary.
2. **The problem is an argument, not part of the algorithm.** Once the fitness
   function is injected, the same driver solves a different problem with nothing
   else changed — and a minimisation is just a maximisation with its sign flipped.
3. The best individual is the number a GA is least honest about. Average fitness
   and **gene spread** are what say whether a population is still searching.
4. A run can end three ways; "the improvements dried up" is the only one that
   needs measuring, and the measurement is cheap.
5. **A cheap stopping rule is not good news on its own.** It reports that a run
   stopped improving — not whether the run was finished or merely stuck.

## The running examples

The prefix is `ga_flow_`. Three fitness functions are handed to one unchanged
driver:

- `sine_landscape` — Lesson 01's `sin(x) − 0.2·|x|`, now just one problem among others.
- `closeness_to_target` — `−(x − 4.2)²`, a minimisation written as a maximisation.
- `staircase` — flat steps rising towards x = 4, added in step 5. Two details are
  deliberate. Its top step is **interior**: at a border, `clamp()` would hand it to
  any mutation that overshoots. And the tent it is cut from is 10.5 rather than
  10.0, so the top step is **0.5 wide rather than a single point** — see the
  correction below.

## The five steps

| # | Script | What it adds | What its output proves |
|---|---|---|---|
| 1 | `ga_flow_01_the_five_phases.py` | `evolve_one_generation()`, `run()`, a per-phase census | Same seed, same answer as Lesson 01: `x=+1.372 f=+0.706`. The run cost **107** fitness evaluations, not the obvious 110 — an individual selected but neither crossed nor mutated is the same object and is never re-evaluated |
| 2 | `ga_flow_02_injected_fitness.py` | the fitness function becomes an argument | One driver, two problems: the sine run reproduces step 1 exactly, the target run lands **0.0038** from +4.200. The only difference between the runs is which function was passed in |
| 3 | `ga_flow_03_population_metrics.py` | `population_metrics()`, the history table, its figure | On the sine problem, average fitness climbs from −1.5588 to +0.6228 while gene spread falls from **6.665 to 0.064** by generation 8. The champion stands still while the population collapses underneath it |
| 4 | `ga_flow_04_stopping_condition.py` | `PATIENCE`, `MIN_IMPROVEMENT`, stop on stagnation, control runs | The rule saves **49 to 51** of the 60 allowed generations and costs less than `MIN_IMPROVEMENT` on both problems. It also exposes a wart: on 1 of 2 problems the final generation no longer holds the champion — nothing here protects it |
| 5 | `ga_flow_05_stepped_landscapes.py` | `staircase()`, the third problem, the verdict against the dense-grid reference | The staircase does **not** break the rule: the run reaches `+2.0000`, the landscape's dense-grid reference, and the rule stops it having lost nothing. Every one of the three problems matches its grid reference, shortfall `+0.0000` |

**These numbers are verified against actual output.** Any slide, handout or
translation that states a figure must state one of these. Re-run the script
rather than trusting this table if the code has changed.

### A refuted hypothesis, and a correction

Step 5 was designed to catch the stopping rule out. The plan: a stepped landscape
on which a patience rule would mistake "has not yet found the next step" for "has
finished", quit early, and lose fitness a longer run would have found.

**It does not happen.** The run climbs the staircase, reaches its top step, and
the rule stops it six generations later having lost nothing — at every tent height
and step width tried. The hypothesis is refuted, and the reason is worth more than
the hypothesis was: these steps *rise*, so every step boundary is a comparison
selection can act on. Flat in places is not the same as directionless. The step
now reports that, and keeps the instrument the investigation produced — the
comparison of each run against a dense-grid reference, which is what
distinguishes "the rule cost nothing" from "the run succeeded".

**The correction.** A first version of this step reported the staircase ending
`+0.2000` short of an optimum of `+2.0000`. That number was an artefact. With the
tent at exactly 10.0, the condition for the top step holds at the single point
x = 4.0 and nowhere else — an optimum of **zero width**, which the brute-force grid
could see (its spacing of 0.001 happens to contain 4.0) and which no search could
ever land on. The defect was found while building Lesson 05, whose step 4 tried to
reproduce the shortfall and could not. The tent is now 10.5, the top step is 0.5
wide, and the shortfall it was invented to explain does not exist. The general
rule is recorded in `staircase()`'s docstring: *an optimum the grid can see and
the search cannot is not a benchmark, it is a bug.*

## How the code is marked up

Two devices, mandatory course-wide:

- **The recipe** — each script after the first opens with a `CHANGES FROM …` block
  listing, in order, the changes that turn the previous script into this one.
- **The bands** — each recipe item has its own `# --- NEW (n) name ---` band at the
  site of the change. Everything outside a band is code the students already have.
  Step 4's first item is a changed constant, so it marks the line instead:
  `MAX_GENERATIONS = 60   # --- CHANGED ---`.

## Running it

```bash
uv run en/Lesson_02_GA_Flow/src/ga_flow_01_the_five_phases.py
```

Self-contained, no arguments, figures saved to `../figures/`. The whole sequence
runs in about **8 s**.

## Folder contents

```
Lesson_02_GA_Flow/
├── README.md      this file — the lesson's contract
├── src/           the five numbered scripts
├── figures/       one figure per script (step 5 saves two)
```

## Paths and status, September 25, 2026

- Shared scripts: [`src/`](src/), 5 steps; reproducible figures in [`figures/`](figures/).
- Independent study: [`student/lesson_02.pdf`](student/lesson_02.pdf) and [`student/lesson_02_ga_flow.ipynb`](student/lesson_02_ga_flow.ipynb), with TeX source beside the PDF.
- Instructor use: [`instructor/presentation/lesson_02_compact.pdf`](instructor/presentation/lesson_02_compact.pdf) and [`instructor/notebooks/lesson_02_compact.ipynb`](instructor/notebooks/lesson_02_compact.ipynb). The compact deck has 9 slides; the notebook has explanatory comments next to its code.

The extensive route retains the complete progression; the compact pair is self-contained for roughly 40 minutes. Examples are synthetic. Transfer to the Planta Física challenge is a modeling choice, not a partner-validated outcome. Student distribution remains blocked until rights for Gridin-derived material are documented. Canvas and hosted Colab were not changed or tested.
