# Lesson 01 — Your First Genetic Algorithm

A genetic algorithm is built in front of the students, one operator at a time,
until it solves a one-variable maximisation problem — and then the same code is
run again with a different seed and fails, which is what the rest of the course
exists to address.

- **Source in the book:** Gridin, *Learning Genetic Algorithms with Python*,
  Chapter 1 (*Introduction*) — 4 sections, ~1,900 words, 8 figures, one 91-line
  script (`03-referencias/…/Chapter01/your_first_genetic_algorithm.py`).
- **What the refactor does to it:** that single script is split into **seven
  numbered steps**, so each operator gets its own runnable file and its own
  slide.
- **Target duration:** the seven-step classroom route fills about 50 minutes.
  Steps 1–2 are fast; steps 3–5 carry the lesson; steps 6–7 are a payoff and a
  cliffhanger. The code-reading appendix and notebook experiments are optional
  self-study material outside that route.
- **Prerequisites:** none. This is the first technical lesson of the course.

## Mathematical model

The search problem is
\[
\max_{x\in[-10,10]} f(x),\qquad f(x)=\sin(x)-0.2|x|.
\]
An individual stores one candidate gene \(x\); its fitness is the evaluated
value \(f(x)\). A population is a finite sample, so its best member is only the
**best observed** value. A dense grid gives a sampled reference, not an exact
proof of the continuous maximizer. The two seeded runs establish possible GA
behaviours; neither seed estimates a success probability.

## What the student leaves with

1. An optimisation problem can be attacked without derivatives, and without
   knowing anything about the function beyond how to evaluate it.
2. A GA works on a **population**, not a point.
3. The three operators do three different jobs: selection **discards**,
   crossover **recombines**, mutation **invents**. Removing any one of them
   breaks the search in a specific, predictable way.
4. A GA is **not** guaranteed to find the optimum. The same code and parameters,
   driven by a different random sequence, can get stuck.

## The running example

```
f(x) = sin(x) - 0.2 * |x|,   x in [-10, 10]      (maximise)
```

Chosen because it is one-dimensional (everything can be drawn on one axis),
multimodal (four peaks, so getting stuck is a real risk rather than a story),
and has a global maximum that is not the peak nearest the origin.

## The seven steps

| # | Script | What it adds | What its output proves |
|---|---|---|---|
| 1 | `first_example_01_the_landscape.py` | `objective()`, the plot | There are four peaks; hill-climbing stops at whichever one it starts on |
| 2 | `first_example_02_random_population.py` | `Individual`, `create_random()`, the population | Ten blind guesses land nowhere near the optimum |
| 3 | `first_example_03_selection.py` | `select_tournament()` | No new `x` value appears — selection only copies; 4 of 10 individuals go extinct |
| 4 | `first_example_04_crossover.py` | `clamp()`, `crossover_blend()`, `crossover()` | 14 of 20 children fall outside the parents' interval — that is what `alpha > 0` buys |
| 5 | `first_example_05_mutation.py` | `mutate_gaussian()`, `mutate()` | From `x = -4.6`, the seeded sample observes 0/2000 arrivals with sigma = 1.0 and 110/2000 with sigma = 3.0 |
| 6 | `first_example_06_the_full_loop.py` | the generational loop, the convergence plot | Ten generations find `x = +1.372, f = +0.706`, near the highest peak; a 400-point grid reports `x = +1.378, f = +0.706` |
| 7 | `first_example_07_local_optimum.py` | nothing; `SEED` changes from 52 to 16 | In this ten-generation run, the same algorithm settles on `x = -4.417, f = +0.073` |

**These numbers are verified against actual output.** Any slide, handout or
translation that states a figure must state one of these, not a plausible
substitute. Re-run the script rather than trusting this table if the code has
changed since.

## How the code is marked up

Two devices, both mandatory course-wide, both already
applied here:

- **The recipe** — every script after the first opens with a `CHANGES FROM …`
  block in its docstring: an ordered list of the changes that turn the previous
  script into this one. That order is the order to type them in class, and the
  order the slides must follow.
- **The bands** — each recipe item has its own `# --- NEW (n) name ---` band in
  the body, at the site of the change, numbered and named to match the recipe.
  Everything outside a band is code the students already have. Step 7 changes a
  value instead of adding code, so it marks the line: `SEED = 16   # --- CHANGED ---`.

A slide for step *n* is therefore mechanical to write: it is the recipe of
script *n*, in order, with the banded code as the listings.

Each script also explains its supporting Python locally, so it can be read on
its own. Comments and focused docstrings cover imports, paths, random state,
arrays, object references, comprehensions, formatting, validation, and plotting.
The slides keep the seven-step classroom route concise and collect these
language notes in a clearly marked code-reading appendix.

Maintainers preparing another student notebook should follow the shared
[guided notebook workflow](../_NOTEBOOK-WORKFLOW.md) rather than inferring the
process from this lesson alone.

## Running it

For the classroom route, open the [compact Lesson 01 notebook](notebooks/lesson_01_intro_compact.ipynb).
It combines the seven script-derived milestones with executable code, inline
figures, prediction questions, and final checks. The [complete notebook](notebooks/lesson_01_intro.ipynb)
keeps the detailed code-reading notes and optional experiments as reference.
[Notebook instructions](notebooks/README.md) explain how to run either version
locally, in VS Code, or in Colab.

From the repository root, with `uv` handling the environment:

```bash
uv run en/Lesson_01_Intro/src/first_example_01_the_landscape.py
```

Every script is self-contained: no imports across steps, no imports across
lessons, no command-line arguments. Figures are **saved** to `../figures/`,
never shown, so the whole sequence runs unattended.

## Folder contents

```
Lesson_01_Intro/
├── README.md      this file — the lesson's contract
├── src/           the seven numbered scripts
├── notebooks/     complete and compact guided notebooks, with saved outputs
├── figures/       generated plots (never edited by hand, never committed by hand)
└── slides/        complete and compact Beamer decks
```

## Status

| Piece | State |
|---|---|
| Code | Done. Seven independently readable scripts; all run clean and preserve the original seeded numerical results. |
| Complete notebook | Guided English walkthrough with 104 cells (34 executed code cells), detailed reading notes, optional experiments, checks, and seven embedded figures. |
| Compact notebook | Classroom route with 29 cells (9 executed code cells), seven embedded figures, prediction prompts, checks, and the same baseline results. |
| Complete slides | 48-page Beamer deck with the code-reading appendix. |
| Compact slides | 17-page Beamer deck, compiled twice without overflows and reviewed visually; code listings follow the canonical script bands. |
| Figures | Done. All seven scripts save a figure named after the script. Steps 3–5 added date omitted (selection multiplicities, blend children inside/outside, mutation reach). Step 5 no longer imports `scipy`. |

