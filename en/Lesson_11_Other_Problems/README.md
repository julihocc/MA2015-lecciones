# Lesson 11 — Other Common Problems

This lesson develops two small problems that pull in opposite directions. A
system of equations has an answer that can be *checked*, and a search space
small enough to walk. Graph colouring has no separate objective at all — the
constraint is the objective — and a space of 205 trillion colourings. Together
they are the course's occasion to ask the question it has so far avoided: when
is a genetic algorithm the wrong tool?

- **Source in the book:** Grid Gridin, *Learning Genetic Algorithms with Python*,
  Chapter 11 — 2 sections, 1,429 words, 8 figures and 317 lines across 6 files.
  The lightest chapter in the book.
- **Target duration:** 27 minutes by book weight. Eight steps, four per problem.
- **Prerequisites:** Lessons 03, 04, 05 and 08. Lesson 08 in particular: it is
  the lesson that established you usually cannot tell whether a run succeeded,
  and both problems here are cases where you can.

The brief asks for five to eight steps per example. This lesson has four per
example, deliberately: 27 minutes will not carry ten steps, and neither problem
has ten teachable milestones in it. Padding either sequence would have meant
inventing content the chapter does not contain.

## Mathematical model

The equation search minimizes
\(R(x,y,z)=|f|+|g|+|w|\) over
\([-20,20]^3\cap\mathbb Z^3\), a space of \(41^3=68{,}921\) triples.
Residual zero is a verifiable witness; enumerating the box proves uniqueness
there. For the 30-vertex graph, the three-colour chromosome space has
(3^{30}=205{,}891{,}132{,}094{,}649) assignments. Graph colouring minimizes
\(C(c)=\sum_{(u,v)\in E}I[c_u=c_v]\). Independent uniform three-colourings
have \(\mathbb E[C]=|E|/3\). A zero-conflict colouring proves feasibility;
exact backtracking, not the GA, proves that two colours are impossible.

## What the student leaves with

1. A verifiable objective changes what a stopping rule can be. Both sequences
   stop on a proof (`residual == 0`, `conflicts == 0`), not on a budget.
2. When the constraint *is* the objective, the landscape near the answer is a
   plateau: one conflict and two conflicts look almost the same to the search.
3. Fitness-driven operators — children that must beat their parents, mutations
   kept only when they help — buy a better average and cost extra evaluations.
   Both halves of that trade are measured here, not asserted.
4. A finite integer box can simply be enumerated, and enumeration answers
   questions a genetic algorithm cannot: how many solutions exist, and whether
   the instance is solvable at all.
5. A genetic algorithm earns its place when the space stops being walkable —
   not when it happens to beat a loop you could have written in four lines.

## The eight steps

### System of equations — the answer can be checked

The example is Gridin's non-linear integer system in `x, y, z`. It is chosen
because it is the rare problem in this course whose answer is verifiable by
substitution, and whose search space is finite and small.

| # | Script | Verified result |
|---|---|---|
| 1 | `equations_01_the_system.py` | Genes are integers in `[-20, 20]`, so the whole space is 68,921 triples. Residuals are integers, not floats: along the line `(y, z) = (1, 1)` they run from 3 digits at `x = 2` to 31 digits at `x = 20` |
| 2 | `equations_02_random_population.py` | 400 uniform draws (0.58% of the box) find 0 solutions; the best is `(0, 1, 2)` with residual 453, while the median draw has a 957-digit residual and the worst 14,187 digits |
| 3 | `equations_03_the_full_ga.py` | Seed 3 reaches residual 0 at generation 5 after 2,794 evaluations, at `x = -6, y = 2, z = 3`; `f`, `g` and `w` are each individually 0 |
| 4 | `equations_04_exhaustive_search.py` | All 5 seeds tried find the same root, in 5–28 generations and 2,794–13,764 evaluations. Enumerating the box costs 68,921 evaluations (about 10× more, roughly 3.7 s) and returns strictly more: there is **exactly one** root in the box |

Step 4 does not end the way it was planned to end. The expectation was that
exhaustive search would expose the genetic algorithm as wasteful. It does the
opposite — the GA is about ten times cheaper in evaluations and reliable across
every seed tried. The argument against it here is therefore not cost but
*claim*: enumeration proves uniqueness, has no seed, no population size and no
crossover rate, and always returns the same answer. The verdict flips with the
box, which costs `side³`: widening it to `[-100, 100]` is 8,120,601 triples,
118 times this run.

### Graph colouring — the constraint is the objective

The instance is Gridin's, kept verbatim: 30 vertices, 64 edges, degrees 3 to 6.
Fitness is the negated count of edges whose ends share a colour, so a legal
colouring and an optimal colouring are the same object.

| # | Script | Verified result |
|---|---|---|
| 1 | `colouring_01_the_graph.py` | 30 vertices, 64 distinct edges, mean degree 4.27, and 3³⁰ = 205,891,132,094,649 colourings. A random colouring at seed 7 breaks 26 of the 64 edges |
| 2 | `colouring_02_random_population.py` | 500 random colourings yield 0 legal ones; the sample mean is 21.51 conflicts against the predicted `|E|/3 = 21.33`, and the best draw still breaks 11 edges (17.2%) |
| 3 | `colouring_03_fitness_driven_operators.py` | Plain two-point crossover moves the population mean by +0.03 edges; the fitness-driven version by −2.32. Greedy recolouring moves the mean by −1.02 but leaves 206 of 500 individuals untouched, and its best individual (11) is *worse* than a blind recolour's best (9) |
| 4 | `colouring_04_the_full_search.py` | **1 of 4 seeds** reaches a legal colouring (seed 11, generation 15). Seeds 7 and 16 stall at 2 conflicts and seed 1 at 1 conflict, all still stuck after the full 200 generations. Backtracking finds a proper 3-colouring in 40 search nodes and proves no 2-colouring exists in 4 nodes — 0.69 ms for both |

This failure is the point of the lesson and must not be rewritten as a success.
Gridin reports the genetic algorithm solving this instance; measured over four
seeds it does so once. The instance really is 3-colourable — the backtracking
step proves it — so the run is a genuine miss, not an impossible problem. And
from the run alone you could not tell those two cases apart, which is exactly
lesson 08's problem appearing in a case where one function call settles it.

Step 3 also carries a small honest wrinkle worth a minute in class: greed
improves the average and gives up the lucky tail.

## How the code is marked up

Every script after the first in each sequence opens its docstring with a
**recipe** — the numbered, ordered list of changes that turn the previous script
into this one. Each recipe item has exactly one matching **band** in the body,
carrying the same number and the same name:

```python
# --- NEW (3) exact_colouring() ------------------------------------------------
def exact_colouring(k: int) -> Tuple[List[int], int]:
    ...
# ------------------------------------------------------------------------------
```

Everything outside a band is code the students already have. Bands never carry
forward: they mark what is new in *this* step only. A step that changes a value
or a signature rather than adding code marks the line itself with
`# --- CHANGED ---`; `equations_04` does this where `run()` starts taking a seed.

## Running it

From `MA2015 lessons/`:

```bash
uv run en/Lesson_11_Other_Problems/src/equations_01_the_system.py
```

All eight scripts run end to end in about 25 seconds, the slowest being
`colouring_04_the_full_search.py` (four 200-generation runs, ~7.5 s) and
`equations_04_exhaustive_search.py` (~4 s). Figures are always saved to
`figures/`, never shown.

**This lesson is the only one in the course that needs a package beyond numpy,
matplotlib and pandas:** `python-igraph`, used by `colouring_01` and
`colouring_04` for the graph layout. It is declared in `pyproject.toml`. See

## Folder contents

| Path | Contents |
|---|---|
| `src/` | Eight numbered scripts: `equations_01`–`equations_04`, `colouring_01`–`colouring_04` |
| `figures/` | Eight PNGs, one per script, each named after its producing script |

## Status

| Piece | State |
|---|---|
| Code | Done. Eight numbered scripts run clean and pass the recipe/band checker. |
| Figures | Done. One generated figure per numbered script. |

## Guided notebooks

Student entry point: [`notebooks/README.md`](notebooks/README.md). This lesson has 2 independently runnable sequences: [`lesson_11_equations.ipynb`](notebooks/lesson_11_equations.ipynb), [`lesson_11_graph_colouring.ipynb`](notebooks/lesson_11_graph_colouring.ipynb). The retained outputs were validated on date omitted against the frozen scripts in three fresh-kernel path modes; see the notebook README for environment details and the explicit hosted-Colab gap.

