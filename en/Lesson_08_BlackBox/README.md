# Lesson 08 — Black-Box Functions

This lesson removes the visual and exhaustive-search guarantees used in the
earlier lessons. Students first treat a five-gene function as an opaque system,
build a legal mixed-type chromosome and a GA around it, then open the box and
discover that the apparent optimum is a pole rather than a maximum.

- **Source in the book:** Grid Gridin, *Learning Genetic Algorithms with Python*,
    Chapter 8 (*Black-Box Function*) — 3 sections, 2,781 words, 4 figures and
  251 lines across 7 files.
- **Target duration:** 51 minutes by book weight.
- **Prerequisites:** Lessons 02–07, especially population diagnostics,
  selection, mutation repair and repeated-run evidence.

## Mathematical model

The legal search space is a Cartesian product of real intervals, a finite
integer set and a categorical set. Constructor repair is a projection onto
that mixed domain and therefore changes the distribution produced by the raw
operators. The exposed denominator is
(D=(n+1)^2(1+a+b)(120-x^2)r_\phi(x;b,n)+\tfrac12), where (r_\phi) is the
implemented trigonometric residual. When (D(x)) changes sign continuously, bisection
locates a zero of \(D\); the objective then has a pole. Values growing near a
pole do not identify a finite maximizer: the original optimization problem is
unbounded or ill-posed on any domain containing that singularity.

## What the student leaves with

1. A black-box interface defines legal inputs, but says nothing about the
   shape, continuity or boundedness of its output.
2. Mixed real, integer and categorical genes need explicit legality rules.
3. Silent repair can make a configured mutation much weaker than its nominal
   probability suggests.
4. Agreement among repeated searches does not prove that the reported value is
   an optimum.
5. Domain knowledge remains necessary when an optimiser reports success.

## The running example

`complicated_one(a, b, x, n, fun_name)` combines three bounded real genes, one
integer gene and one categorical gene. It was chosen because the same example
supports the entire argument: plotting fails, brute force becomes unreliable,
mixed-gene operators need repair, and the GA ultimately exposes a singularity.

## The seven steps

| # | Script | What it adds | What its output proves |
|---|---|---|---|
| 1 | `black_box_01_no_picture.py` | the declared domain and six one-dimensional slices | Five live slices agree on `x = -10.950`, but their maxima differ by `9.23×`; the picture has not resolved the value |
| 2 | `black_box_02_the_grid.py` | an evaluation timer, the book grid and four refinements | The book grid costs 105,525,000 calls; refinement makes the reported maximum grow 61,099-fold instead of converge |
| 3 | `black_box_03_the_chromosome.py` | legality repair and a mixed-type `Individual` | The constructor repairs all five illegal examples; the initial population samples only 10% of the declared `x` range |
| 4 | `black_box_04_the_operators.py` | per-gene crossover, mutation and a repair census | 41.5% of forced integer mutations snap back unchanged; coarse lattices can silence mutation completely |
| 5 | `black_box_05_the_search.py` | rank selection, the generation loop and diagnostics | The best fitness reaches `2.278e-09`, but the curve is still rising and 97.5% of the final population has left the initial `x` interval |
| 6 | `black_box_06_the_verdict.py` | twelve independent searches | Champions occupy a narrow `x` interval while their fitness differs by 21,504×, which is inconsistent with convergence to a finite maximum |
| 7 | `black_box_07_opening_the_box.py` | denominator isolation, bisection and a distance ladder | The apparent optimum is a pole near `x = -10.9535836`; the reported fitness measures distance to the singularity |

The numerical statements above were re-run on date omitted. Timing in step 2 is
machine-dependent; scripts and slides should quote the measured value from the
current run rather than freezing it as a universal constant.

## How the code is marked up

Every script after the first begins with a `CHANGES FROM ...` recipe. Each
recipe item has a matching numbered `NEW` band in the program body, or a
`CHANGED` marker when the step changes a value. The course checker confirms the
counts match for all seven scripts.

## Running it

From `MA2015 lessons/`:

```bash
uv run en/Lesson_08_BlackBox/src/black_box_01_no_picture.py
```

Each numbered script is self-contained and saves its figures under
`figures/` without opening an interactive window.

## Status

| Piece | State |
|---|---|
| Code | Done. Seven scripts run clean and pass the recipe/band checker. |
| Figures | Done. Seven figures are generated, one per numbered script. |

## Guided notebooks

Student entry point: [`notebooks/README.md`](notebooks/README.md). This lesson has 1 independently runnable sequence: [`lesson_08_black_box.ipynb`](notebooks/lesson_08_black_box.ipynb). The retained outputs were validated on date omitted against the frozen scripts in three fresh-kernel path modes; see the notebook README for environment details and the explicit hosted-Colab gap.

