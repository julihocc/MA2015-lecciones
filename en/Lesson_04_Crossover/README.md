# Lesson 04 — Crossover

Seven crossover operators, all of them applied to the same two parents and
measured with the same instrument, so that the lesson ends not with a parade of
techniques but with one table showing that the operators split into two families
which cannot be swapped. A crossover chosen for the wrong representation does
not produce a worse solution — it produces something that is not a solution.

- **Source in the book:** Gridin, *Learning Genetic Algorithms with Python*,
  Chapter 4 (*Crossover*) — 7 sections, ~2,748 words, 14 figures, 157 lines
  across 7 files.
- **Budget:** 51 min (Chapter 4 is 7.8% of the book).
- **Prerequisites:** Lesson 01 (BLX-alpha and `clamp()` were introduced there,
  and are not re-derived here) and Lesson 03 (the method: measure one operator
  many times from a fixed seed, and read the columns).

## Mathematical model

For fixed parents, an operator's **reach** is the support of its child
distribution. One- and (n)-point crossover copy alternating contiguous
parental segments; this implementation restricts (n)-point cuts to
`range(1, L-1)`, unlike the canonical (1,\ldots,L-1) set. Uniform crossover has \(2^L\) possible first children when all
parental gene pairs differ. Linear crossover reaches its two affine children.
BLX-\(\alpha\) samples uniformly from
\([m-\alpha d,M+\alpha d]\), where \(d=M-m\), so the probability of leaving
the parental interval is \(2\alpha/(1+2\alpha)\). OX1 preserves membership and
uniqueness only for permutations. Samples illustrate support; enumeration is
what establishes an exact reachable-set count.

## What the student leaves with

1. **Crossover is defined by what it can reach.** Given two fixed parents, an
   operator's reachable set is a finite, countable thing. One-point crossover on
   a six-gene chromosome can produce exactly **10** different children; uniform
   crossover can produce **64**, which is all of them.
2. **"More crossover points" is not a strength dial.** 3-point crossover reaches
   **8** children here — *fewer* than one-point's 10 — because n cut points drawn
   from the interior positions admit fewer segment patterns than one free cut.
3. **Cut-and-swap operators copy; arithmetic operators compute.** The cut family
   puts a value the parents did not have into **0.00%** of gene slots. Blend puts
   one into **99.57%** — and at alpha = 0.5 it also leaves the search space, so
   only **50.08%** of its children are legal until `clamp()` repairs them.
4. **Fitness-driven crossover buys safety with motion.** It is never worse than
   the parents (**0.00%** of draws), and it pays for that by handing back the two
   parents unchanged in **16.50%** of draws.
5. **The two families do not overlap.** On a route of nine stops, one-point,
   n-point, blend and linear produce a legal child **0.00%** of the time; uniform
   manages **0.95%**, and every one of those is a parent handed straight back.
   Ordered crossover is legal 100% of the time on routes — and on real-valued
   genes it silently returns the parents. **Zero** operators are useful on both.

## The running example

One fixed pair of parents, and the cloud of children an operator can make from
them. That cloud is what the prefix `offspring_` names, and one function
(`offspring_report`) measures it for every operator over **2,000 draws** from a
fixed seed, so the rows of every table are comparable by construction.

The parents are Gridin's own: `SEED = 3`, six genes drawn uniformly from
`[0, 10]` — `[2.38 5.44 3.70 6.04 6.26 0.66]` and `[0.13 8.37 2.59 2.34 9.96
4.70]`. From step 6 the same seed produces two routes through nine delivery
stops, which is the chapter's other representation and the size Gridin uses in
`order.py`.

The instrument reports four numbers per operator: how many **different children**
it reached, the share of gene slots holding a **new** value, the share of
children that are a **clone** of a parent, and the share that are **legal**. The
first three describe the operator; the last one describes the marriage between
the operator and the representation, and it is where the lesson lands.

## The seven steps

| # | Script | What it adds | What its output proves |
|---|---|---|---|
| 1 | `offspring_01_two_parents.py` | the parents, the instrument, and the do-nothing baseline | The zero row: 2 children reached, 0.00% new genes, 100.00% clones, 100.00% legal. Also the number every later step is measured against: choosing freely between the parents allows 2⁶ = **64** chromosomes |
| 2 | `offspring_02_one_point.py` | `crossover_one_point()`, and the enumeration of its whole reach | 2,000 draws produce **10** distinct children; enumerating the 5 cut points produces the same 10. **0.00%** new genes: the operator chooses which parent, never what value |
| 3 | `offspring_03_more_cuts.py` | `crossover_n_point()`, `crossover_uniform()` | Reach is not monotone in cuts: one-point **10**, 2-point **12**, 3-point **8**, uniform **64 of 64**. Uniform's price: **3.20%** of its children are a parent handed back |
| 4 | `offspring_04_blend.py` | `clamp()`, `crossover_blend()`, `crossover_linear()` | The first computed genes (**99.57%** new) and the first illegal children. alpha = 0: **0.00%** of genes beyond the parents, 100% legal — Lesson 01's inward collapse, measured. alpha = 0.5: **49.63%** beyond, only **50.08%** legal; with `clamp()`, 100% legal and still 49.63% beyond. Linear: 100.00% new genes and exactly **2** reachable children |
| 5 | `offspring_05_fitness_driven.py` | `objective()`, `crossover_fitness_driven()` | Blind blend is worse than its parents in **41.70%** of draws — crossover guarantees nothing. The fitness-driven rule is worse in **0.00%**, and pays for it: **16.50%** of draws return the parents unchanged, and only **56.10%** of what it returns is a child |
| 6 | `offspring_06_the_divide.py` | the nine stops, `tour_length()`, `is_legal_tour()` | The same operators, a route instead of a point. **8 of 8** cut points give a route that visits one stop twice and another never. Legal children: one-point **0.00%**, 3-point **0.00%**, blend **0.00%**, linear **0.00%**, uniform **0.95%** — and the count of uniform's legal children that were not simply a parent is **0** |
| 7 | `offspring_07_ordered.py` | `crossover_order()`, and the comparison the lesson was built for | OX1 is legal **100.00%** on routes and keeps **7.17 of 9** parent legs, against **3.71** for a random route. Handed the real-valued parents it does not crash: it reaches **2** children, **100.00%** of them a parent. Final table: operators useful on both representations — **0** |

**These numbers are verified against actual output.** Any slide, handout or
translation that states a figure must state one of these. Re-run the script
rather than trusting this table if the code has changed.

A full genetic algorithm on a route is Lesson 10. This lesson stops at what
the operators do to one pair of parents.

## Two results that were not planned

- **Step 3 was designed to show that more cut points mean more mixing.** They do
  not: 3-point reaches 8 children where one-point reaches 10, because Gridin's
  `n_point` samples its cuts from `range(1, len - 1)` and three cuts out of four
  interior positions admit only four segment patterns. The step was rewritten
  around the measurement. The count is length-dependent and the script says so —
  what generalises is that "number of cuts" is not the quantity that matters,
  and that uniform crossover is the only member of the family that reaches the
  complete set.
- **Step 7 was designed to end with ordered crossover working.** It ends with
  ordered crossover *failing silently* instead. Handed two parents that share no
  gene values, OX1's fill loop never skips anything and overwrites its own
  swath, so it returns the parents unchanged — legal, plausible, and completely
  inert. That is the more useful ending: the illegal route of step 6 is loud and
  gets found, and this one does not.

  Two smaller honest results sit in the same step. OX1 children average a route
  of **52.95** against **51.23** for 2,000 random routes: on nine stops, and with
  two mediocre parents, inheriting structure is visible (7.17 legs vs 3.71) while
  inheriting quality is not. Crossover supplies the first; only selection
  supplies the second.

## How the code is marked up

Two devices, mandatory course-wide:

- **The recipe** — each script after the first opens with a `CHANGES FROM …`
  block listing, in order, the changes that turn the previous script into this
  one. That order is the order to type in class and the order the slides follow.
- **The bands** — each recipe item has its own `# --- NEW (n) name ---` band at
  the site of the change, numbered and named to match. Everything outside a band
  is code the students already have.

Checked mechanically:

```bash
```

## Running it

```bash
uv run en/Lesson_04_Crossover/src/offspring_01_two_parents.py
```

Each script is self-contained, takes no arguments, and saves its figure to
`../figures/` under its own name. The whole sequence runs in about **26 s**.

## Fit against the budget

Seven steps in 51 minutes is tight — Lesson 03 fits six into 48. Steps 2 and 3
are the pair to merge if the session runs short: they share one idea (the
cut-and-swap family reaches a countable set and computes nothing), and step 2's
enumeration can be shown from step 3's table instead of on its own. Steps 6 and 7
must not be cut or split across sessions; the divide only teaches if both halves
land in the same hour.

The book's 14 figures are not reproduced. Seven are enough — one per script,
each one showing that script's own measurement.

## Folder contents

```
Lesson_04_Crossover/
├── README.md      this file — the lesson's contract
├── src/           the seven numbered scripts
├── figures/       one figure per script, plus `offspring_02_one_point_schematic.png`
```

## Status

| Piece | State |
|---|---|
| Code | Done. Seven scripts, all run clean, one figure each. |
| Figures | Done — seven named after their scripts, plus the step-2 schematic `offspring_02_one_point_schematic.png`. |
| Slides | Done. |
| Spanish mirror | Scripts exist under `es/Leccion_04_Cruza/`. Remaining work is polish (the lesson workflow), not a rewrite. |

## Guided notebooks

Student entry point: [`notebooks/README.md`](notebooks/README.md). This lesson has 1 independently runnable sequence: [`lesson_04_crossover.ipynb`](notebooks/lesson_04_crossover.ipynb). The retained outputs were validated on date omitted against the frozen scripts in three fresh-kernel path modes; see the notebook README for environment details and the explicit hosted-Colab gap.

