# Lesson 13 — Improving Performance

The last lesson of the course, and the only one about what a genetic algorithm
costs rather than about what it finds. One roster problem is carried through
seven steps while the search itself is held fixed: the answer is the same
`-87` in six of them, and the whole content of the lesson is the falling bill
underneath it. The seventh step is the exception, and it is last for that
reason.

- **Source in the book:** Gridin, *Learning Genetic Algorithms with Python*,
  Chapter 13 (*Improving Performance*) — 5 sections, 1,717 words, 6 figures,
  644 lines across 10 files (`caching/`, `coarse/`, `snapshot/`, `parallel/`).
  All five sections are covered. Section 13.1 was **missing from this lesson's
  earlier material** and is now step 2.
- **Budget:** 32 min (Chapter 13 is 4.9% of the book). Seven steps is more than
  32 minutes comfortably holds — see *If the session is short* below.
- **Prerequisites:** Lessons 02 (the flow and its evaluation census) and 09
  (binary chromosomes, and a scheduling problem in the same family).

## Mathematical model

Runtime separates objective calls, elementary work per call, serialization and
process overhead. Cache hit rate is (h=H/(H+M)); if objective work dominates,
the ideal work-only speedup is (1/(1-h)), while observed wall-clock speedup
also includes lookup, serialization and process overhead. Caching is valid only for a deterministic objective keyed by
the complete immutable genome. An exact restart needs the population,
fitness/cache state, generation counters and pseudorandom state. Workers have
separate memory under Windows spawn and Unix fork copy-on-write, so worker-local
counters are not shared. For surrogate $\tilde f$, error is
$e(x)=\tilde f(x)-f(x)$; candidates selected with $\tilde f$ must be evaluated
afterward using the real $f$.

## The measuring rule this lesson runs on

A lesson about speed is the easiest place in a course to print something false.
A wall-clock number is a fact about one laptop on one afternoon; it is not a
property of the algorithm, and a slide that quotes it as one will be wrong in
the next room.

So every script carries two counters and states its results in them:

- `COST["calls"]` — calls to the fitness function.
- `COST["work"]` — elementary steps inside those calls, one per (employee,
  shift) cell examined. One full evaluation costs **210 work units**.

Both are identical on every machine, every Python and every run with this seed.
**Exactly one script prints seconds** — step 5, which has to, because the
question it asks is about processes. It prints them next to a paragraph saying
what they are worth, and its argument rests on the ratio between two timings
taken on the same machine in the same run, never on either number alone.

## What the student leaves with

1. Evaluating once and storing the result is the cheapest real optimisation
   available, and it is free: same seed, same answer, 54% fewer evaluations.
2. A cache is worth what the search's repeats are worth — measured, not
   assumed. The same dictionary returns 14.4% or 3.2% depending on the mutation
   rate, and nothing about the cache changes between those two runs.
3. Deferring evaluation until breeding is finished is worth more than the
   process pool it was written to enable.
4. Multiprocessing removes no work at all; it relocates it, and it takes the
   program's globals — cache and counters included — out of reach on the way.
5. Only approximation actually makes the problem smaller, and it is the only
   technique here that has to be judged instead of verified.

## The running example

The prefix is `shifts_`. One problem throughout: a 5-employee, 7-day,
3-shift-per-day roster, a 105-bit chromosome, maximising
`−(staffing deviations + 5 × rest violations)`, so a flawless roster scores 0.
It was chosen because its objective splits cleanly in half — one term walks the
roster by shift, the other by employee — which is what makes step 7's coarse
version an honest 50% cheaper rather than a fudge, and because Lesson 09 has
already taught a scheduling problem, so no lesson time is spent on the domain.

`SEED = 3` throughout. With it, the full objective reaches fitness **−87**, and
every step from 2 to 6 reproduces exactly that.

## The seven steps

| # | Script | What it adds | What its output proves |
|---|---|---|---|
| 1 | `shifts_01_the_cost_of_a_run.py` | the roster problem, a plain GA, and the two counters | The run builds **428** individuals and calls fitness **930** times — **2.17 evaluations per individual**, split SELECT 300 / BREED 0 / STATS 630. **195,300** work units, best fitness **−87** |
| 2 | `shifts_02_evaluate_once.py` | **book §13.1**: evaluate in the constructor, store the value | **428** calls, **89,880** work units: **54.0%** less, with SELECT and STATS at **0**. Best fitness still **−87** — the saving cost nothing |
| 3 | `shifts_03_the_cache.py` | **§13.2**: a genome→fitness dictionary, plus a hit-rate scan | **19** hits in 428 builds, **409** distinct genomes, 4.4% less work, answer unchanged at **−87**. The scan shows the hit rate is the *search's* property: **14.4%** at mutation p=0.10, **6.1%** at 0.25, **4.4%** at 0.50, **3.2%** at 0.75 |
| 4 | `shifts_04_snapshot.py` | **§13.4**: dump and restore the population, and the cache | 30 rosters restore for **30 evaluations / 6,300 work units** from genes alone, and for **0 / 0** when the cache is restored first. The cache file is **14.0×** the size of the gene file — a bad trade at 210 units a call, an excellent one at a minute a call |
| 5 | `shifts_05_parallel_evaluation.py` | **§13.5**: batch the generation, then hand it to a `Pool` | Batching alone drops the run from 409 to **291** evaluations (**−28.9%**), because a crossed-then-mutated child used to be evaluated twice. The pooled run gives the identical answer and the identical best-so-far in all 10 generations — while the parent's counters read **0 evaluations, 0 work units** |
| 6 | `shifts_06_cache_across_processes.py` | the parent owns the cache and totals the workers' costs | Pooled run now reports **291 / 8 hits / 61,110**, matching the sequential run on all three, answer still **−87**. Step 5's private per-worker caches had been discarding every hit |
| 7 | `shifts_07_coarse_fitness.py` | **§13.3**: drop the expensive half of the objective | **105** work units per call instead of 210, **51.0%** less work for the run. The coarse search reaches **0 on its own objective** — and its roster scores **−160** on the real one against **−87**, with **32** rest violations instead of 17 |

**These numbers are verified against actual output.** Any slide, handout or
translation quoting a figure must quote one of these. The only quantity
deliberately absent from this table is a time in seconds.

### The result that was not planned

Step 5 was written to introduce `multiprocessing`. Its largest measured effect
has nothing to do with it: restructuring so that a whole generation can be
dispatched at once also stops the program evaluating the intermediate children
that mutation immediately replaces, and that alone removes 118 of 409
evaluations. The pool, on this machine, changed no count whatsoever. The step
now reports the batching saving first and the pool second, in that order,
because that is the order the evidence puts them in.

### Where this lesson meets Lesson 02

Lesson 02's step 1 counts **107** fitness evaluations in a run where 110 look
obvious, and explains the discrepancy: an individual that is selected but
neither crossed nor mutated is the same object and is never re-evaluated. That
count is only true of a program that has already applied §13.1 — which Lesson
02 does without naming. Step 2 here is the technique behind it, and step 5
extends it: an individual that is crossed *and* mutated need not be evaluated
twice either.

### If the session is short

Seven steps do not fit 32 minutes at the pace the rest of the course is taught.
Cut **step 4 (snapshots)**: it is the most self-contained of the seven, and
nothing in steps 5 to 7 calls into it. Its four functions do ride along in those
scripts, because each script is the whole program as it stood — so if step 4 is
cut, say in one sentence what `dump_population` and `dump_cache` are there for
rather than leaving a student to wonder. Steps 5 and 6 must be taught together or
not at all: step 5 ends on a broken meter that step 6 repairs.

## How the code is marked up

Every script after the first opens with a **recipe**: the ordered list of
changes that turns the previous script into this one. Each recipe item has one
**band** in the body, at the site of the change, with the same number and name:

```python
# --- NEW (1) coarse_fitness() -------------------------------------------------
```

Bands mark what is new *in this step only*. A step that changes a value rather
than adding code marks the line instead, with `# --- CHANGED ---`; step 3 has
en/Lesson_13_Performance/src` to verify the correspondence.

## Running it

From `MA2015 lessons/`:

```bash
uv run en/Lesson_13_Performance/src/shifts_01_the_cost_of_a_run.py
```

Every script is self-contained, takes no arguments, fixes `SEED = 3` and saves
its figure under `figures/`. The seven together run in **about 11 seconds**.
Steps 5 to 7 start a `multiprocessing.Pool` sized `min(4, os.cpu_count())` and
print the core count they found, so their console output differs by machine in
that one line; every counted result does not.

## Folder contents

| Path | Contents |
|---|---|
| `src/` | the seven numbered scripts |
| `figures/` | one PNG per script, named after it |
| `snapshots/` | written by step 4 at run time (`shifts_population.json`, `shifts_cache.json`) |

## Status

| Piece | State |
|---|---|
| Code | Done. Seven numbered scripts, all exit clean, all pass the band checker. |
| Figures | Done. One generated figure per script. |

## Guided notebooks

Student entry point: [`notebooks/README.md`](notebooks/README.md). This lesson has 1 independently runnable sequence: [`lesson_13_performance.ipynb`](notebooks/lesson_13_performance.ipynb). The retained outputs were validated on date omitted against the frozen scripts in three fresh-kernel path modes; see the notebook README for environment details and the explicit hosted-Colab gap.

