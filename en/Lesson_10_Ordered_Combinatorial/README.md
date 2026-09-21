# Lesson 10 — Ordered Combinatorial Optimisation

This lesson develops two problems in which chromosome order and uniqueness are
part of correctness. The TSP sequence preserves a permutation of 48 cities; the
team sequence preserves eleven distinct players in role-specific roster slots.

- **Source in the book:** Grid Gridin, *Learning Genetic Algorithms with Python*,
  Chapter 10 — 2 sections, 2,883 words, 12 figures and 523 lines across 9 files.
- **Target duration:** 53 minutes by book weight. Use TSP as the core lesson and
  the roster as an application or continuation.
- **Prerequisites:** Lessons 04, 05 and 09.

## Mathematical model

A legal tour is a permutation \(\pi\) and has closed length
\(L(\pi)=\sum_i d(\pi_i,\pi_{i+1})\), with the final city joined to the first.
OX1 and inversion preserve permutation membership; ordinary one-point crossover
does not. A legal roster has eleven distinct identifiers in role-specific
positions, total cost at most the budget, and objective equal to total skill.
The experiments certify legality and compare against stated baselines; they do
not certify globally optimal tours or rosters.

## What the student leaves with

1. A legal ordered chromosome contains each required member exactly once.
2. Ordinary one-point crossover usually duplicates and loses members.
3. Ordered crossover, swap and inversion preserve legality but move differently.
4. A valid GA result still needs a baseline.
5. Position, uniqueness, budget and quality should remain separately auditable.

## The ten steps

### Travelling salesperson

| # | Script | Verified result |
|---|---|---|
| 1 | `tsp_01_the_route.py` | The supplied instance contains 48 cities; the identity route is legal but long |
| 2 | `tsp_02_random_population.py` | All 500 shuffled routes are legal, with a wide length distribution |
| 3 | `tsp_03_ordered_crossover.py` | One-point crossover produces 0/1,000 legal children; ordered crossover produces 1,000/1,000 |
| 4 | `tsp_04_mutation.py` | Swap and inversion preserve every city; their mean length changes are +451.0 and +238.9 |
| 5 | `tsp_05_the_full_search.py` | The GA route is legal but 57.2% longer than nearest neighbour after 10,100 evaluations |

### Football roster

| # | Script | Verified result |
|---|---|---|
| 1 | `team_01_the_roster.py` | The source-backed pool contains 12 candidates for each role block and an explicit €600m budget |
| 2 | `team_02_random_rosters.py` | All 1,000 rosters contain eleven distinct players; only 440 meet the budget |
| 3 | `team_03_budget_repair.py` | Repair makes 1,000/1,000 teams affordable and preserves uniqueness |
| 4 | `team_04_ordered_operators.py` | All 100 children remain legal; mean skill rises from 979.6 to 985.3 |
| 5 | `team_05_the_full_search.py` | The GA reaches skill 991 at €597.5m and ties the best equal-evaluation repaired-random roster |

The TSP failure and roster tie are part of the lesson. Neither result should be
rewritten as an algorithmic win.

## Running it

From `MA2015 lessons/`:

```bash
uv run en/Lesson_10_Ordered_Combinatorial/src/tsp_01_the_route.py
```

`att48_xy.txt` and `players_20.csv` remain beside the scripts. Lesson 12 may
reuse the TSP concepts and names, but its scripts must restate what they need
locally rather than importing across lessons.

## Status

| Piece | State |
|---|---|
| Code | Done. Ten numbered scripts run clean and pass the recipe/band checker. |
| Figures | Done. One generated figure per numbered script. |

## Guided notebooks

Student entry point: [`notebooks/README.md`](notebooks/README.md). This lesson has 2 independently runnable sequences: [`lesson_10_tsp.ipynb`](notebooks/lesson_10_tsp.ipynb), [`lesson_10_team.ipynb`](notebooks/lesson_10_team.ipynb). The retained outputs were validated on date omitted against the frozen scripts in three fresh-kernel path modes; see the notebook README for environment details and the explicit hosted-Colab gap.

