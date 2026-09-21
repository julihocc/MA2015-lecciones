# Lesson 09 — Binary Combinatorial Optimisation

This lesson develops three binary problems in parallel: selecting items,
assigning employees to shifts and placing radars. Each example receives its own
five-step sequence so students can see how representation, feasibility and the
objective change while the binary operators remain recognisable.

- **Source in the book:** Grid Gridin, *Learning Genetic Algorithms with Python*,
  Chapter 9 — 3 sections, 6,157 words, 38 figures and 1,283 lines across 21 files.
- **Target duration:** 114 minutes by book weight. Teach it as three blocks or
  split it across sessions; fifteen steps do not fit one ordinary class.
- **Prerequisites:** Lessons 02–07.

## Mathematical model

All three chromosomes are binary, but their models differ. Knapsack maximizes
\(\sum_i v_i x_i\) subject to \(\sum_i w_i x_i\le C\). Radar minimizes
\(U(x)(|S|+1)+\sum_i x_i\), where \(U\) is the number of uncovered targets;
because at most \(|S|\) radars can be active, one fewer uncovered target always
dominates every possible radar-count change. For schedule bits (x_{eds}),
minimize (\sum_{eds}c_{eds}x_{eds}), subject to
(\sum_e x_{eds}=d_s) and (\sum_{ds}x_{eds}\le5). Repair maps raw bit strings
to feasible strings and changes the distribution the GA actually searches.

## What the student leaves with

1. A binary chromosome can encode selection, assignment or placement, but the
   meaning of one bit comes from the problem.
2. Feasibility must remain visible. A penalty, repair rule and exact benchmark
   answer different questions.
3. Selection cannot rescue a population whose infeasible members all tie.
4. Repair changes the distribution being searched and therefore must be
   measured rather than treated as free plumbing.
5. Small instances should be checked exactly; larger ones need honest baselines.

## The fifteen steps

### Knapsack

| # | Script | Verified result |
|---|---|---|
| 1 | `knapsack_01_the_instance.py` | The 12-item instance has 4,096 chromosomes; exact optimum is value 50 at weight 20 |
| 2 | `knapsack_02_random_population.py` | 233 of 1,000 random chromosomes are feasible; best random value is 47 |
| 3 | `knapsack_03_repair.py` | Repair makes 1,000/1,000 feasible and changes 767 chromosomes |
| 4 | `knapsack_04_selection_and_crossover.py` | One generation raises mean value from 39.00 to 40.73; all stored children remain feasible |
| 5 | `knapsack_05_the_full_search.py` | The GA reaches the exact value 50 at weight 20 with zero gap in 4,060 evaluations |

### Scheduling

| # | Script | Verified result |
|---|---|---|
| 1 | `schedule_01_the_requirements.py` | 168 bits encode 35 required assignments under capacity 40 |
| 2 | `schedule_02_random_rosters.py` | 0 of 500 random rosters are feasible; the best still has 14 violations |
| 3 | `schedule_03_repair.py` | Repair makes 500/500 rosters feasible; repaired costs range from 58 upward |
| 4 | `schedule_04_operators.py` | All 100 children are feasible; mean cost falls from 77.8 to 61.2 |
| 5 | `schedule_05_the_full_search.py` | GA cost 41 beats the best of 5,100 repaired-random rosters, cost 52, at equal evaluation count |

### Radar placement

| # | Script | Verified result |
|---|---|---|
| 1 | `radar_01_the_landscape.py` | Nine sites define 512 plans and jointly cover all 36 targets |
| 2 | `radar_02_the_chromosome.py` | 21 of 500 random plans cover every target |
| 3 | `radar_03_the_penalty.py` | The audited penalty ranks every feasible plan ahead of every infeasible plan |
| 4 | `radar_04_operators.py` | One generation lowers mean objective from 86.0 to 42.6 |
| 5 | `radar_05_the_full_search.py` | GA and exhaustive enumeration both find five radars with full coverage |

## Running it

From `MA2015 lessons/`:

```bash
uv run en/Lesson_09_Binary_Combinatorial/src/knapsack_01_the_instance.py
```

Every numbered script is self-contained, fixes its seed and saves a figure
under `figures/`. Scripts after the first of each prefix carry matching recipes
and numbered code bands.

## Status

| Piece | State |
|---|---|
| Code | Done. Fifteen numbered scripts run clean and pass the recipe/band checker. |
| Figures | Done. One generated figure per script. |

## Guided notebooks

Student entry point: [`notebooks/README.md`](notebooks/README.md). This lesson has 3 independently runnable sequences: [`lesson_09_knapsack.ipynb`](notebooks/lesson_09_knapsack.ipynb), [`lesson_09_scheduling.ipynb`](notebooks/lesson_09_scheduling.ipynb), [`lesson_09_radar.ipynb`](notebooks/lesson_09_radar.ipynb). The retained outputs were validated on date omitted against the frozen scripts in three fresh-kernel path modes; see the notebook README for environment details and the explicit hosted-Colab gap.

