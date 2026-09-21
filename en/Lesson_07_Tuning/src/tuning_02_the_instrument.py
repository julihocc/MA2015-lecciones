"""
Lesson 07 - Step 2: The instrument
===================================
NEW IN THIS STEP: standard_error(), measure(), and the knobs turned into
arguments.

Step 1 ran the book's method and watched it fail: one run per setting, a
ranking decided in the fourth decimal, win counts that flip with the seed.
Lesson 06 built the cure - a success rate over hundreds of runs, judged from
outside by a dense-grid reference and a stated tolerance. This step aims that
instrument at the first knob, crossover probability, on the course's
baseline: population 10, mutation probability 0.1, Lesson 01's sine
landscape, ten generations.

The measured answer on these 500 seeds is clear: the success rate climbs monotonically
from 55.6% at crossover 0.0 to 83.0% at crossover 1.0. The paired outcomes
measure that gap without treating the settings as independent. But the table is missing one column, and that column
changes how it reads: what each setting spent. Step 3 adds it.

CHANGES FROM tuning_01_single_runs.py
Introduce them in this order:
    1. the knobs become arguments  run() takes the probabilities per call
    2. standard_error()            the one-SE estimate used with each measured rate
    3. measure()                   the sweep instrument: N runs per setting
    4. the course's baseline       POPULATION_SIZE, MUTATION_PROBABILITY, RUNS

REMOVED FROM tuning_01_single_runs.py: the per-generation history, the
win-count table and the small-multiple figure (step 1's point is made), and
the book's protocol constants (population 16, mutation 0.2, seed 63), which
were the specimen, not the study.

Run it:  python tuning_02_the_instrument.py

measure() sweeps one knob over RUNS = 500 paired seeds and retains each
per-seed outcome. The crossover rate climbs from 55.6% to 83.0%; the gap uses
the standard error of the paired differences. The table is still missing the
cost column.
"""
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

# The algorithm, exactly as Lessons 01-06 left it.
MUTATION_MU, MUTATION_SIGMA = 0.0, 1.0
TOURNAMENT_SIZE, BLEND_ALPHA = 3, 1.0
POPULATION_SIZE = 10   # --- CHANGED --- from the book's 16: the course baseline
MUTATION_PROBABILITY = 0.1
RUNS = 500
FIGURES = Path(__file__).resolve().parent.parent / "figures"


@dataclass(frozen=True)
class Problem:
    """A landscape, its box, and the generation budget it is run with.

    Args:
        name: label for the printed tables.
        fitness: function of a gene list.
        low, high: the box.
        genes: chromosome length.
        generations: the run's budget.

    Example:
        TWO_D is Gridin's landscape on [-100, 100]^2 with 25 generations;
        SINE is Lesson 01's landscape on [-10, 10] with 10.
    """
    name: str
    fitness: Callable[[list[float]], float]
    low: float
    high: float
    genes: int
    generations: int


def sine_landscape(genes: list[float]) -> float:
    """Lesson 01's landscape, unchanged: one gene, one smooth global peak.

    Args:
        genes: a one-element list holding x.

    Returns:
        sin(x) - 0.2 * |x|. Larger is better.

    Example:
        Success on this landscape means within 0.01 of the brute-forced
        optimum +0.705908, a target 1.43% of the box wide.
    """
    return float(np.sin(genes[0]) - 0.2 * abs(genes[0]))


SINE = Problem("sine 1-D", sine_landscape, -10.0, 10.0, 1, 10)


class Individual:
    """One candidate solution, carrying the problem that judges it.


    Example:
        Each construction is one fitness call; the budget lessons count
        them because a knob that raises success also raises spend.
    """

    def __init__(self, genes: list[float], problem: Problem) -> None:
        """Build an individual and score it immediately.

        Args:
            genes: the chromosome, clamped to the problem's box.
            problem: landscape, bounds, gene count.

        Example:
            Each construction increments Individual.evaluations in the
            budget lessons, because a new Individual is one fitness call.
        """
        self.problem = problem
        self.genes = [clamp(g, problem) for g in genes]
        self.fitness = float(problem.fitness(self.genes))

    def __repr__(self) -> str:
        """Compact snapshot used in every printed table of this lesson.

        Returns:
            A one-line snapshot such as 'x=+1.372 f=+0.706' or
            '(+1.372) f=+0.7060', used in every printed table.
        """
        coordinates = ", ".join(f"{g:+.3f}" for g in self.genes)
        return f"({coordinates}) f={self.fitness:+.4f}"


def clamp(gene: float, problem: Problem) -> float:
    """Crossover and mutation propose anything; the box has edges.

    Args:
        gene: raw gene (float).
        problem: carries the box [low, high].

    Returns:
        gene projected onto [problem.low, problem.high].

    Example:
        Needed because blend alpha = 1.0 and N(0, 1) mutation both
        leave the box of whichever landscape is injected.
    """
    return max(problem.low, min(problem.high, gene))


def select_tournament(population: list[Individual]) -> list[Individual]:
    """Lesson 03's winner: the pressure is one integer, set on purpose.

    Args:
        population: current Individuals. Tournament size is the module
            constant TOURNAMENT_SIZE = 3.

    Returns:
        A list of the same length, holding references into population.

    Example:
        k = 3 is Lesson 03's measured setting: 2.74 expected copies of
        the best, not a wheel and not a floor.
    """
    return [max([random.choice(population) for _ in range(TOURNAMENT_SIZE)],
                key=lambda i: i.fitness) for _ in range(len(population))]


def crossover(parent1: Individual, parent2: Individual
              ) -> tuple[Individual, Individual]:
    """Blend crossover, one blending factor per gene.

    Args:
        parent1, parent2: parent Individuals.

    Returns:
        Two children, one blending factor per gene, then clamped.

    Example:
        BLEND_ALPHA = 1.0, the course default since Lesson 01 step 4.
    """
    genes1, genes2 = [], []
    for a, b in zip(parent1.genes, parent2.genes):
        shift = (1 + 2 * BLEND_ALPHA) * random.random() - BLEND_ALPHA
        genes1.append((1 - shift) * a + shift * b)
        genes2.append(shift * a + (1 - shift) * b)
    return (Individual(genes1, parent1.problem),
            Individual(genes2, parent1.problem))


def mutate(individual: Individual, sigma: float = MUTATION_SIGMA) -> Individual:
    """One coin decides whether the individual mutates; then every gene moves.

    Args:
        individual: the parent Individual.
        sigma: Gaussian step, default MUTATION_SIGMA = 1.0.

    Returns:
        A new Individual; every gene moves if the individual-level coin
        in the caller came up heads.

    Example:
        Per-individual probability 0.1 is Lesson 05's convention, reused
        so later counts of evaluations stay comparable.
    """
    return Individual([g + random.gauss(MUTATION_MU, sigma)
                       for g in individual.genes], individual.problem)


# --- NEW (1) the knobs become arguments ---------------------------------------
def run(problem: Problem, seed: int, population_size: int,
        crossover_probability: float, mutation_probability: float
        ) -> Individual:
    """One complete run, returning its answer: the best individual ever held.

    The probabilities arrive as arguments because this lesson turns them -
    a knob that lives in a global constant cannot be swept. Run i uses seed
    i, so every setting faces the same dice and the comparison is paired.

    Args:
        problem: the sine landscape.
        seed: run i uses seed i, so every setting faces the same dice.
        population_size, crossover_probability, mutation_probability:
            the three knobs. One is swept, the others sit in `fixed`.

    Returns:
        The list of generations.

    Example:
        Crossover 0.0 spends 20 evals/run, 1.0 spends 120; mutation
        0.2 is where the margin over blind search peaks at +5.9%.
    """
    random.seed(seed)
    population = [Individual([random.uniform(problem.low, problem.high)
                              for _ in range(problem.genes)], problem)
                  for _ in range(population_size)]
    ever = max(population, key=lambda i: i.fitness)
    for _ in range(problem.generations):
        selected = select_tournament(population)
        crossed: list[Individual] = []
        for parent1, parent2 in zip(selected[::2], selected[1::2]):
            if random.random() < crossover_probability:
                crossed.extend(crossover(parent1, parent2))
            else:
                crossed.extend([parent1, parent2])
        population = [mutate(ind) if random.random() < mutation_probability
                      else ind for ind in crossed]
        best = max(population, key=lambda i: i.fitness)
        if best.fitness > ever.fitness:
            ever = best
    return ever
# ------------------------------------------------------------------------------


def brute_force_optimum(problem: Problem, points_per_axis: int) -> float:
    """The best value on a regular grid: the truth, from outside any run.

    Args:
        problem: the landscape.
        points_per_axis: grid resolution. 20001 on 1-D; much coarser
            on 2-D because the cost is exponential in gene count.

    Returns:
        The best value the grid can see.

    Example:
        Sine 1-D: +0.705908. Book 2-D: +0.500000, occupying
        0.003197% of that grid.
    """
    axis = np.linspace(problem.low, problem.high, points_per_axis)
    return float(max(problem.fitness([float(x)]) for x in axis))


TOLERANCE = 0.01


def verdict(individual: Individual, optimum: float) -> bool:
    """Did this run arrive? A yes/no, decided from outside the run.

    Args:
        individual: a run's champion (best-ever).
        optimum: brute_force_optimum of the same landscape.

    Returns:
        True iff fitness >= optimum - TOLERANCE (0.01).

    Example:
        0 of 8 book-2-D runs succeed (target has no volume). 5 of 8
        sine runs succeed: 62.5% ± 17.1%. On 1000 sine runs: 79.4% ± 1.3%.
    """
    return individual.fitness >= optimum - TOLERANCE


# --- NEW (2) standard_error() -------------------------------------------------
def standard_error(rate: float, n: int) -> float:
    """The wobble of a measured rate: sqrt(p(1-p)/n).

    Lesson 06's plug-in standard error, promoted to permanent equipment: no tuning
    decision in this lesson is quoted without it.

    Args:
        rate: an observed success fraction.
        n: the number of runs.

    Returns:
        sqrt(p(1-p)/n).

    Example:
        8 runs at 62.5% carry ±17.1 points; 1000 runs at 79.4% carry
        ±1.3. The 8-run estimate missed by 16.9 points, 1.0 of its
        own standard deviations.
    """
    return (rate * (1 - rate) / n) ** 0.5
# ------------------------------------------------------------------------------


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Approximate 95% binomial interval, including boundary observations."""
    rate = successes / n
    denominator = 1 + z ** 2 / n
    centre = (rate + z ** 2 / (2 * n)) / denominator
    radius = z / denominator * (
        rate * (1 - rate) / n + z ** 2 / (4 * n ** 2)
    ) ** 0.5
    return centre - radius, centre + radius


# --- NEW (3) measure() --------------------------------------------------------
def measure(problem: Problem, knob: str, values: tuple, fixed: dict,
            runs: int, optimum: float
            ) -> list[tuple[float, float, float, list[int]]]:
    """The sweep instrument: for each value of `knob`, with the other knobs
    held at `fixed`, run the algorithm `runs` times and return
    (value, success rate, standard error, per-seed outcomes) per setting.

    Same seeds for every setting, so the dice are paired and the differences
    are the knob's, not luck's.

    Args:
        problem: sine 1-D.
        knob: name of the argument of run() being swept.
        values: the settings.
        fixed: the other knobs, held still.
        runs: 500 in the one-knob steps, 100 in the grid.
        optimum: dense-grid reference.

    Returns:
        Rate summaries plus the 0/1 outcomes needed for paired differences.

    Example:
        Crossover climbs 55.6% to 83.0% over 500 runs, 27.4 points
        apart; the paired outcomes quantify the uncertainty of that gap.
    """
    rows = []
    for value in values:
        settings = {**fixed, knob: value}
        outcomes = [int(verdict(run(problem, seed, **settings), optimum))
                    for seed in range(runs)]
        successes = sum(outcomes)
        rate = successes / runs
        rows.append((value, rate, standard_error(rate, runs), outcomes))
    return rows
# ------------------------------------------------------------------------------


# ===== The first knob, measured ===============================================
OPTIMUM = brute_force_optimum(SINE, 20001)
FIXED = {"population_size": POPULATION_SIZE,
         "mutation_probability": MUTATION_PROBABILITY}
VALUES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

print(f"The target: grid reference {OPTIMUM:+.6f}, success means within {TOLERANCE} "
      f"of it.")
print(f"The instrument: {RUNS} runs per setting, population "
      f"{FIXED['population_size']}, mutation "
      f"{FIXED['mutation_probability']} fixed, the same {RUNS} seeds for "
      f"every setting.\n")
print("  crossover | success rate")
print("  ----------+--------------")
rows = measure(SINE, "crossover_probability", VALUES, FIXED, RUNS, OPTIMUM)
for value, rate, error, outcomes in rows:
    low, high = wilson_interval(sum(outcomes), RUNS)
    print(f"  {value:>9} | {rate:6.1%} +/- {error:4.1%} one SE; "
          f"95% Wilson [{low:.1%}, {high:.1%}]")

first, last = rows[0], rows[-1]
gap = last[1] - first[1]
differences = [after - before for before, after in zip(first[3], last[3])]
paired_error = statistics.stdev(differences) / len(differences) ** 0.5
print(f"\nThe endpoint contrast that step 1 could not hold still is clear in this sample: the rate "
      f"climbs monotonically")
print(f"from {first[1]:.1%} at crossover {first[0]} to {last[1]:.1%} at "
      f"crossover {last[0]}. The ends of the")
print(f"dial are {gap:.1%} apart; the paired standard error is "
      f"{paired_error:.1%}. Neighbouring gaps are smaller, and this sweep is")
print("exploratory rather than a multiple-comparison test; it is not the chasm the")
print("single runs pretended. Crossover works, and now it is measured rather")
print("than witnessed on one seed.")
print("\nOne column is missing from this table, and it changes how the table")
print("reads: what each setting spent. Step 3 adds it.")

# The picture: observed rates with 95% Wilson intervals.
fig, ax = plt.subplots(figsize=(7.5, 4.2))
intervals = [wilson_interval(sum(row[3]), RUNS) for row in rows]
lower_errors = [row[1] - interval[0] for row, interval in zip(rows, intervals)]
upper_errors = [interval[1] - row[1] for row, interval in zip(rows, intervals)]
ax.errorbar([r[0] for r in rows], [r[1] for r in rows],
            yerr=[lower_errors, upper_errors], fmt="o-", color="tab:blue",
            capsize=4, linewidth=1.6)
ax.set_title(f"Crossover probability, measured: {RUNS} runs per setting "
             f"(population {FIXED['population_size']}, "
             f"mutation {FIXED['mutation_probability']})")
ax.set_xlabel("crossover probability")
ax.set_ylabel("observed success rate (95% Wilson interval)")
ax.grid(True, linestyle=":", alpha=0.5)
FIGURES.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURES / "tuning_02_the_instrument.png", dpi=150,
            bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved to {FIGURES}/tuning_02_the_instrument.png")

