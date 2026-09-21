"""
Lesson 07 - Step 4: Mutation, and where the bell curve hides
=============================================================
NEW IN THIS STEP: the second knob. The instrument does not change - that is
the point of having built it.

Every textbook draws the mutation curve as a bell: too little mutation and
the search stagnates, too much and it degenerates into random walk, with a
sweet spot in between. Gridin's Chapter 7 shows single runs at 0.0, 0.3 and
1.0 and gestures at the same shape. This step measures the knob properly -
seven settings, 500 runs each, crossover fixed at 0.8 - and the bell curve
is not there. The raw success rate climbs monotonically all the way to
mutation 1.0.

Then the margin column tells the truth. Priced against blind search at
equal budget, the curve DOES bend: negative at mutation 0.0 (the GA without
mutation is worse than rolling dice), a broad peak around 0.2, negative
again at 1.0. The bell curve the textbooks promise exists - but it lives in
the margin over blind search, where the budget is priced, not in the raw
rate, where every extra evaluation looks like progress.

CHANGES FROM tuning_03_the_budget_knob.py
Introduce them in this order:
    1. the second knob   PM_VALUES swept, crossover fixed at 0.8

Run it:  python tuning_04_mutation.py

The same instrument, aimed at mutation probability with crossover fixed at
0.8. The raw success rate is monotone to mutation 1.0 (61.8% to 89.8%) — no
bell curve. The margin over blind search bends: -10.9% at 0.0 (worse than
dice), peak +5.9% near 0.2, -3.7% at 1.0. The bell curve lives in the margin.
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
POPULATION_SIZE = 10
CROSSOVER_PROBABILITY = 0.8
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

    `evaluations` counts every fitness evaluation: the cost of a run, and of
    a knob setting.


    Example:
        Each construction is one fitness call; the budget lessons count
        them because a knob that raises success also raises spend.
    """

    evaluations = 0

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
        Individual.evaluations += 1

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


def run(problem: Problem, seed: int, population_size: int,
        crossover_probability: float, mutation_probability: float
        ) -> Individual:
    """One complete run, returning its answer: the best individual ever held.

    Run i uses seed i, so every setting faces the same dice.

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


def within_tolerance_share(problem: Problem, optimum: float,
                           points_per_axis: int) -> float:
    """What fraction of the box scores within TOLERANCE of the optimum -
    also the hit probability of one blind draw.

    Args:
        problem, optimum, points_per_axis: as brute_force_optimum.

    Returns:
        Fraction of the grid that scores within TOLERANCE of the
        optimum — the hit probability of one blind draw.

    Example:
        Book 2-D: 0.003197%. Sine 1-D: 1.4299% of the box (0.286 wide).
    """
    axis = np.linspace(problem.low, problem.high, points_per_axis)
    values = np.array([problem.fitness([float(x)]) for x in axis])
    return float(np.mean(values >= optimum - TOLERANCE))


def standard_error(rate: float, n: int) -> float:
    """The wobble of a measured rate: sqrt(p(1-p)/n).

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


def measure(problem: Problem, knob: str, values: tuple, fixed: dict,
            runs: int, optimum: float) -> list[tuple]:
    """The sweep instrument: per setting, (value, success rate, standard
    error, mean evaluations per run). Same seeds for every setting.

    Args:
        problem: sine 1-D.
        knob: name of the argument of run() being swept.
        values: the settings.
        fixed: the other knobs, held still.
        runs: 500 in the one-knob steps, 100 in the grid.
        optimum: brute-forced.

    Returns:
        (value, success rate, standard error) per setting. Same
        seeds for every setting, so the dice are paired.

    Example:
        Crossover climbs 55.6% to 83.0% over 500 runs, 27.4 points
        apart (~10 SE of the difference).
    """
    rows = []
    for value in values:
        settings = {**fixed, knob: value}
        successes, costs = 0, []
        for seed in range(runs):
            Individual.evaluations = 0
            champion = run(problem, seed, **settings)
            costs.append(Individual.evaluations)
            successes += verdict(champion, optimum)
        rate = successes / runs
        rows.append((value, rate, standard_error(rate, runs),
                     statistics.fmean(costs)))
    return rows


def blind_success(share: float, evaluations: float) -> float:
    """The success rate of spending `evaluations` blind uniform draws:
    1 - (1 - share) ** E. The benchmark a setting has to beat to justify
    its budget.

    Args:
        share: within_tolerance_share of the landscape.
        evaluations: what the GA spent.

    Returns:
        1 - (1 - share)^E, the chance at least one of E uniform
        draws hits the target.

    Example:
        At population 10 the GA's 80.0% vs blind 76.3% is a +3.7
        point margin; at population 30, +0.7 points. In the tuning
        grid only 20 of 36 cells beat their own budget.
    """
    return 1.0 - (1.0 - share) ** evaluations


OPTIMUM = brute_force_optimum(SINE, 20001)
SHARE = within_tolerance_share(SINE, OPTIMUM, 20001)

# --- CHANGED --- the second knob: mutation swept, crossover fixed at 0.8
FIXED = {"population_size": POPULATION_SIZE,
         "crossover_probability": CROSSOVER_PROBABILITY}
KNOB = "mutation_probability"
VALUES = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)

print(f"The target: optimum {OPTIMUM:+.6f}; a blind draw hits it "
      f"{SHARE:.2%} of the time.")
print(f"The instrument: {RUNS} runs per setting, population "
      f"{FIXED['population_size']}, crossover "
      f"{FIXED['crossover_probability']} fixed.\n")
print("  mutation | success rate  | evals/run | blind at same | margin")
print("           |               |           |    budget     |")
print("  ---------+---------------+-----------+---------------+--------")
rows = measure(SINE, KNOB, VALUES, FIXED, RUNS, OPTIMUM)
for value, rate, error, cost in rows:
    blind = blind_success(SHARE, cost)
    print(f"  {value:>8} | {rate:6.1%} +/-{error:4.1%} | {cost:9.1f} | "
          f"{blind:12.1%} | {rate - blind:+5.1%}")

rates = [r[1] for r in rows]
margins = [r[1] - blind_success(SHARE, r[3]) for r in rows]
best_margin = max(range(len(rows)), key=lambda i: margins[i])
print(f"\nThe raw rate climbs monotonically, {rates[0]:.1%} to {rates[-1]:.1%}"
      f" - no bell curve. Read")
print("only that column and mutation 1.0 wins, and the textbook shape is")
print("nowhere. But the margin column bends: negative at mutation 0.0")
print(f"({margins[0]:+.1%} - without mutation the GA is WORSE than blind "
      f"draws at equal budget),")
print(f"a broad peak around mutation {rows[best_margin][0]} "
      f"({margins[best_margin]:+.1%}), negative again at "
      f"1.0 ({margins[-1]:+.1%}).")
print("\nThe bell curve the textbooks promise exists. It lives in the margin")
print("over blind search, where the budget is priced - not in the raw rate,")
print("where every extra evaluation looks like progress.")
print("\nAnd mutation 0.0 is the warning Lesson 05 made, now priced: without")
print("mutation the population can only remix its initial genes, and once it")
print("converges it re-spends its budget re-evaluating near-duplicates.")
print("\nThe third knob is the purest budget knob of all. Step 5.")

# The picture: the rate climbs and never bends; the margin does.
fig, (ax_rate, ax_margin) = plt.subplots(1, 2, figsize=(11.5, 4.0))
values = [r[0] for r in rows]
ax_rate.errorbar(values, rates, yerr=[r[2] for r in rows], fmt="o-",
                 color="tab:blue", capsize=4, linewidth=1.6)
ax_rate.set_title("The raw rate: monotone - no bell curve here", fontsize=10)
ax_rate.set_xlabel("mutation probability")
ax_rate.set_ylabel("success rate")
ax_rate.grid(True, linestyle=":", alpha=0.5)

ax_margin.axhline(0.0, color="black", linewidth=0.8)
ax_margin.plot(values, margins, "o-", color="tab:red", linewidth=1.6)
ax_margin.annotate(f"peak near {rows[best_margin][0]}",
                   xy=(rows[best_margin][0], margins[best_margin]),
                   xytext=(14, -4), textcoords="offset points", ha="left",
                   fontsize=8)
ax_margin.set_title("The margin over blind search: the bell curve lives here",
                    fontsize=10)
ax_margin.set_xlabel("mutation probability")
ax_margin.set_ylabel("success rate minus blind, same budget")
ax_margin.grid(True, linestyle=":", alpha=0.5)

fig.suptitle("Mutation: measured in the wrong currency, it looks free")
FIGURES.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURES / "tuning_04_mutation.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved to {FIGURES}/tuning_04_mutation.png")

