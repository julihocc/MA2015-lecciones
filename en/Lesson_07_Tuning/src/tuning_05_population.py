"""
Lesson 07 - Step 5: Population, the purest budget knob
=======================================================
NEW IN THIS STEP: the third knob - the book's own four population sizes,
measured on the course's terms.

Crossover and mutation at least pretend to be about search strategy.
Population size pretends nothing: a bigger population is more evaluations,
full stop. It is the budget knob with no disguise, and Lesson 06 step 4
already caught it once - effectiveness climbs, efficiency falls. This step
re-measures it inside the tuning study, with the book's own settings
(6, 10, 20, 50, the book's mutation 0.2), to close the pattern before the
final map.

The pattern holds, and it sharpens. Population 50 succeeds in every one of
500 runs - a perfect score, bought at 550 evaluations a run, where the
yield has collapsed to a fifth of what population 6 delivers per 1000
evaluations. And priced against blind search, the margin peaks at
population 10 and vanishes at 50: at that budget, the GA and rolling dice
are the same machine. "Which population is best?" is a question with no
answer until a currency is named - and naming currencies is the lesson.

CHANGES FROM tuning_04_mutation.py
Introduce them in this order:
    1. the third knob   POPULATION_SIZES swept, crossover 0.8 and mutation 0.2 fixed

Run it:  python tuning_05_population.py

The third knob, the book's sizes 6, 10, 20, 50. Population 50 records 500/500
successes (95% Wilson interval 99.2% to 100%) at 550 evals/run; yield collapses 9.25 to 1.82 successes
per 1000 evals. The margin peaks at population 10 (+5.9%) and vanishes at 50.
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
CROSSOVER_PROBABILITY = 0.8
MUTATION_PROBABILITY = 0.2
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


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Approximate 95% interval that retains uncertainty at 0% and 100%."""
    rate = successes / n
    denominator = 1 + z ** 2 / n
    centre = (rate + z ** 2 / (2 * n)) / denominator
    radius = z / denominator * (
        rate * (1 - rate) / n + z ** 2 / (4 * n ** 2)
    ) ** 0.5
    return centre - radius, centre + radius


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

# --- CHANGED --- the third knob: the book's population sizes, crossover 0.8
# and the book's mutation 0.2 fixed
FIXED = {"crossover_probability": CROSSOVER_PROBABILITY,
         "mutation_probability": MUTATION_PROBABILITY}
KNOB = "population_size"
VALUES = (6, 10, 20, 50)

print(f"The target: optimum {OPTIMUM:+.6f}; a blind draw hits it "
      f"{SHARE:.2%} of the time.")
print(f"The instrument: {RUNS} runs per setting, crossover "
      f"{FIXED['crossover_probability']}, mutation "
      f"{FIXED['mutation_probability']} fixed - the book's own four "
      f"sizes.\n")
print("  population | success rate  | evals/run | blind at same | margin |"
      " successes per")
print("             |               |           |    budget     |        |"
      " 1000 evals")
print("  -----------+---------------+-----------+---------------+--------+"
      "--------------")
rows = measure(SINE, KNOB, VALUES, FIXED, RUNS, OPTIMUM)
for value, rate, error, cost in rows:
    blind = blind_success(SHARE, cost)
    print(f"  {value:>10} | {rate:6.1%} +/-{error:4.1%} | {cost:9.1f} | "
          f"{blind:12.1%} | {rate - blind:+5.1%} | {1000 * rate / cost:8.2f}")

first, last = rows[0], rows[-1]
last_interval = wilson_interval(round(last[1] * RUNS), RUNS)
margins = [r[1] - blind_success(SHARE, r[3]) for r in rows]
best_margin = max(range(len(rows)), key=lambda i: margins[i])
print(f"\nPopulation {last[0]} records {RUNS}/{RUNS} successes; its 95% Wilson "
      f"interval is [{last_interval[0]:.1%}, {last_interval[1]:.1%}]. That observed score is bought")
print(f"at {last[3]:.0f} evaluations a run. Population {first[0]} succeeds "
      f"{first[1]:.1%} of the time at")
print(f"{first[3]:.0f}. The yield falls from {1000 * first[1] / first[3]:.2f} "
      f"to {1000 * last[1] / last[3]:.2f} successes per")
print("1000 evaluations: a five-fold collapse for a climb from "
      f"{first[1]:.1%} to {last[1]:.1%}.")
print(f"\nAnd the margin over blind search peaks at population "
      f"{rows[best_margin][0]} ({margins[best_margin]:+.1%}) and vanishes at "
      f"{last[0]} ({margins[-1]:+.1%}):")
print("at that budget the GA and rolling dice are the same machine.")
print("\nSo: which population is best? There is no answer until a currency is")
print("named. Effectiveness says 50. Efficiency says 6. The margin says 10.")
print("And the knobs do not act one at a time - step 6 maps them together.")

# The picture: the rate climbs into the blind curve; the yield collapses.
fig, (ax_rate, ax_yield) = plt.subplots(1, 2, figsize=(11.5, 4.0))
costs = [r[3] for r in rows]
rates = [r[1] for r in rows]
budget_line = np.linspace(min(costs) * 0.9, max(costs) * 1.02, 300)
ax_rate.plot(budget_line, [blind_success(SHARE, e) for e in budget_line],
             color="tab:gray", linestyle="--", linewidth=1.4,
             label="blind draws, same budget")
ax_rate.errorbar(costs, rates, yerr=[r[2] for r in rows], fmt="o-",
                 color="tab:blue", capsize=4, linewidth=1.6,
                 label="the GA, population 6-50")
for value, rate, _, cost in rows:
    ax_rate.annotate(f"pop {value}", xy=(cost, rate), xytext=(0, 8),
                     textcoords="offset points", ha="center", fontsize=8)
ax_rate.set_title("At population 50 the GA merges into the blind curve",
                  fontsize=10)
ax_rate.set_xlabel("evaluations per run")
ax_rate.set_ylabel("success rate")
ax_rate.grid(True, linestyle=":", alpha=0.5)
ax_rate.legend(fontsize=8, loc="center right")

ax_yield.plot([r[0] for r in rows], [1000 * r[1] / r[3] for r in rows], "o-",
              color="tab:red", linewidth=1.6)
ax_yield.set_title("The yield collapses as the population grows", fontsize=10)
ax_yield.set_xlabel("population size")
ax_yield.set_ylabel("successes per 1000 evaluations")
ax_yield.grid(True, linestyle=":", alpha=0.5)

fig.suptitle("Population: the budget knob with no disguise")
FIGURES.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURES / "tuning_05_population.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved to {FIGURES}/tuning_05_population.png")

