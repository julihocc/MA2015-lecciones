"""
Lesson 07 - Step 3: The budget knob
====================================
NEW IN THIS STEP: the evaluation counter, blind_success(), and the margin
column.

Step 2 measured the crossover knob and found a clean monotone climb: 55.6%
at 0.0 to 83.0% at 1.0. It closed with the missing column. Crossover does
not come free - every crossed pair creates two new individuals, and every
new individual is a fitness evaluation, the only currency a GA spends. So
the dial that "improves" the algorithm also inflates its budget, and the
question becomes: how much of the climb is better search, and how much is
just more spending?

The instrument for that question is Lesson 06's: count the evaluations, then
price each setting against blind uniform draws given the same budget. The
answer, on this landscape, is uncomfortable. The knob is a budget knob: the
margin over blind search shrinks monotonically as crossover is turned up,
and the hundred evaluations that separate the two ends of the dial buy the
GA less than a hundred blind draws would have bought.

CHANGES FROM tuning_02_the_instrument.py
Introduce them in this order:
    1. the evaluation counter  Individual counts; measure() collects per setting
    2. blind_success()         the equal-budget benchmark: 1 - (1 - q) ** E
    3. the margin column       rate minus blind at the same budget

Run it:  python tuning_03_the_budget_knob.py

The evaluation counter rides on Individual, as in Lesson 06. Crossover 0.0
spends 20 evals/run, 1.0 spends 120; the margin over blind search falls +30.6
to +0.8 points. The 100 extra evaluations buy the GA 27.4 points where 100
blind draws buy 57.2 — the knob is a budget knob.
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


# --- NEW (1) the evaluation counter -------------------------------------------
class Individual:
    """One candidate solution, carrying the problem that judges it.

    `evaluations` counts every fitness evaluation the algorithm performs:
    it is incremented each time an Individual is created, and reset to zero
    before each run. Fitness is the only thing a GA ever pays for, so this
    counter IS the cost of a run - and of a knob setting.


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
# ------------------------------------------------------------------------------


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
    """The sweep instrument, now with the cost column: per setting it returns
    (value, success rate, standard error, mean evaluations per run).

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
            Individual.evaluations = 0          # NEW (1): collected per run
            champion = run(problem, seed, **settings)
            costs.append(Individual.evaluations)
            successes += verdict(champion, optimum)
        rate = successes / runs
        rows.append((value, rate, standard_error(rate, runs),
                     statistics.fmean(costs)))
    return rows


# --- NEW (2) blind_success() --------------------------------------------------
def blind_success(share: float, evaluations: float) -> float:
    """The success rate of spending `evaluations` blind uniform draws.

    Each draw hits the target with probability `share`, independently, so the
    probability that at least one of E draws hits is 1 - (1 - share) ** E.
    This is the benchmark a knob setting has to beat to justify its budget:
    the same money, no operators, no memory, no population.

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
# ------------------------------------------------------------------------------


OPTIMUM = brute_force_optimum(SINE, 20001)
SHARE = within_tolerance_share(SINE, OPTIMUM, 20001)
FIXED = {"population_size": POPULATION_SIZE,
         "mutation_probability": MUTATION_PROBABILITY}
VALUES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

print(f"The target: optimum {OPTIMUM:+.6f}; a blind draw hits it "
      f"{SHARE:.2%} of the time.")
print(f"The instrument: {RUNS} runs per setting, population "
      f"{FIXED['population_size']}, mutation "
      f"{FIXED['mutation_probability']} fixed - and now every run is "
      f"priced.\n")

# --- NEW (3) the margin column ------------------------------------------------
print("  crossover | success rate  | evals/run | blind at same | margin")
print("            |               |           |    budget     |")
print("  ----------+---------------+-----------+---------------+--------")
rows = measure(SINE, "crossover_probability", VALUES, FIXED, RUNS, OPTIMUM)
for value, rate, error, cost in rows:
    blind = blind_success(SHARE, cost)
    print(f"  {value:>9} | {rate:6.1%} +/-{error:4.1%} | {cost:9.1f} | "
          f"{blind:12.1%} | {rate - blind:+5.1%}")
# ------------------------------------------------------------------------------

first, last = rows[0], rows[-1]
extra_evals = last[3] - first[3]
ga_gain = last[1] - first[1]
blind_gain = blind_success(SHARE, last[3]) - blind_success(SHARE, first[3])
print(f"\nEvery turn of the knob buys evaluations: crossover {first[0]} "
      f"spends {first[3]:.0f} a run,")
print(f"crossover {last[0]} spends {last[3]:.0f}. The margin over blind "
      f"search at equal budget")
print(f"shrinks monotonically, from {first[1] - blind_success(SHARE, first[3]):+.1%} "
      f"to {last[1] - blind_success(SHARE, last[3]):+.1%}.")
print(f"\nAnd the {extra_evals:.0f} evaluations between the two ends buy the "
      f"GA {ga_gain:.1%} - while the")
print(f"same {extra_evals:.0f} spent blind would have bought {blind_gain:.1%}. "
      f"On this landscape,")
print("the crossover dial is not a quality dial. It is a budget dial, and a")
print("wasteful one.")
print("\nBefore writing crossover off, remember the currency: this target is")
print(f"wide ({SHARE:.2%} of the box), so blind money goes far here. Where it")
print("goes nowhere - the combinatorial problems of Lessons 09 to 11 - the")
print("operators are the only game in town. And the second knob tells a")
print("different story. Step 4.")

# The picture: the climb against cost, with the blind curve underneath; and
# the margin, falling.
fig, (ax_rate, ax_margin) = plt.subplots(1, 2, figsize=(11.5, 4.0))
costs = [r[3] for r in rows]
rates = [r[1] for r in rows]
budget_line = np.linspace(min(costs) * 0.9, max(costs) * 1.05, 200)
ax_rate.plot(budget_line, [blind_success(SHARE, e) for e in budget_line],
             color="tab:gray", linestyle="--", linewidth=1.4,
             label="blind draws, same budget")
ax_rate.errorbar(costs, rates, yerr=[r[2] for r in rows], fmt="o-",
                 color="tab:blue", capsize=4, linewidth=1.6,
                 label="the GA, crossover 0.0-1.0")
for value, rate, _, cost in rows:
    ax_rate.annotate(f"pc {value}", xy=(cost, rate), xytext=(0, 8),
                     textcoords="offset points", ha="center", fontsize=8)
ax_rate.set_title("The climb is mostly spending", fontsize=10)
ax_rate.set_xlabel("evaluations per run")
ax_rate.set_ylabel("success rate")
ax_rate.grid(True, linestyle=":", alpha=0.5)
ax_rate.legend(fontsize=8, loc="lower right")

margins = [r[1] - blind_success(SHARE, r[3]) for r in rows]
ax_margin.bar([str(r[0]) for r in rows], margins, color="tab:red", alpha=0.8)
ax_margin.set_title("The margin over blind shrinks as the knob opens",
                    fontsize=10)
ax_margin.set_xlabel("crossover probability")
ax_margin.set_ylabel("success rate minus blind, same budget")
ax_margin.grid(True, axis="y", linestyle=":", alpha=0.5)

fig.suptitle("The crossover knob is a budget knob")
FIGURES.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURES / "tuning_03_the_budget_knob.png", dpi=150,
            bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved to {FIGURES}/tuning_03_the_budget_knob.png")

