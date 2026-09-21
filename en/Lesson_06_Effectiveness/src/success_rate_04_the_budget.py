"""
Lesson 06 - Step 4: The budget
==============================
NEW IN THIS STEP: the evaluation counter, the population sweep, and
blind_success() - the benchmark nobody has run yet.

Step 3 measured effectiveness: a success rate of 79.4% at population 10, and
it closed with the question a rate cannot answer - what did those evaluations
cost, and were they well spent? This step counts them. Gridin's Chapter 6
does it with a class attribute on Individual that counts every fitness
evaluation, and with a sweep of population sizes that plots the final fitness
against the number of individuals evaluated.

Two measurements fall out, and they pull in opposite directions. Raising the
population from 10 to 30 raises the success rate from 80% to 99% - and raises
the cost of a run from 100 evaluations to 300. Effectiveness climbs,
efficiency falls, monotonically, both of them. There is no configuration
that is best at both; there is only a choice, and Lesson 07 exists to make it.

Then the honest verdict. Spend the same budget on blind uniform draws and
the GA's margin on this landscape turns out to be a few points, not a
chasm - because a target 0.286 wide is easy for everyone. That is a property
of the problem, not a defect of the algorithm: on the combinatorial problems
of Lessons 09 to 11 a blind draw essentially never hits, and the operators
are the only game in town.

CHANGES FROM success_rate_03_a_thousand_runs.py
Introduce them in this order:
    1. the evaluation counter  Individual counts every fitness evaluation
    2. the population sweep    POPULATION_SIZES, re-running the instrument per size
    3. blind_success()         the same budget spent on blind draws: 1 - (1 - q) ** E
    4. RUNS = 500              changed from 1000: five populations, 500 runs apiece

REMOVED FROM success_rate_03_a_thousand_runs.py: the peak-finding and the
histogram (the distribution is step 3's story; this one is about cost).

Run it:  python success_rate_04_the_budget.py

Individual.evaluations counts every fitness call, because that is what a run
actually spends. blind_success() is 1-(1-share)^E, the hit rate of E uniform
draws. Population 10 to 30: success 80.0% to 99.4%, cost 100 to 300
evaluations, yield falls 8.00 to 3.31 successes per 1000 evaluations. Blind
search at equal budget reaches 76.2% and 98.7% — the GA's margin is +3.8 and
+0.7 points.
"""
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

# The algorithm, exactly as Lessons 01-05 left it.
CROSSOVER_PROBABILITY = 0.8
MUTATION_PROBABILITY = 0.1
MUTATION_MU, MUTATION_SIGMA = 0.0, 1.0
TOURNAMENT_SIZE, BLEND_ALPHA = 3, 1.0
# Run i uses seed i, for the whole lesson.
RUNS = 500   # --- CHANGED --- from 1000: five populations, 500 runs apiece
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

    `evaluations` counts every fitness evaluation the algorithm performs,
    across the whole experiment: it is incremented each time an Individual is
    created, and reset to zero before each run. Fitness is the only thing a
    GA ever pays for, so this counter IS the cost of a run.


    Example:
        Eight seeds on book_2d spread 0.8946 in best-ever fitness; on sine,
        5 of 8 runs land within 0.01 of the optimum.
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
        Individual.evaluations += 1  # a new Individual is one fitness call; this IS the cost of a run

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


def select_tournament(population: list[Individual],
                      size: int) -> list[Individual]:
    """Lesson 03's winner: the pressure is one integer, set on purpose.

    Args:
        population: current Individuals.
        size: candidates per tournament. 3 is Lesson 01's default.

    Returns:
        A list of the same length, holding references into population.

    Example:
        On Lesson 01's SEED = 52 population, 4 of 10 individuals go
        extinct. In Lesson 03, k = 3 gives 2.74 copies of the best,
        k = 5 gives 4.13, k = 10 gives 6.52.
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


def evolve_one_generation(population: list[Individual],
                          size: int) -> list[Individual]:
    """SELECT -> CROSSOVER -> MUTATE -> replace, once.

    Args:
        population: current generation.
        size: population size, forwarded to tournament selection.

    Returns:
        The next generation.
    """
    selected = select_tournament(population, size)
    crossed: list[Individual] = []
    for parent1, parent2 in zip(selected[::2], selected[1::2]):
        if random.random() < CROSSOVER_PROBABILITY:
            crossed.extend(crossover(parent1, parent2))
        else:
            crossed.extend([parent1, parent2])
    return [mutate(ind) if random.random() < MUTATION_PROBABILITY else ind
            for ind in crossed]


def run(problem: Problem, seed: int, population_size: int
        ) -> list[list[Individual]]:
    """One complete run: initialise, then evolve for the problem's budget.

    Args:
        problem: the landscape under test.
        seed: run i uses seed i.
        population_size: the budget knob this step sweeps.

    Returns:
        The list of generations. Individual.evaluations counts the cost.

    Example:
        Population 10 spends about 100 evaluations and succeeds 80.0%;
        population 30 spends 300 and succeeds 99.4%.
    """
    random.seed(seed)
    population = [Individual([random.uniform(problem.low, problem.high)
                              for _ in range(problem.genes)], problem)
                  for _ in range(population_size)]
    generations = [population]
    for _ in range(problem.generations):
        population = evolve_one_generation(population, population_size)
        generations.append(population)
    return generations


def best_ever(generations: list[list[Individual]]) -> Individual:
    """The best individual the run ever held, which is the run's answer.

    Args:
        generations: every generation the run held.

    Returns:
        The Individual with the highest fitness ever seen, which may
        not be in the last generation (this flow has no elitism).

    Example:
        On 1 of 2 problems in Lesson 02 step 4 the last generation no
        longer holds the champion.
    """
    return max((ind for pop in generations for ind in pop),
               key=lambda i: i.fitness)


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
    """What fraction of a dense grid scores within TOLERANCE of the reference.

    This fraction approximates the uniform continuous hit probability; it is
    not an exact measure of the target set.

    Args:
        problem, optimum, points_per_axis: as brute_force_optimum.

    Returns:
        Fraction of the grid within TOLERANCE — an approximation to the
        hit probability of one blind draw.

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
    """Approximate 95% binomial interval that remains valid at 0% and 100%."""
    rate = successes / n
    denominator = 1 + z ** 2 / n
    centre = (rate + z ** 2 / (2 * n)) / denominator
    radius = z / denominator * (
        rate * (1 - rate) / n + z ** 2 / (4 * n ** 2)
    ) ** 0.5
    return centre - radius, centre + radius


# --- NEW (3) blind_success() --------------------------------------------------
def blind_success(share: float, evaluations: float) -> float:
    """The success rate of spending `evaluations` blind uniform draws.

    Each draw hits the target with probability `share`, independently, so the
    probability that at least one of E draws hits is 1 - (1 - share) ** E.
    This is the benchmark the GA has to beat to justify its machinery: the
    same budget, no operators, no memory, no population.

    Args:
        share: within_tolerance_share of the landscape.
        evaluations: what the GA spent.

    Returns:
        1 - (1 - share)^E, the chance at least one of E uniform
        draws hits the target.

    Example:
        At population 10 the GA's 80.0% vs blind 76.2% is a +3.8
        point margin; at population 30, +0.7 points. In the tuning
        grid only 20 of 36 cells beat their own budget.
    """
    return 1.0 - (1.0 - share) ** evaluations
# ------------------------------------------------------------------------------


optimum = brute_force_optimum(SINE, 20001)
share = within_tolerance_share(SINE, optimum, 20001)
print(f"The target: grid reference {optimum:+.6f}, success means within "
      f"{TOLERANCE} of it; the grid estimates a blind hit share of "
      f"{share:.2%}.\n")

# --- NEW (2) the population sweep ---------------------------------------------
# The book's sweep: the same seeds, the same budget of generations, and only
# the population size changing. Each size re-runs the whole Monte Carlo
# instrument, counting what every run cost.
POPULATION_SIZES = (10, 15, 20, 25, 30)

print(f"{RUNS} runs per population size, {SINE.generations} generations "
      f"each:\n")
print("  population | rate (one SE) | 95% Wilson interval | evaluations/run | successes per")
print("             |               |                     |   (mean, max)   | 1000 evaluations")
print("  -----------+---------------+---------------------+-----------------+----------------")
sweep = []
for size in POPULATION_SIZES:
    successes, costs = 0, []
    for seed in range(RUNS):
        Individual.evaluations = 0
        champion = best_ever(run(SINE, seed, size))
        costs.append(Individual.evaluations)
        successes += verdict(champion, optimum)
    rate = successes / RUNS
    mean_cost = statistics.fmean(costs)
    interval = wilson_interval(successes, RUNS)
    blind_average = statistics.fmean(blind_success(share, cost) for cost in costs)
    sweep.append((size, successes, rate, interval, mean_cost, max(costs), blind_average))
    print(f"  {size:>10} | {rate:6.1%} +/-{standard_error(rate, RUNS):4.1%} "
          f"| [{interval[0]:5.1%}, {interval[1]:5.1%}]   "
          f"| {mean_cost:6.1f}, {max(costs):>4}  | {1000 * rate / mean_cost:9.2f}")
# ------------------------------------------------------------------------------

# ===== Part 1: effectiveness climbs, efficiency falls =========================
(first_size, _, first_rate, _, first_cost, _, _), \
    (last_size, _, last_rate, _, last_cost, _, _) \
    = sweep[0], sweep[-1]
print(f"\nRead the table both ways. From population {first_size} to "
      f"{last_size}:")
print(f"    the success rate climbs {first_rate:.1%} -> {last_rate:.1%} "
      f"(+{last_rate - first_rate:.1%}),")
print(f"    the cost of a run climbs {first_cost:.0f} -> {last_cost:.0f} "
      f"evaluations (x{last_cost / first_cost:.1f}),")
print(f"    and the yield falls {1000 * first_rate / first_cost:.2f} -> "
      f"{1000 * last_rate / last_cost:.2f} successes per 1000 evaluations.")
print("Effectiveness and efficiency pull in opposite directions, "
      "monotonically, both of")
print("them. There is no configuration that is best at both - there is only a "
      "choice,")
print("and Lesson 07 exists to make it.")

# ===== Part 2: the benchmark nobody ran yet ===================================
print(f"\nNow spend the same budgets blind. One uniform draw hits the target "
      f"{share:.2%} of")
print("the time; E independent draws hit at least once with probability "
      "1 - (1 - q) ** E:\n")
print("  budget (evaluations) | blind draws | the GA at that budget | margin")
print("  ---------------------+-------------+-----------------------+-------")
for size, _, rate, _, mean_cost, _, blind in (sweep[0], sweep[-1]):
    print(f"  {mean_cost:>20.0f} | {blind:10.1%} | {rate:21.1%} | "
          f"{rate - blind:+5.1%}")
print(f"\nOn a target this wide - {share:.2%} of the whole box - the machinery "
      "of Lessons 01-05")
print("buys a few points over rolling dice, not a chasm. That is a property "
      "of the problem,")
print("not a defect of the algorithm: where the target has no width worth "
      "naming - the")
print("knapsacks, schedules and tours of Lessons 09 to 11 - a blind draw "
      "essentially never")
print("hits, and the operators are the only game in town.")

# The picture: rate against cost, with the blind benchmark; and the yield.
fig, (ax_rate, ax_yield) = plt.subplots(1, 2, figsize=(11.5, 4.0))
sizes = [row[0] for row in sweep]
rates = [row[2] for row in sweep]
costs = [row[4] for row in sweep]
intervals = [row[3] for row in sweep]
errors = ([rate - interval[0] for rate, interval in zip(rates, intervals)],
          [interval[1] - rate for rate, interval in zip(rates, intervals)])

budget_line = np.linspace(min(costs) * 0.9, max(costs) * 1.05, 200)
ax_rate.plot(budget_line, [blind_success(share, e) for e in budget_line],
             color="tab:gray", linestyle="--", linewidth=1.4,
             label="blind draws, same budget")
ax_rate.errorbar(costs, rates, yerr=errors, fmt="o-", color="tab:blue",
                 capsize=4, linewidth=1.6,
                 label="the GA, 95% Wilson intervals")
for size, _, rate, _, cost, _, _ in sweep:
    ax_rate.annotate(f"pop {size}", xy=(cost, rate),
                     xytext=(0, 9), textcoords="offset points",
                     ha="center", fontsize=8)
ax_rate.set_title("Success rate against cost: the margin over blind is slim",
                  fontsize=10)
ax_rate.set_xlabel("evaluations per run")
ax_rate.set_ylabel("success rate")
ax_rate.grid(True, linestyle=":", alpha=0.5)
ax_rate.legend(fontsize=8, loc="lower right")

ax_yield.plot(sizes, [1000 * row[2] / row[4] for row in sweep], "o-",
              color="tab:red", linewidth=1.6)
ax_yield.set_title("The yield falls as the population grows", fontsize=10)
ax_yield.set_xlabel("population size")
ax_yield.set_ylabel("successes per 1000 evaluations")
ax_yield.grid(True, linestyle=":", alpha=0.5)

fig.suptitle("Effectiveness climbs, efficiency falls - there is only a choice")
FIGURES.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURES / "success_rate_04_the_budget.png", dpi=150,
            bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved to {FIGURES}/success_rate_04_the_budget.png")

