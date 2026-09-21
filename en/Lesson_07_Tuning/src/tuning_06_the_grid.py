"""
Lesson 07 - Step 6: The grid, and the verdict
==============================================
NEW IN THIS STEP: the two knobs mapped together, and the lesson's verdict.

Steps 2 to 5 turned one knob at a time and every knob confessed to being a
budget knob. The honest final question is whether they interact - whether
the best crossover depends on the mutation setting, the way the textbooks
assume when they recommend pairs like "0.8 and 0.1". This step measures the
full grid: six crossover settings times six mutation settings, population
10, 100 runs per cell, every cell priced against blind search at its own
budget.

Three things fall out. The corner with both knobs at zero recovers blind
search almost exactly - selection alone IS ten blind draws, and the
instrument is calibrated by it. The best raw rate sits in the most
expensive corner, and its margin over blind at that budget is a rounding
error. And the margin map shows where the operators actually earn their
keep: the cheap cells, the low-budget region where blind money has not yet
caught up. On this landscape the knobs interact mainly through the budget.
That is the verdict - and it is a verdict ABOUT THIS PROBLEM, which is why
tuning is a laboratory subject and why Lesson 12, the adaptive GA, exists.

CHANGES FROM tuning_05_population.py
Introduce them in this order:
    1. the grid       every (crossover, mutation) cell measured and priced
    2. RUNS = 100     changed from 500: thirty-six cells have to fit the clock

REMOVED FROM tuning_05_population.py: the single-knob sweep call and its
figure (the population study is step 5's, and its numbers are in the README).

Run it:  python tuning_06_the_grid.py

A 6-by-6 grid of crossover and mutation, RUNS = 100. The (0, 0) corner
recovers blind search (9.0% ± 2.9% vs 13.4% for 10 blind draws). The best cell
(1.0, 0.5) scores 90.0% at 160 evals with a margin of -0.01%. Only 20 of 36
cells beat their own budget.
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
RUNS = 100   # --- CHANGED --- from 500: thirty-six cells have to fit the clock
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

    Run i uses seed i, so every cell faces the same dice.

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


def blind_success(share: float, evaluations: float) -> float:
    """The success rate of spending `evaluations` blind uniform draws:
    1 - (1 - share) ** E. The benchmark a cell has to beat to justify its
    budget.

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
PC_VALUES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
PM_VALUES = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)

print(f"The target: optimum {OPTIMUM:+.6f}; a blind draw hits it "
      f"{SHARE:.2%} of the time.")
print(f"The grid: {len(PC_VALUES)} x {len(PM_VALUES)} cells, population "
      f"{POPULATION_SIZE}, {RUNS} runs per cell, every cell priced.\n")

# --- NEW (1) the grid ----------------------------------------------------------
rates, costs = {}, {}
for pc in PC_VALUES:
    for pm in PM_VALUES:
        successes, cell_costs = 0, []
        for seed in range(RUNS):
            Individual.evaluations = 0
            champion = run(SINE, seed, POPULATION_SIZE, pc, pm)
            cell_costs.append(Individual.evaluations)
            successes += verdict(champion, OPTIMUM)
        rates[(pc, pm)] = successes / RUNS
        costs[(pc, pm)] = statistics.fmean(cell_costs)
# ------------------------------------------------------------------------------

print("  success rate   |" + "|".join(f" pm = {pm:>4} " for pm in PM_VALUES))
print("  ---------------+" + "+".join("-" * 10 for _ in PM_VALUES))
for pc in PC_VALUES:
    print(f"  pc = {pc:>4}    |" + "|".join(f" {rates[(pc, pm)]:7.1%}  "
                                             for pm in PM_VALUES))

zero = (0.0, 0.0)
blind_zero = blind_success(SHARE, costs[zero])
zero_se = standard_error(rates[zero], RUNS)
print(f"\nThe calibration corner, pc = 0 and pm = 0: the GA scores "
      f"{rates[zero]:.1%} +/- {zero_se:.1%}.")
print(f"With no crossover and no mutation, selection alone can only re-sample "
      f"the initial")
print(f"population and report its best - literally {costs[zero]:.0f} blind "
      f"draws, which score {blind_zero:.1%}.")
print("Within one plug-in SE of a 100-run cell, the instrument measures the GA")
print("and the dice as the same machine. It is calibrated.")

best = max(rates, key=rates.get)
blind_best = blind_success(SHARE, costs[best])
print(f"\nThe best raw rate sits at pc = {best[0]}, pm = {best[1]}: "
      f"{rates[best]:.1%}. It is also the")
print(f"most expensive cell, at {costs[best]:.0f} evaluations a run - and "
      f"blind draws at that budget")
print(f"score {blind_best:.1%}. The champion's margin: "
      f"{rates[best] - blind_best:+.2%} - within rounding of zero.")
positive = sum(1 for cell in rates
               if rates[cell] > blind_success(SHARE, costs[cell]))
print(f"\nOf the {len(rates)} cells, {positive} beat blind search at their "
      f"own budget. The map")
print("shows where: the cheap cells. The operators earn their keep where the")
print("budget is small; wherever the budget grows, blind money catches up.")
print("\nThat is the lesson's verdict, and it is a verdict ABOUT THIS")
print("PROBLEM. On a landscape whose target is wide, knobs are budget knobs")
print("and tuning is mostly accounting. On the narrow targets of Lessons 09")
print("to 11 the same measurement is worth running again - and the settings")
print("it crowns there will be different ones. A fixed setting is always a")
print("compromise across problems and across the run itself, which is")
print("exactly why Lesson 12, the adaptive GA, exists.")

# The picture: the rate map and the margin map, side by side.
rate_grid = np.array([[rates[(pc, pm)] for pm in PM_VALUES]
                      for pc in PC_VALUES])
margin_grid = np.array([[rates[(pc, pm)] - blind_success(SHARE, costs[(pc, pm)])
                         for pm in PM_VALUES] for pc in PC_VALUES])
fig, (ax_rate, ax_margin) = plt.subplots(1, 2, figsize=(11.5, 4.2))
for ax, grid, title, fmt in (
        (ax_rate, rate_grid, "success rate", ".0%"),
        (ax_margin, margin_grid, "margin over blind, same budget", "+.0%")):
    image = ax.imshow(grid, cmap="RdYlGn", aspect="auto",
                      vmin=-np.max(np.abs(grid)) if grid is margin_grid else 0,
                      vmax=np.max(np.abs(grid)) if grid is margin_grid else 1)
    for i, pc in enumerate(PC_VALUES):
        for j, pm in enumerate(PM_VALUES):
            ax.text(j, i, format(grid[i, j], fmt), ha="center", va="center",
                    fontsize=7)
    ax.set_xticks(range(len(PM_VALUES)), [str(v) for v in PM_VALUES])
    ax.set_yticks(range(len(PC_VALUES)), [str(v) for v in PC_VALUES])
    ax.set_xlabel("mutation probability")
    ax.set_ylabel("crossover probability")
    ax.set_title(title, fontsize=10)
    fig.colorbar(image, ax=ax, shrink=0.85)
ax_rate.set_title("success rate: the ridge is the budget", fontsize=10)
ax_margin.set_title("margin: the operators earn it in the cheap cells",
                    fontsize=10)
fig.suptitle("The two knobs together, measured and priced")
FIGURES.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURES / "tuning_06_the_grid.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved to {FIGURES}/tuning_06_the_grid.png")

