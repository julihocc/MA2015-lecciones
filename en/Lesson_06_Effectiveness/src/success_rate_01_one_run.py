"""
Lesson 06 - Step 1: One run, and what it is worth
==================================================
NEW IN THIS STEP: nothing is new to the students. This is Lessons 01-05's
algorithm, unchanged, pointed at the two-dimensional landscape Gridin opens
Chapter 6 with - and then run eight times instead of once.

Every lesson so far ended by reading a single run. Lesson 01 read one run and
concluded the algorithm works; Lesson 02 read one run and concluded a stopping
rule is free; Lesson 03 and Lesson 05 both closed with an open question they
could not answer, and both questions had the same shape: does this change help,
over runs? This lesson is the machinery for answering that, and it starts by
showing why the question cannot be dodged.

A genetic algorithm is a random variable. It takes a seed and returns an answer,
and the answer changes when the seed does - not by a rounding error, but by more
than the whole range of improvement the run achieves. The per-generation curve
printed below looks like a convergence proof. It is one sample.

The problem is the book's: f(x, y) = sin(x)cos(x) - (|(x + 50)(y - 10)| / 10)^0.1
on [-100, 100]^2. It is worth reading before running anything. The first term
oscillates between -0.5 and +0.5. The second term is a penalty that vanishes
only on the two lines x = -50 and y = 10, and it approaches zero so slowly - a
tenth power - that being close is worth almost nothing. Step 2 measures what
that costs.

The operators are the course's, not the book's: tournament of 3 (Lesson 03),
blend crossover with alpha = 1.0 (Lesson 04), gaussian mutation with sigma = 1.0
behind a per-individual coin (Lesson 05). Keeping them means every number in
this lesson continues the earlier ones instead of restarting them.

Run it:  python success_rate_01_one_run.py

book_2d() is Gridin's two-dimensional landscape. run() is the course GA,
unchanged, pointed at it. history() records average, best-of-gen and
best-ever. Across 8 seeds the answers spread 0.8946 — 1.5 times the tracked
run's whole gain of +0.5822. Seed 3 reports -0.0129, seed 4 reports -0.9076,
and both reports are true.
"""
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

# The algorithm, exactly as Lessons 01-05 left it.
POPULATION_SIZE = 10
CROSSOVER_PROBABILITY = 0.8
MUTATION_PROBABILITY = 0.1
MUTATION_MU, MUTATION_SIGMA = 0.0, 1.0
TOURNAMENT_SIZE, BLEND_ALPHA = 3, 1.0
# Run i uses seed i. That convention holds for the whole lesson: it makes any
# subset of runs reproducible on its own, and it lets a later step reproduce
# another lesson's block of runs exactly by asking for the same seeds.
RUNS = 8
FIGURES = Path(__file__).resolve().parent.parent / "figures"


@dataclass(frozen=True)
class Problem:
    """A landscape, its box, and the generation budget it is run with.

    Lesson 02 made the fitness function an argument. Here the bounds, the number
    of genes and the budget travel with it, because this lesson compares runs
    and a comparison is only fair if both sides were given the same budget.

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


def book_2d(genes: list[float]) -> float:
    """Gridin's Chapter 6 landscape. The penalty term is the interesting half.

    Args:
        genes: [x, y] in [-100, 100]^2.

    Returns:
        sin(x)cos(x) minus a tenth-power penalty that vanishes on
        x = -50 and y = 10.

    Example:
        The dense-grid reference is +0.500000, but it occupies
        0.003197% of the grid — all 128 winning points sit on y = 10.
    """
    x, y = genes
    return np.sin(x) * np.cos(x) - pow(abs((x + 50.0) * (y - 10.0)) / 10.0, 0.1)


TWO_D = Problem("book 2-D", book_2d, -100.0, 100.0, 2, 25)


class Individual:
    """One candidate solution, carrying the problem that judges it.


    Example:
        Eight seeds on book_2d spread 0.8946 in best-ever fitness; on sine,
        5 of 8 runs land within 0.01 of the optimum.
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

    This is Lesson 05's convention (per-gene probability 1.0), and it matters
    here for a reason that has nothing to do with biology: it is the operator
    whose random stream Lesson 05 measured, so step 5 can reproduce its runs.

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


def evolve_one_generation(population: list[Individual]) -> list[Individual]:
    """SELECT -> CROSSOVER -> MUTATE -> replace, once.

    Args:
        population: current generation.

    Returns:
        The next generation, and in Lesson 02 a census of pairs / crossed
        / mutated / created.

    Example:
        Created counts the true evaluation cost: 107 rather than 110 in
        the Lesson 01-reproducing run, because an un-crossed, un-mutated
        selected individual is the same object.
    """
    selected = select_tournament(population)
    crossed: list[Individual] = []
    for parent1, parent2 in zip(selected[::2], selected[1::2]):
        if random.random() < CROSSOVER_PROBABILITY:
            crossed.extend(crossover(parent1, parent2))
        else:
            crossed.extend([parent1, parent2])
    return [mutate(ind) if random.random() < MUTATION_PROBABILITY else ind
            for ind in crossed]


def run(problem: Problem, seed: int) -> list[list[Individual]]:
    """One complete run: initialise, then evolve for the problem's budget.

    Args:
        problem: landscape, box, gene count and generation budget.
        seed: run i uses seed i, so subsets of runs are reproducible.

    Returns:
        The list of generations.

    Example:
        Eight seeds on book_2d spread 0.8946 in best-ever fitness.
    """
    random.seed(seed)
    population = [Individual([random.uniform(problem.low, problem.high)
                              for _ in range(problem.genes)], problem)
                  for _ in range(POPULATION_SIZE)]
    generations = [population]
    for _ in range(problem.generations):
        population = evolve_one_generation(population)
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


def history(generations: list[list[Individual]]) -> list[tuple[float, float, float]]:
    """Average, best-of-generation and best-ever, one row per generation.

    Args:
        generations: every generation of one run.

    Returns:
        Per-generation curves used to draw the figure (average / best
        / best-ever, or best-ever only, depending on the lesson).

    Example:
        The tracked Lesson 06 run gains +0.5822; eight seeds of the
        same configuration spread 0.8946.
    """
    rows, ever = [], -float("inf")
    for population in generations:
        fitnesses = [ind.fitness for ind in population]
        ever = max(ever, max(fitnesses))
        rows.append((statistics.fmean(fitnesses), max(fitnesses), ever))
    return rows


# ===== Part 1: read one run the way every lesson so far has read one ==========
print(f"One run on the {TWO_D.name} landscape, seed 0, "
      f"{TWO_D.generations} generations:\n")
rows = history(run(TWO_D, 0))
print("  generation |  average |  best of gen |  best ever")
print("  -----------+----------+--------------+-----------")
for generation, (average, best, ever) in enumerate(rows):
    print(f"  {generation:>10} | {average:+8.4f} | {best:+12.4f} | {ever:+10.4f}")
first_ever, last_ever = rows[0][2], rows[-1][2]
print(f"\nThe run improved its best-ever from {first_ever:+.4f} to "
      f"{last_ever:+.4f}, a gain of {last_ever - first_ever:+.4f}.")
print("Average fitness follows it up. The curve is monotone, the population")
print("catches up with its champion, and every lesson so far would have stopped")
print("reading here.")

# ===== Part 2: the same configuration, eight times ============================
print(f"\nThe same configuration, nothing changed but the seed, {RUNS} times:\n")
print("  seed |  best ever found | where")
print("  -----+------------------+---------------------------")
answers = []
for seed in range(RUNS):
    champion = best_ever(run(TWO_D, seed))
    answers.append(champion.fitness)
    where = ", ".join(f"{g:+.2f}" for g in champion.genes)
    print(f"  {seed:>4} | {champion.fitness:+16.4f} | ({where})")

spread = max(answers) - min(answers)
gain = last_ever - first_ever
print(f"\nBest answer {max(answers):+.4f}, worst {min(answers):+.4f}, "
      f"spread {spread:.4f}.")
print(f"Mean {statistics.fmean(answers):+.4f}, standard deviation "
      f"{statistics.stdev(answers):.4f}.")
print(f"\nThe spread across seeds is {spread / gain:.1f} times the improvement")
print(f"the tracked run achieved ({gain:+.4f}). Which seed you happened to use")
print("decides more about the answer than the search does.")
worst_seed = min(range(RUNS), key=lambda s: answers[s])
best_seed = max(range(RUNS), key=lambda s: answers[s])
print(f"\nSeed {best_seed} would have you report {max(answers):+.4f}; seed "
      f"{worst_seed} would have you report {min(answers):+.4f}.")
print("Both reports would be true, and neither would be about the algorithm.")
print("\nSo: how many runs, and what exactly should be counted? Step 2 has to")
print("settle what counts as a success before it can be counted at all.")

# The picture: the tracked run's three curves, and the eight best-ever curves.
fig, (ax_one, ax_many) = plt.subplots(1, 2, figsize=(11.5, 4.0))
ax_one.plot([r[0] for r in rows], label="average of generation")
ax_one.plot([r[1] for r in rows], label="best of generation")
ax_one.plot([r[2] for r in rows], label="best ever", linewidth=2.0)
ax_one.set_title("One run, seed 0: the picture every lesson so far read",
                 fontsize=10)
ax_one.set_xlabel("generation")
ax_one.set_ylabel("fitness")
ax_one.grid(True, linestyle=":", alpha=0.5)
ax_one.legend(fontsize=8, loc="lower right")

for seed in range(RUNS):
    ax_many.plot([r[2] for r in history(run(TWO_D, seed))],
                 label=f"seed {seed}", linewidth=1.3)
ax_many.set_title(f"Best ever, {RUNS} seeds: spread {spread:.2f} "
                  f"vs one run's gain {gain:.2f}", fontsize=10)
ax_many.set_xlabel("generation")
ax_many.set_ylabel("best fitness so far")
ax_many.grid(True, linestyle=":", alpha=0.5)
ax_many.legend(fontsize=7, ncol=2, loc="lower right")

fig.suptitle("A genetic algorithm is a random variable, not a procedure")
FIGURES.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURES / "success_rate_01_one_run.png", dpi=150,
            bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved to {FIGURES}/success_rate_01_one_run.png")

