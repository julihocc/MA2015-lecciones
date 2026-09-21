"""
Lesson 11 - Equations 3: The genetic algorithm solves it
=========================================================
NEW IN THIS STEP: Individual, the three operators, and run().

Nothing here is new machinery: rank selection with elitism came from lesson 03,
blend crossover from lesson 04, Gaussian deviation from lesson 05. What is new
is the stopping rule. Every other lesson stopped after a fixed budget because it
could not tell whether it had won. This one stops when the residual is zero,
because that is a proof.

CHANGES FROM equations_02_random_population.py
Introduce them in this order:
    1. Individual                    a candidate that carries its own residual
    2. selection_rank_with_elite()   rank selection, two elites carried through
    3. crossover_blend()             real-valued blend, then clamped back to integers
    4. mutation_random_deviation()   Gaussian nudge on each gene with probability 0.5
    5. run()                         the loop, stopping the moment the residual is 0

Run it:  python equations_03_the_full_ga.py

Seed 3 reaches residual 0 at generation 5 after 2,794 evaluations, at
x = -6, y = 2, z = 3; f, g and w are each individually 0.
"""
from math import factorial
from pathlib import Path
import random
from typing import List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 3
FIGURES = Path(__file__).resolve().parent.parent / "figures"
POPULATION_SIZE = 400
CROSSOVER_PROBABILITY = 0.8
MUTATION_PROBABILITY = 0.4
MAX_GENERATIONS = 100
ELITE_SIZE = 2

BOX_LOW, BOX_HIGH = -20, 20


def f(x: int, y: int, z: int) -> int:
    """First equation of the system; a solution makes it zero.

    Args:
        x, y, z: integers in [-20, 20].

    Returns:
        One integer residual.
    """
    return (x * y + 2) ** 2 + (x + y) ** (abs(z - 2) ** 3 + 2) + x * y * z


def g(x: int, y: int, z: int) -> int:
    """Second equation. The factorial is why z has to stay small.

    Args:
        x, y, z: integers in [-20, 20].

    Returns:
        One integer residual.
    """
    return x * (y * z + 10) - factorial(abs(z - 3)) + y ** abs(x) + 11 * z


def w(x: int, y: int, z: int) -> int:
    """Third equation.

    Args:
        x, y, z: integers in [-20, 20].

    Returns:
        One integer residual.
    """
    return (x + 7 * y) ** abs(z + x) - (z + 16) ** 2 - 151


def total_error(x: int, y: int, z: int) -> int:
    """Sum of absolute residuals. Exactly zero means an exact solution.

    Args:
        x, y, z: integers in [-20, 20].

    Returns:
        ``|f|+|g|+|w|``. Seed 3's champion scores exactly 0.
    """
    try:
        return abs(f(x, y, z)) + abs(g(x, y, z)) + abs(w(x, y, z))
    except (ValueError, ZeroDivisionError):
        return 10 ** 100


def digits(error: int) -> int:
    """Decimal length of the residual, computed from the bit length because
    these integers are routinely too long for str() to convert.

    Args:
        error: a non-negative integer residual.

    Returns:
        Number of decimal digits.
    """
    if error == 0:
        return 1
    estimate = int(error.bit_length() * 0.30103) + 1
    while 10 ** (estimate - 1) > error:
        estimate -= 1
    while 10 ** estimate <= error:
        estimate += 1
    return estimate


def clamp(value: float) -> int:
    """Genes are integers inside the box; crossover and mutation are not.

    Args:
        value: a proposed real gene.

    Returns:
        The nearest integer in [-20, 20].
    """
    return max(BOX_LOW, min(BOX_HIGH, round(value)))


def random_triple() -> Tuple[int, int, int]:
    """One uniform draw from the box. This is the whole of 'random search'.

    Returns:
        An ``(x, y, z)`` triple with each gene in [-20, 20].
    """
    return tuple(random.randint(BOX_LOW, BOX_HIGH) for _ in range(3))


# --- NEW (1) Individual -------------------------------------------------------
class Individual:
    """A candidate triple. Fitness is the negated residual, so larger is better
    and the target value 0 is the largest fitness the problem admits.

    Args:
        genes: three numbers, rounded and clamped into [-20, 20].

    Example:
        Seed 3's champion is ``x = -6, y = 2, z = 3`` with residual 0.
    """

    def __init__(self, genes: Sequence[float]) -> None:
        """Clamp genes and store negated residual as fitness.

        Args:
            genes: three numbers, not necessarily integers.
        """
        self.genes: List[int] = [clamp(gene) for gene in genes]
        self.fitness: int = -total_error(*self.genes)

    def __str__(self) -> str:
        return f"x={self.genes[0]}, y={self.genes[1]}, z={self.genes[2]}"
# ------------------------------------------------------------------------------


# --- NEW (2) selection_rank_with_elite() --------------------------------------
def selection_rank_with_elite(individuals: List[Individual],
                              elite_size: int = 0) -> List[Individual]:
    """Rank selection: the sampling weight depends on position, not on fitness.
    That is the only thing that works here - the fitness values differ by
    thousands of orders of magnitude, so proportional selection would give the
    whole population to whichever individual happens to lead.

    Args:
        individuals: the current generation.
        elite_size: how many of the current best are copied through. 2 here.

    Returns:
        A new list of the same length.
    """
    ordered = sorted(individuals, key=lambda ind: ind.fitness, reverse=True)
    rank_distance = 1.0 / len(individuals)
    ranks = [1.0 - index * rank_distance for index in range(len(individuals))]
    ranks_sum = sum(ranks)
    selected = ordered[:elite_size]
    for _ in range(len(ordered) - elite_size):
        threshold = random.random() * ranks_sum
        running = 0.0
        for index, rank in enumerate(ranks):
            running += rank
            if running > threshold:
                selected.append(ordered[index])
                break
    return selected
# ------------------------------------------------------------------------------


# --- NEW (3) crossover_blend() ------------------------------------------------
def crossover_blend(first: Sequence[int], second: Sequence[int],
                    alpha: float = 0.8) -> Tuple[List[float], List[float]]:
    """Sample each child gene from an interval stretched beyond the parents, so
    the pair can reach values outside the segment that separates them.

    Args:
        first, second: parent gene lists, already integers.
        alpha: blend extension. 0.8 here.

    Returns:
        Two real-valued children; ``Individual`` clamps them back.
    """
    child_one, child_two = list(first), list(second)
    for index in range(len(first)):
        span = abs(child_two[index] - child_one[index])
        low = min(child_one[index], child_two[index]) - alpha * span
        high = max(child_one[index], child_two[index]) + alpha * span
        child_one[index] = low + random.random() * (high - low)
        child_two[index] = low + random.random() * (high - low)
    return child_one, child_two
# ------------------------------------------------------------------------------


# --- NEW (4) mutation_random_deviation() --------------------------------------
def mutation_random_deviation(genes: Sequence[int], sigma: float = 3.0,
                              probability: float = 0.5) -> List[float]:
    """A Gaussian nudge. Rounding back to integers is what makes it a jump.

    Args:
        genes: parent genes, already integers.
        sigma: 3.0, large enough to hop to a neighbouring integer.
        probability: per-gene coin. 0.5 here.

    Returns:
        A real-valued mutant; ``Individual`` clamps it back.
    """
    mutant = list(genes)
    for index in range(len(mutant)):
        if random.random() < probability:
            mutant[index] += random.gauss(0.0, sigma)
    return mutant
# ------------------------------------------------------------------------------


# --- NEW (5) run() ------------------------------------------------------------
def run() -> Tuple[Individual, int, int, List[int]]:
    """Return the best individual, the generation reached, the number of
    residual evaluations spent, and the best residual after each generation.

    Returns:
        ``(best, generation, evaluations, history)``. Stops at residual 0.

    Example:
        Seed 3 reaches residual 0 at generation 5 after 2,794 evaluations,
        at ``x = -6, y = 2, z = 3``.
    """
    random.seed(SEED)
    population = [Individual(random_triple()) for _ in range(POPULATION_SIZE)]
    evaluations = POPULATION_SIZE
    best = max(population, key=lambda ind: ind.fitness)
    history = [-best.fitness]
    generation = 0
    while generation < MAX_GENERATIONS and best.fitness != 0:
        generation += 1
        parents = selection_rank_with_elite(population, ELITE_SIZE)
        crossed: List[Individual] = []
        for one, two in zip(parents[::2], parents[1::2]):
            if random.random() < CROSSOVER_PROBABILITY:
                genes_one, genes_two = crossover_blend(one.genes, two.genes)
                crossed += [Individual(genes_one), Individual(genes_two)]
                evaluations += 2
            else:
                crossed += [one, two]
        population = []
        for candidate in crossed:
            if random.random() < MUTATION_PROBABILITY:
                population.append(Individual(mutation_random_deviation(candidate.genes)))
                evaluations += 1
            else:
                population.append(candidate)
        champion = max(population, key=lambda ind: ind.fitness)
        if champion.fitness > best.fitness:
            best = champion
        history.append(-best.fitness)
    return best, generation, evaluations, history
# ------------------------------------------------------------------------------


best, generations, evaluations, history = run()
side = BOX_HIGH - BOX_LOW + 1
x, y, z = best.genes

print("Lesson 11 - Equations 3: the genetic algorithm solves it")
print(f"Seed {SEED}, population {POPULATION_SIZE}, generation cap {MAX_GENERATIONS}")
print(f"Stopped at generation:   {generations}")
print(f"Residual evaluations:    {evaluations:,}")
print(f"Best candidate:          {best}")
print(f"Residual:                {-best.fitness}")
print("Checking the three equations one by one, which is the point of this problem:")
print(f"  f({x}, {y}, {z}) = {f(x, y, z)}")
print(f"  g({x}, {y}, {z}) = {g(x, y, z)}")
print(f"  w({x}, {y}, {z}) = {w(x, y, z)}")
print(f"All three are zero:      {f(x, y, z) == g(x, y, z) == w(x, y, z) == 0}")
print(f"Evaluations spent versus the size of the whole box: "
      f"{evaluations:,} of {side ** 3:,} ({evaluations / side ** 3:.0%})")
print("That last line is the question step 4 answers.")

FIGURES.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(range(len(history)), [digits(error) for error in history], "o-")
ax.set(xlabel="generation", ylabel="decimal digits of the best residual",
       title=f"Seed {SEED}: the residual reaches exactly zero at generation {generations}")
ax.axhline(1, color="crimson", linestyle="--", label="residual = 0")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "equations_03_the_full_ga.png", dpi=160)
plt.close(fig)

