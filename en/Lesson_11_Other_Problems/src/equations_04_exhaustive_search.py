"""
Lesson 11 - Equations 4: The method that does not need a genetic algorithm
==========================================================================
NEW IN THIS STEP: enumerate_box(), a seeded run(), and the honest comparison.

The genes are integers and the box is finite, so the entire search space can be
walked. This is the step where the course finally asks the question it has been
avoiding: given that a genetic algorithm is a heuristic, when is it the wrong
tool? The answer here is not the one the evaluation counts suggest, and the
step is written around what the code actually measures.

CHANGES FROM equations_03_the_full_ga.py
Introduce them in this order:
    1. enumerate_box()   walk every integer triple in the box and keep every root
    2. run()             take the seed as an argument, so the GA can be repeated
    3. the comparison    cost, reliability, and what each method can claim

Run it:  python equations_04_exhaustive_search.py

All 5 seeds find the same root, in 5–28 generations and 2,794–13,764
evaluations. Enumerating the box costs 68,921 evaluations and returns
exactly one root.
"""
from math import factorial
from pathlib import Path
import random
import time
from typing import Dict, List, Sequence, Tuple

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
        ``|f|+|g|+|w|``. Enumeration finds exactly one triple where this is 0.
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


class Individual:
    """A candidate triple. Fitness is the negated residual, so larger is better
    and the target value 0 is the largest fitness the problem admits.

    Args:
        genes: three numbers, rounded and clamped into [-20, 20].
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


def run(seed: int) -> Tuple[Individual, int, int, List[int]]:   # --- CHANGED --- (2) run() takes the seed
    """Return the best individual, the generation reached, the number of
    residual evaluations spent, and the best residual after each generation.

    Args:
        seed: the only input that differs across the five GA runs.

    Returns:
        ``(best, generation, evaluations, history)``.

    Example:
        All five seeds find the same root in 5–28 generations and
        2,794–13,764 evaluations.
    """
    random.seed(seed)
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


# --- NEW (1) enumerate_box() --------------------------------------------------
def enumerate_box() -> Tuple[List[Tuple[int, int, int]], int, Dict[int, int]]:
    """Evaluate every integer triple in the box. Returns every exact root, the
    number of evaluations, and the smallest residual digit count per z slice.

    Returns:
        ``(roots, evaluations, per_slice)``. ``evaluations`` is 68,921.

    Example:
        Exactly one root: ``(-6, 2, 3)``. That uniqueness is what the GA
        cannot claim.
    """
    roots: List[Tuple[int, int, int]] = []
    per_slice: Dict[int, int] = {}
    evaluations = 0
    for z in range(BOX_LOW, BOX_HIGH + 1):
        smallest = None
        for x in range(BOX_LOW, BOX_HIGH + 1):
            for y in range(BOX_LOW, BOX_HIGH + 1):
                error = total_error(x, y, z)
                evaluations += 1
                size = digits(error) if error else 0
                if smallest is None or size < smallest:
                    smallest = size
                if error == 0:
                    roots.append((x, y, z))
        per_slice[z] = smallest
    return roots, evaluations, per_slice
# ------------------------------------------------------------------------------


# --- NEW (3) the comparison ---------------------------------------------------
SEEDS = [3, 1, 7, 16, 42]
side = BOX_LOW * 0 + (BOX_HIGH - BOX_LOW + 1)

ga_start = time.perf_counter()
ga_results = [run(seed) for seed in SEEDS]
ga_seconds = time.perf_counter() - ga_start
solved = [result for result in ga_results if result[0].fitness == 0]
ga_evaluations = [result[2] for result in ga_results]
ga_generations = [result[1] for result in ga_results]

exhaustive_start = time.perf_counter()
roots, exhaustive_evaluations, per_slice = enumerate_box()
exhaustive_seconds = time.perf_counter() - exhaustive_start
# ------------------------------------------------------------------------------

best, generations, evaluations, history = ga_results[0]

print("Lesson 11 - Equations 4: the method that does not need a genetic algorithm")
print(f"Search box: {side ** 3:,} integer triples")
print()
print(f"Genetic algorithm, seeds {SEEDS}:")
print(f"  found an exact root:   {len(solved)} of {len(SEEDS)} runs")
print(f"  generations needed:    {min(ga_generations)} to {max(ga_generations)}")
print(f"  residual evaluations:  {min(ga_evaluations):,} to {max(ga_evaluations):,}")
print(f"  total wall time:       {ga_seconds:.2f} s for {len(SEEDS)} runs")
print(f"  roots reported:        {len({tuple(result[0].genes) for result in solved})} distinct")
print()
print("Exhaustive enumeration of the same box:")
print(f"  residual evaluations:  {exhaustive_evaluations:,}")
print(f"  wall time:             {exhaustive_seconds:.2f} s")
print(f"  roots found:           {len(roots)} -> {roots}")
print()
ratio = exhaustive_evaluations / (sum(ga_evaluations) / len(ga_evaluations))
print(f"The genetic algorithm is cheaper: it spends on average "
      f"{sum(ga_evaluations) / len(ga_evaluations):,.0f} evaluations, "
      f"{ratio:.0f} times fewer than enumeration.")
print("It is still the weaker choice here, and the reason is not the cost:")
print(f"  - enumeration proves the box holds exactly {len(roots)} root; the GA "
      f"cannot say that,")
print("  - enumeration has no seed, no population size and no crossover rate,")
print(f"  - enumeration always returns the same answer; the GA returned one here "
      f"in {len(solved)} of {len(SEEDS)} runs, which is a measurement, not a guarantee.")
wider = 201
print(f"What would change the verdict is the box. It costs side^3, so widening "
      f"it to [-100, 100] means {wider ** 3:,} triples, "
      f"{wider ** 3 / side ** 3:.0f} times this run.")
print("A genetic algorithm earns its place when the space stops being walkable,")
print("not when it happens to beat a loop you could have written in four lines.")

FIGURES.mkdir(exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].bar(["GA (mean of\n%d seeds)" % len(SEEDS), "exhaustive"],
            [sum(ga_evaluations) / len(ga_evaluations), exhaustive_evaluations],
            color=["#4c72b0", "#c44e52"])
axes[0].set(ylabel="residual evaluations", title="Cost is not the argument")
axes[0].set_yscale("log")
zs = sorted(per_slice)
axes[1].plot(zs, [per_slice[z] for z in zs], "o-")
axes[1].set(xlabel="z", ylabel="smallest residual digits in the slice",
            title="Enumeration also maps the box: the only zero is at z = %d" % roots[0][2])
fig.tight_layout()
fig.savefig(FIGURES / "equations_04_exhaustive_search.png", dpi=160)
plt.close(fig)

