"""
Lesson 03 - Step 1: The population, and how to measure what selection does
===========================================================================
Selection never invents anything. It takes a population of N and returns a
population of N, built only from copies of what was already there. So the only
thing worth measuring is the BOOKKEEPING: who got copied, how many times, and
who disappeared.

This step builds nothing but that measuring apparatus, and applies it to the
cheapest possible selection - the one that copies everybody exactly once. Every
method in steps 2 to 6 is measured against this baseline.

The population is the same ten individuals as Lesson 01 (same objective, same
SEED = 52), so the numbers continue rather than restart.

Run it:  python selection_pressure_01_the_population.py

create_population() rebuilds Lesson 01's ten individuals (SEED = 52).
selection_counts() tallies copies by identity, gene_spread() is the diversity
number, and select_nothing() is the zero-pressure baseline: 0 extinct,
diversity 6.665, one copy of the best. Fitness is negative almost everywhere,
so no roulette wheel can be built yet.
"""
import random
import statistics
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SEED = 52
POPULATION_SIZE = 10
GENE_MIN, GENE_MAX = -10.0, 10.0
FIGURES = Path(__file__).resolve().parent.parent / "figures"


def objective(x):
    """The function we want to MAXIMISE - Lesson 01's landscape, unchanged.

    Works on a single float and on a numpy array, so the same line draws the
    curve and scores an individual.

    Args:
        x: a real gene, typically in [-10, 10].

    Returns:
        sin(x) - 0.2 * |x|. Larger is better.

    Example:
        The global peak of this landscape is near x = +1.372, f = +0.706.
    """
    return np.sin(x) - 0.2 * abs(x)


class Individual:
    """One candidate solution. The name is only there to make the selection
    tables readable: every method below reports who was copied, by name.


    Example:
        The ten named individuals of SEED = 52 are Lesson 01's population;
        the baseline copies each once (diversity 6.665).
    """

    def __init__(self, name: str, gene_list: list[float]) -> None:
        """Build an individual and score it immediately.

        Args:
            name: a letter A-J, so the copy tables are readable.
            gene_list: a one-element list holding x.

        Example:
            The ten named individuals of SEED = 52 are Lesson 01's
            population, reused so the numbers continue.
        """
        self.name = name
        self.gene_list = gene_list
        self.fitness = float(objective(gene_list[0]))

    @property
    def gene(self) -> float:
        """The first gene as a float rather than a one-element list.

        Returns:
            The first (or only) gene as a float, so plots and tables do
            not keep writing gene_list[0].
        """
        return self.gene_list[0]

    def __repr__(self) -> str:
        """Compact snapshot used in every printed table of this lesson.

        Returns:
            A one-line snapshot such as 'x=+1.372 f=+0.706' or
            '(+1.372) f=+0.7060', used in every printed table.
        """
        return f"{self.name}  x={self.gene:+.3f}  f={self.fitness:+.3f}"


def create_population(size: int = POPULATION_SIZE) -> list[Individual]:
    """Ten random individuals, named A to J in creation order.

    random.seed() is called here rather than at module level so that every
    experiment in this lesson starts from the same population.

    Args:
        size: POPULATION_SIZE = 10.

    Returns:
        Individuals A-J drawn with SEED = 52, the same cloud as
        Lesson 01 step 2.

    Example:
        Diversity of this baseline is 6.665; that is the zero of
        every later spread column.
    """
    random.seed(SEED)
    genes = [random.uniform(GENE_MIN, GENE_MAX) for _ in range(size)]
    return [Individual(chr(ord("A") + i), [g]) for i, g in enumerate(genes)]


def selection_counts(population: list[Individual],
                     selected: list[Individual]) -> list[int]:
    """How many copies of each individual a selection produced.

    Counting by id() works because selection returns references to the very
    objects it was given - it copies no genes, it only points at individuals.

    Args:
        population: the original N individuals.
        selected: the N copies a selector returned.

    Returns:
        A list of N integers: how many times each original was pointed at.

    Example:
        Counted by id() because selection copies references, not genes;
        that is why 4 of 10 can go extinct without any x value changing.
    """
    counts = Counter(id(ind) for ind in selected)
    return [counts.get(id(ind), 0) for ind in population]


def count_extinct(copies: list[int]) -> int:
    """Individuals selected zero times. Their genes are gone for good.

    Args:
        copies: the list from selection_counts.

    Returns:
        How many individuals were selected zero times.

    Example:
        The baseline is 0 of 10; tournament k = 3 on this population
        matches Lesson 01's 4 of 10.
    """
    return sum(1 for c in copies if c == 0)


def gene_spread(individuals: list[Individual]) -> float:
    """Diversity, as the standard deviation of the genes.

    One number is crude, but it is the number that matters: when it reaches
    zero, crossover can no longer produce anything new and the search is over.

    Args:
        individuals: a population (or a selected generation).

    Returns:
        Population standard deviation of the genes.

    Example:
        Baseline 6.665; rank-with-elite-1 costs it down to 5.210.
    """
    return statistics.pstdev([ind.gene for ind in individuals])


def print_copies_table(population: list[Individual], copies: list[int]) -> None:
    """The table every step of this lesson prints: who survived, how often.

    Args:
        population, copies: who, and how many times.

    Returns:
        None. Prints the table every step of this lesson shows.
    """
    for ind, n in zip(population, copies):
        print(f"    {ind}   copies: {n}  {'#' * n}")


def select_nothing(population: list[Individual]) -> list[Individual]:
    """The zero-pressure baseline: everybody reproduces exactly once.

    This is not a useful operator - it is the yardstick. A selection method is
    interesting exactly to the extent that its numbers differ from these.

    Args:
        population: the current N individuals.

    Returns:
        A shallow copy: everybody reproduces exactly once.

    Example:
        0 extinct, diversity 6.665, 1 copy of the best — the zero
        of every later measurement.
    """
    return list(population)


population = create_population()

print("The population (sorted by fitness, best first):")
for ind in sorted(population, key=lambda i: -i.fitness):
    print("   ", ind)

best = max(population, key=lambda i: i.fitness)
worst = min(population, key=lambda i: i.fitness)
print(f"\nBest is {best.name} at f={best.fitness:+.3f}, "
      f"worst is {worst.name} at f={worst.fitness:+.3f}.")
print(f"Diversity (standard deviation of the genes): {gene_spread(population):.3f}")

print("\nThe baseline - selection that copies everybody once:")
copies = selection_counts(population, select_nothing(population))
print_copies_table(population, copies)
print(f"    extinct: {count_extinct(copies)} of {POPULATION_SIZE}"
      f"   |   diversity: {gene_spread(select_nothing(population)):.3f}"
      f"   |   copies of the best: {copies[population.index(best)]}")
print("Nothing happens, which is the point: no pressure, no loss.")

# One fact about this population decides the whole of step 2.
non_negative = [ind for ind in population if ind.fitness >= 0]
print(f"\nNote: {len(non_negative)} of {POPULATION_SIZE} individuals have "
      f"fitness >= 0.")
print("Every fitness here is negative, because f(x) = sin(x) - 0.2|x| is")
print("negative almost everywhere on [-10, +10]. A roulette wheel cannot be")
print("built from negative sector widths, so the first method in step 2 does")
print("not work on this population until we do something about that.")

fig, (ax_curve, ax_bars) = plt.subplots(1, 2, figsize=(11, 4))

x = np.linspace(GENE_MIN, GENE_MAX, 400)
ax_curve.plot(x, objective(x), color="tab:blue", linewidth=1.2)
ax_curve.scatter([i.gene for i in population], [i.fitness for i in population],
                 color="tab:red", zorder=3)
for ind in population:
    ax_curve.annotate(ind.name, (ind.gene, ind.fitness),
                      textcoords="offset points", xytext=(4, 5), fontsize=9)
ax_curve.axhline(0.0, color="grey", linewidth=0.8, linestyle="--")
ax_curve.set_title("The ten individuals on Lesson 01's landscape")
ax_curve.set_xlabel("x")
ax_curve.set_ylabel("f(x)")
ax_curve.grid(True, linestyle=":", alpha=0.5)

ordered = sorted(population, key=lambda i: i.fitness)
ax_bars.barh([i.name for i in ordered], [i.fitness for i in ordered],
             color="tab:red")
ax_bars.axvline(0.0, color="grey", linewidth=0.8, linestyle="--")
ax_bars.set_title("Raw fitness: every value is negative")
ax_bars.set_xlabel("f(x)")
ax_bars.grid(True, axis="x", linestyle=":", alpha=0.5)

FIGURES.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURES / "selection_pressure_01_the_population.png",
            dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved to {FIGURES}/selection_pressure_01_the_population.png")

