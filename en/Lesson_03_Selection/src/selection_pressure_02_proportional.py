"""
Lesson 03 - Step 2: Proportional selection, and the floor nobody mentions
=========================================================================
NEW IN THIS STEP: shift_to_positive(), select_proportional(), pressure_report().

The standard picture of selection is a roulette wheel whose sectors are as wide
as the individuals' fitness. On this population the picture does not fit: every
fitness is negative, and a sector cannot have a negative width. The usual
repair is to shift all the values up until the worst one is non-negative - and
that repair silently introduces a parameter, the floor, which turns out to
control the selection pressure more strongly than the fitness values do.

CHANGES FROM selection_pressure_01_the_population.py
Introduce them in this order:
    1. shift_to_positive()    a wheel needs non-negative sectors; the floor is a free choice
    2. select_proportional()  the wheel itself: one spin per slot of the new generation
    3. pressure_report()      the measurement every later step reuses: many spins, one row
    4. the two floors         spin the same wheel under floor 0.001 and floor 3.0
    5. the wheel figure       both wheels unrolled, so the sector widths can be compared

Run it:  python selection_pressure_02_proportional.py

shift_to_positive() lifts every fitness so the worst lands on a chosen floor —
a free parameter, not a fact about the problem. select_proportional() spins
that wheel once per slot. pressure_report() averages 2000 draws: the best
expects 2.35 copies at floor 0.001 but 1.30 at floor 3.0.
"""

import random
import statistics
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SEED = 52
POPULATION_SIZE = 10
GENE_MIN, GENE_MAX = -10.0, 10.0
FITNESS_FLOOR_TIGHT = 0.001      # the worst individual keeps almost no sector
FITNESS_FLOOR_GENEROUS = 3.0     # the worst individual keeps a wide sector
TRIALS = 2000                    # draws per measurement, for stable averages
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


# --- NEW (1) shift_to_positive() ----------------------------------------------
def shift_to_positive(values: list[float], floor: float) -> list[float]:
    """Move a list of fitness values so the worst one lands exactly on `floor`.

    A wheel cannot have negative sectors, and on this population every fitness
    is negative, so something like this is unavoidable. What is easy to miss is
    that `floor` is a free parameter: the problem does not supply it, and the
    rest of this step shows it decides the selection pressure.

    Args:
        values: raw fitnesses, here all negative.
        floor: where the worst value lands after the shift. A free
            parameter: 0.001 or 3.0 in this step.

    Returns:
        Shifted, non-negative sector widths.

    Example:
        The best expects 2.35 copies at floor 0.001 and 1.30 at
        floor 3.0 — same population, same wheel.
    """
    lowest = min(values)
    return [value - lowest + floor for value in values]  # floor is free; the problem does not supply it
# ------------------------------------------------------------------------------


# --- NEW (2) select_proportional() --------------------------------------------
def select_proportional(population: list[Individual],
                        floor: float) -> list[Individual]:
    """Fitness proportional selection: the roulette wheel.

    Each individual owns a sector as wide as its shifted fitness. One spin per
    slot of the new generation: throw a point at the wheel and walk the
    cumulative widths until you pass it.

    Args:
        population: the ten named individuals.
        floor: argument of shift_to_positive.

    Returns:
        N references, one spin of the wheel per slot.

    Example:
        The floor *is* the pressure; changing it from 0.001 to 3.0
        is a larger effect than anything the fitness values do.
    """
    weights = shift_to_positive([ind.fitness for ind in population], floor)
    total = sum(weights)
    selected: list[Individual] = []
    for _ in range(len(population)):
        point = random.random() * total
        cumulative = 0.0
        for ind, weight in zip(population, weights):
            cumulative += weight
            if cumulative > point:
                selected.append(ind)
                break
    return selected
# ------------------------------------------------------------------------------


# --- NEW (3) pressure_report() ------------------------------------------------
Selector = Callable[[list[Individual]], list[Individual]]


def pressure_report(label: str, selector: Selector,
                    population: list[Individual],
                    trials: int = TRIALS) -> dict:
    """Run one selector `trials` times and summarise the pressure it applies.

    Every method in this lesson is measured by this one function, always from
    the same seed, so the rows of the comparison tables are comparable by
    construction rather than by hope. The raw per-draw counts are kept because
    step 5 is about their spread, not their mean.

    Args:
        label: row name in the comparison table.
        selector: a function population -> selected.
        population: the fixed ten individuals.
        trials: 2000 draws from SEED = 52.

    Returns:
        Mean copies of the best, their sd, extinct count, % of draws
        that lost the best, mean selected fitness, gene spread, and
        the raw per-draw copy list (needed by the SUS step).

    Example:
        Rank sits at 1.80 copies of the best and loses it in 13.6%
        of draws; adding one elite takes those to 2.64 and 0.0%.
    """
    best = max(population, key=lambda ind: ind.fitness)
    random.seed(SEED)
    best_copies, extinctions, fitnesses, spreads = [], [], [], []
    for _ in range(trials):
        selected = selector(population)
        copies = selection_counts(population, selected)
        best_copies.append(copies[population.index(best)])
        extinctions.append(count_extinct(copies))
        fitnesses.append(statistics.mean(ind.fitness for ind in selected))
        spreads.append(gene_spread(selected))
    return {
        "label": label,
        "copies": statistics.mean(best_copies),
        "sd": statistics.pstdev(best_copies),
        "extinct": statistics.mean(extinctions),
        "lost": 100 * sum(1 for c in best_copies if c == 0) / trials,
        "fitness": statistics.mean(fitnesses),
        "spread": statistics.mean(spreads),
        "best_copies": best_copies,
    }


def print_pressure_header() -> None:
    """One header for every comparison table in steps 2 to 6.

    Returns:
        None. Prints the shared header of every comparison table
        in steps 2 to 6.
    """
    print(f"{'method':26s} {'copies':>6} {'sd':>5} {'extinct':>7} "
          f"{'lost':>6} {'mean f':>8} {'spread':>7}")
    print(f"{'':26s} {'of best':>6} {'':>5} {'of 10':>7} "
          f"{'best':>6} {'selected':>8} {'genes':>7}")


def print_pressure_row(row: dict) -> None:
    """Print one measured row of the pressure table.

    Args:
        row: a dict from pressure_report.

    Returns:
        None. Prints one measured line.
    """
    print(f"{row['label']:26s} {row['copies']:6.2f} {row['sd']:5.2f} "
          f"{row['extinct']:7.2f} {row['lost']:5.1f}% {row['fitness']:+8.3f} "
          f"{row['spread']:7.3f}")
# ------------------------------------------------------------------------------


# --- NEW (4) the two floors ---------------------------------------------------
population = create_population()
best = max(population, key=lambda ind: ind.fitness)

print("The wheel is built from SHIFTED fitness, and the shift needs a floor.")
for floor in (FITNESS_FLOOR_TIGHT, FITNESS_FLOOR_GENEROUS):
    weights = shift_to_positive([ind.fitness for ind in population], floor)
    shares = [100 * w / sum(weights) for w in weights]
    best_share = shares[population.index(best)]
    print(f"\n  floor = {floor}:   sector widths, in % of the wheel")
    for ind, share in zip(population, shares):
        print(f"    {ind}   sector: {share:5.2f}%  {'=' * round(share)}")
    print(f"    widest sector / narrowest sector = "
          f"{max(shares) / min(shares):.1f}")
    print(f"    the best individual, {best.name}, expects "
          f"{POPULATION_SIZE * best_share / 100:.2f} copies of itself")

print(f"\nOne spin per slot, with floor = {FITNESS_FLOOR_TIGHT}:")
selected = select_proportional(population, FITNESS_FLOOR_TIGHT)
copies = selection_counts(population, selected)
print_copies_table(population, copies)
print(f"    extinct in this single draw: {count_extinct(copies)} of "
      f"{POPULATION_SIZE}   |   diversity: {gene_spread(selected):.3f} "
      f"(was {gene_spread(population):.3f})")

print(f"\nThe same wheel, spun {TRIALS} times under each floor:")
rows = [
    pressure_report("no selection", select_nothing, population),
    pressure_report(f"proportional, floor {FITNESS_FLOOR_TIGHT}",
                    lambda p: select_proportional(p, FITNESS_FLOOR_TIGHT),
                    population),
    pressure_report(f"proportional, floor {FITNESS_FLOOR_GENEROUS}",
                    lambda p: select_proportional(p, FITNESS_FLOOR_GENEROUS),
                    population),
]
print_pressure_header()
for row in rows:
    print_pressure_row(row)

baseline, tight, generous = rows
print("\nSame method, same population, same seed: only the floor moved, and")
print(f"  - the best individual's expected copies went from "
      f"{tight['copies']:.2f} to {generous['copies']:.2f}")
print(f"  - the chance of losing the best individual entirely went from "
      f"{tight['lost']:.1f}% to {generous['lost']:.1f}%")
print(f"  - the diversity of the selected generation went from "
      f"{tight['spread']:.3f} to {generous['spread']:.3f}, "
      f"against {baseline['spread']:.3f} with no selection at all")
print("Nothing in the problem says which floor is right. That is the real")
print("weakness of proportional selection: the pressure it applies depends on")
print("an arbitrary offset as much as on the population. Step 3 removes the")
print("offset completely.")
# ------------------------------------------------------------------------------

# --- NEW (5) the wheel figure -------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(10, 4.5), sharex=True)
for ax, floor in zip(axes, (FITNESS_FLOOR_TIGHT, FITNESS_FLOOR_GENEROUS)):
    weights = shift_to_positive([ind.fitness for ind in population], floor)
    left = 0.0
    for ind, weight in zip(population, weights):
        width = 100 * weight / sum(weights)
        ax.barh([0], [width], left=[left], height=0.6, edgecolor="white")
        if width > 4:
            ax.text(left + width / 2, 0, ind.name,
                    ha="center", va="center", fontsize=9)
        left += width
    ax.set_yticks([])
    ax.set_xlim(0, 100)
    ax.set_title(f"the wheel unrolled, floor = {floor}")
axes[-1].set_xlabel("% of the wheel")

FIGURES.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURES / "selection_pressure_02_proportional.png",
            dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved to {FIGURES}/selection_pressure_02_proportional.png")
# ------------------------------------------------------------------------------

