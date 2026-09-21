"""
Lesson 03 - Step 4: Elitism, the cheapest guarantee in the course
=================================================================
NEW IN THIS STEP: select_rank_with_elite().

Steps 2 and 3 both share a flaw that is easy to overlook: the best individual
is drawn at random like everybody else, so it is sometimes not drawn at all. A
generation can be strictly worse than its parent generation. Elitism fixes that
by copying the top few individuals into the new generation before any draw
happens. It is one line of code, it removes the regression completely, and it
costs diversity - all three of which this step measures.

CHANGES FROM selection_pressure_03_rank.py
Introduce them in this order:
    1. select_rank_with_elite()   the top individuals bypass the draw entirely
    2. the elite rows             what the guarantee is worth and what it costs
    3. the elitism figure         chance of losing the best, and diversity, per method

Run it:  python selection_pressure_04_elitism.py

select_rank_with_elite() copies the top individuals through before any draw.
One elite slot takes the loss of the best from 13.6% to 0.0%, raises pressure
(1.80 to 2.64 copies) and costs diversity (5.448 to 5.210). Elitism exists
because otherwise a generation can be strictly worse than its parent —
Rudolph's guarantee does not apply.
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
ELITE_SIZES = (1, 2)             # how many individuals bypass the draw
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
    return [value - lowest + floor for value in values]


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


def select_rank(population: list[Individual]) -> list[Individual]:
    """Rank selection: the sector widths come from the ORDER, not the values.

    Sort best first and give position i the width 1 - i/N. The widths never
    depend on how far apart the fitness values are, so an outlier cannot
    swallow the wheel - and no shift and no floor are needed at all.

    Args:
        population: the ten named individuals.

    Returns:
        N references drawn from order-only sector widths 1 - i/N.

    Example:
        The best/worst sector ratio is exactly N = 10, whatever the
        fitness scale; the best is still lost in 13.6% of draws.
    """
    ordered = sorted(population, key=lambda ind: -ind.fitness)
    size = len(population)
    weights = [1 - i / size for i in range(size)]
    total = sum(weights)
    selected: list[Individual] = []
    for _ in range(size):
        point = random.random() * total
        cumulative = 0.0
        for ind, weight in zip(ordered, weights):
            cumulative += weight
            if cumulative > point:
                selected.append(ind)
                break
    return selected


# --- NEW (1) select_rank_with_elite() -----------------------------------------
def select_rank_with_elite(population: list[Individual],
                           elite_size: int) -> list[Individual]:
    """Rank selection with the top `elite_size` individuals copied through.

    The elite slots are filled before any random draw happens, so the best
    individual cannot be lost. The remaining N - elite_size slots are ordinary
    rank selection, which is why the body repeats select_rank()'s loop.

    Args:
        population: the ten named individuals.
        elite_size: how many top individuals bypass the draw. 1 or 2.

    Returns:
        The elite, then N - elite_size ordinary rank draws.

    Example:
        elite_size = 1 takes loss of the best from 13.6% to 0.0%.
        Elitism exists because otherwise Rudolph's theorem does not
        apply: a generation can be worse than its parent.
    """
    ordered = sorted(population, key=lambda ind: -ind.fitness)
    size = len(population)
    weights = [1 - i / size for i in range(size)]
    total = sum(weights)
    selected: list[Individual] = list(ordered[:elite_size])
    for _ in range(size - elite_size):
        point = random.random() * total
        cumulative = 0.0
        for ind, weight in zip(ordered, weights):
            cumulative += weight
            if cumulative > point:
                selected.append(ind)
                break
    return selected
# ------------------------------------------------------------------------------


population = create_population()
best = max(population, key=lambda ind: ind.fitness)

# --- NEW (2) the elite rows ---------------------------------------------------
print("One rank draw with one elite slot:")
selected = select_rank_with_elite(population, ELITE_SIZES[0])
copies = selection_counts(population, selected)
print_copies_table(population, copies)
print(f"    the elite slot is {best.name}, the best individual, and it is "
      f"filled before any draw")

print(f"\nMeasured over {TRIALS} draws:")
rows = [
    pressure_report("no selection", select_nothing, population),
    pressure_report(f"proportional, floor {FITNESS_FLOOR_TIGHT}",
                    lambda p: select_proportional(p, FITNESS_FLOOR_TIGHT),
                    population),
    pressure_report(f"proportional, floor {FITNESS_FLOOR_GENEROUS}",
                    lambda p: select_proportional(p, FITNESS_FLOOR_GENEROUS),
                    population),
    pressure_report("rank", select_rank, population),
]
for elite_size in ELITE_SIZES:
    rows.append(pressure_report(
        f"rank + elite({elite_size})",
        lambda p, e=elite_size: select_rank_with_elite(p, e), population))
print_pressure_header()
for row in rows:
    print_pressure_row(row)

rank_row = rows[3]
elite_rows = rows[4:]
print(f"\nThe guarantee, checked over {TRIALS} draws. For each draw, take the")
print("best fitness present in the generation it produced, and keep the worst")
print("of those - the unluckiest generation the method can hand you:")
for label, selector in [("rank", select_rank)] + [
        (f"rank + elite({e})", lambda p, e=e: select_rank_with_elite(p, e))
        for e in ELITE_SIZES]:
    random.seed(SEED)
    unluckiest = min(max(ind.fitness for ind in selector(population))
                     for _ in range(TRIALS))
    print(f"    {label:20s} worst generation best: {unluckiest:+.3f}")
print(f"    the population's own best is {best.fitness:+.3f}")
print("With an elite slot the two numbers are identical, in every draw: the")
print("best fitness in the population can never go down again. Without one it")
print(f"can, and does, in {rank_row['lost']:.1f}% of draws.")

print("\nWhat it costs:")
for row in elite_rows:
    print(f"    {row['label']}: the best individual's expected copies rise "
          f"{rank_row['copies']:.2f} -> {row['copies']:.2f}, "
          f"extinctions {rank_row['extinct']:.2f} -> {row['extinct']:.2f}, "
          f"diversity {rank_row['spread']:.3f} -> {row['spread']:.3f}")
print("So elitism is not free and it is not only a safety net: it also raises")
print("the pressure, because the elite slots are slots the rest cannot win.")
print("On this population the diversity it costs is small, but the mechanism")
print("is the one that ends every run: Lesson 06 measures where it leads.")
# ------------------------------------------------------------------------------

# --- NEW (3) the elitism figure -----------------------------------------------
labels = [row["label"] for row in rows]
positions = np.arange(len(rows))

fig, (ax_lost, ax_div) = plt.subplots(1, 2, figsize=(12, 4.5))
ax_lost.bar(positions, [row["lost"] for row in rows], color="tab:red")
ax_lost.set_xticks(positions)
ax_lost.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
ax_lost.set_ylabel("% of draws")
ax_lost.set_title("draws that lose the best individual")
ax_lost.grid(True, axis="y", linestyle=":", alpha=0.5)

ax_div.bar(positions, [row["spread"] for row in rows], color="tab:blue")
ax_div.axhline(gene_spread(population), color="grey", linestyle="--",
               label=f"no selection ({gene_spread(population):.2f})")
ax_div.set_xticks(positions)
ax_div.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
ax_div.set_ylabel("standard deviation of the genes")
ax_div.set_title("diversity of the selected generation")
ax_div.grid(True, axis="y", linestyle=":", alpha=0.5)
ax_div.legend()

FIGURES.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURES / "selection_pressure_04_elitism.png",
            dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved to {FIGURES}/selection_pressure_04_elitism.png")
# ------------------------------------------------------------------------------

