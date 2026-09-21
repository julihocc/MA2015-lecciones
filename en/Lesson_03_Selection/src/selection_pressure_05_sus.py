"""
Lesson 03 - Step 5: Stochastic universal sampling, or the same wheel spun once
=============================================================================
NEW IN THIS STEP: select_stochastic_universal_sampling().

Proportional selection spins the wheel N times, so luck accumulates: an
individual owning a quarter of the wheel can still end up with nothing, and
another can win four slots in a row. SUS keeps exactly the same wheel and the
same expected number of copies, but takes all N samples in one go, with evenly
spaced pointers. The expectation does not move. The spread collapses.

CHANGES FROM selection_pressure_04_elitism.py
Introduce them in this order:
    1. select_stochastic_universal_sampling()   one random draw, N evenly spaced pointers
    2. the SUS rows                             same expectation as the wheel, far less spread
    3. the spread figure                        how many copies of the best, wheel against SUS

Run it:  python selection_pressure_05_sus.py

select_stochastic_universal_sampling() uses the same wheel as proportional
selection but spaces N pointers evenly, so sampling noise collapses. It does
not fix the floor: 2.35 vs 1.31 copies, the same split step 2 found.
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


# --- NEW (1) select_stochastic_universal_sampling() ---------------------------
def select_stochastic_universal_sampling(population: list[Individual],
                                         floor: float) -> list[Individual]:
    """SUS: one random draw, then N pointers spaced evenly around the wheel.

    The wheel is exactly select_proportional()'s wheel. The difference is that
    the pointers cannot bunch together: a sector wider than the spacing is
    guaranteed at least one copy, and nobody can get lucky twice in a row.

    Args:
        population: the ten named individuals.
        floor: the same free parameter as proportional selection.

    Returns:
        N references from one random start and N equally spaced pointers.

    Example:
        Same wheel, same expectation, spread collapses. It does not
        fix the floor: 2.35 vs 1.31 copies, the split step 2 found.
    """
    ordered = sorted(population, key=lambda ind: -ind.fitness)
    weights = shift_to_positive([ind.fitness for ind in ordered], floor)
    size = len(population)
    spacing = sum(weights) / size
    start = random.uniform(0, spacing)
    selected: list[Individual] = []
    index = 0
    cumulative = weights[0]
    for slot in range(size):
        pointer = start + slot * spacing
        while cumulative < pointer and index < size - 1:
            index += 1
            cumulative += weights[index]
        selected.append(ordered[index])
    return selected
# ------------------------------------------------------------------------------


population = create_population()
best = max(population, key=lambda ind: ind.fitness)

# --- NEW (2) the SUS rows -----------------------------------------------------
print(f"One SUS draw, floor = {FITNESS_FLOOR_TIGHT}:")
selected = select_stochastic_universal_sampling(population,
                                               FITNESS_FLOOR_TIGHT)
copies = selection_counts(population, selected)
print_copies_table(population, copies)
print(f"    extinct in this single draw: {count_extinct(copies)} of "
      f"{POPULATION_SIZE}   |   diversity: {gene_spread(selected):.3f} "
      f"(was {gene_spread(population):.3f})")

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
    pressure_report(f"rank + elite({ELITE_SIZES[0]})",
                    lambda p: select_rank_with_elite(p, ELITE_SIZES[0]),
                    population),
    pressure_report(f"rank + elite({ELITE_SIZES[1]})",
                    lambda p: select_rank_with_elite(p, ELITE_SIZES[1]),
                    population),
    pressure_report(f"SUS, floor {FITNESS_FLOOR_TIGHT}",
                    lambda p: select_stochastic_universal_sampling(
                        p, FITNESS_FLOOR_TIGHT), population),
    pressure_report(f"SUS, floor {FITNESS_FLOOR_GENEROUS}",
                    lambda p: select_stochastic_universal_sampling(
                        p, FITNESS_FLOOR_GENEROUS), population),
]
print_pressure_header()
for row in rows:
    print_pressure_row(row)

by_label = {row["label"]: row for row in rows}
wheel = by_label[f"proportional, floor {FITNESS_FLOOR_TIGHT}"]
sus = by_label[f"SUS, floor {FITNESS_FLOOR_TIGHT}"]

print(f"\nThe same wheel, floor {FITNESS_FLOOR_TIGHT}, sampled two ways:")
print(f"    expected copies of the best   {wheel['copies']:.2f} vs "
      f"{sus['copies']:.2f}   (they differ by "
      f"{abs(wheel['copies'] - sus['copies']):.2f})")
print(f"    spread of that count (sd)     {wheel['sd']:.2f} vs {sus['sd']:.2f}"
      f"   (SUS keeps {100 * sus['sd'] / wheel['sd']:.0f}% of it)")
print(f"    draws that lose the best      {wheel['lost']:.1f}% vs "
      f"{sus['lost']:.1f}%")
print(f"    extinct per draw              {wheel['extinct']:.2f} vs "
      f"{sus['extinct']:.2f}")
print("That is the whole claim for SUS, and it is worth making precisely: it")
print("does not select better individuals, it selects the SAME individuals")
print("more reliably. Sampling noise is not selection pressure, and removing")
print("it is free.")

sus_generous = by_label[f"SUS, floor {FITNESS_FLOOR_GENEROUS}"]
print(f"\nWhat SUS does not fix: the floor. Expected copies of the best are")
print(f"{sus['copies']:.2f} at floor {FITNESS_FLOOR_TIGHT} and "
      f"{sus_generous['copies']:.2f} at floor {FITNESS_FLOOR_GENEROUS}, the "
      f"same split step 2 found.")
print("SUS repairs how the wheel is sampled, not how the wheel is built.")
# ------------------------------------------------------------------------------

# --- NEW (3) the spread figure ------------------------------------------------
fig, (ax_wheel, ax_hist) = plt.subplots(1, 2, figsize=(12, 4.5))

weights = shift_to_positive(
    [ind.fitness for ind in sorted(population, key=lambda i: -i.fitness)],
    FITNESS_FLOOR_TIGHT)
ordered = sorted(population, key=lambda ind: -ind.fitness)
left = 0.0
for ind, weight in zip(ordered, weights):
    width = 100 * weight / sum(weights)
    ax_wheel.barh([0], [width], left=[left], height=0.6, edgecolor="white")
    if width > 4:
        ax_wheel.text(left + width / 2, 0, ind.name, ha="center",
                      va="center", fontsize=9)
    left += width
spacing = 100 / POPULATION_SIZE
random.seed(SEED)
start = random.uniform(0, spacing)
for slot in range(POPULATION_SIZE):
    ax_wheel.axvline(start + slot * spacing, color="black", linewidth=2)
ax_wheel.set_yticks([])
ax_wheel.set_xlim(0, 100)
ax_wheel.set_xlabel("% of the wheel")
ax_wheel.set_title(f"SUS: one draw fixes all {POPULATION_SIZE} pointers")

span = range(0, max(max(wheel["best_copies"]), max(sus["best_copies"])) + 1)
width = 0.4
ax_hist.bar([c - width / 2 for c in span],
            [100 * wheel["best_copies"].count(c) / TRIALS for c in span],
            width, label=f"proportional ({wheel['sd']:.2f} sd)")
ax_hist.bar([c + width / 2 for c in span],
            [100 * sus["best_copies"].count(c) / TRIALS for c in span],
            width, label=f"SUS ({sus['sd']:.2f} sd)")
ax_hist.axvline(wheel["copies"], color="grey", linestyle="--",
                label=f"expected {wheel['copies']:.2f} copies")
ax_hist.set_xticks(list(span))
ax_hist.set_xlabel(f"copies of the best individual, {best.name}, in one draw")
ax_hist.set_ylabel(f"% of {TRIALS} draws")
ax_hist.set_title("same expectation, different spread")
ax_hist.grid(True, axis="y", linestyle=":", alpha=0.5)
ax_hist.legend()

FIGURES.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURES / "selection_pressure_05_sus.png",
            dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved to {FIGURES}/selection_pressure_05_sus.png")
# ------------------------------------------------------------------------------

