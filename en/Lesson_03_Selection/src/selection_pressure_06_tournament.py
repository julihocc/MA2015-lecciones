"""
Lesson 03 - Step 6: Tournament selection, and all five methods on one scale
==========================================================================
NEW IN THIS STEP: select_tournament(), and the comparison the lesson was for.

Tournament selection needs no wheel, no sort, no sum and no floor: pick a few
individuals at random, keep the fittest, repeat. This is the method Lesson 01
used, and with the same seed this file reproduces Lesson 01's draw exactly. Its
real virtue is the one thing none of the other four offer: a single integer that
sets the selection pressure, from none at all to almost total.

With that dial in hand, the five methods can finally be put on one scale - which
is the point of the lesson, and the last table of this step.

CHANGES FROM selection_pressure_05_sus.py
Introduce them in this order:
    1. select_tournament()    no wheel, no sort, no floor - only local comparisons
    2. the draw from Lesson 01 the same seed and the same result as Lesson 01, step 3
    3. the pressure dial      sweep the tournament size and watch the pressure rise
    4. the whole comparison   every method of steps 2 to 6 in one table, sorted by pressure
    5. the dial figure        the dial, and pressure against diversity for every method

Run it:  python selection_pressure_06_tournament.py

select_tournament() needs no wheel: k random candidates, fittest wins. With k
= 3 and SEED = 52 it reproduces Lesson 01 step 3. One integer sets the
pressure: k = 3 gives 2.74 copies of the best, k = 5 gives 4.13, k = 10 gives
6.52. Across all 13 configurations, pressure vs quality correlates +0.95 and
pressure vs diversity -0.92.
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
TOURNAMENT_SIZES = (1, 2, 3, 5, 10)   # the pressure dial, from none to almost total
LESSON_01_TOURNAMENT_SIZE = 3         # the value Lesson 01 used
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


# --- NEW (1) select_tournament() ----------------------------------------------
def select_tournament(population: list[Individual],
                      size: int) -> list[Individual]:
    """Tournament selection: `size` random candidates per slot, fittest wins.

    No sorting, no sums, no shift, no floor - only local comparisons, which is
    why Lesson 01 could use it before any of this machinery existed and why
    most libraries default to it. `size` is the pressure dial: at size 1 it is
    pure chance, at size N the best individual wins almost every slot.

    With replacement, population size N, and rank r=1 best, one slot selects
    rank r with probability ((N-r+1)/N)^size - ((N-r)/N)^size. The probabilities
    telescope to one; multiplying by N gives expected copies.

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
    selected: list[Individual] = []
    for _ in range(len(population)):
        candidates = [random.choice(population) for _ in range(size)]
        selected.append(max(candidates, key=lambda ind: ind.fitness))
    return selected
# ------------------------------------------------------------------------------


population = create_population()
best = max(population, key=lambda ind: ind.fitness)

# --- NEW (2) the draw from Lesson 01 ------------------------------------------
print(f"One tournament draw, size {LESSON_01_TOURNAMENT_SIZE}, from SEED = "
      f"{SEED} - the draw Lesson 01 showed:")
selected = select_tournament(population, LESSON_01_TOURNAMENT_SIZE)
copies = selection_counts(population, selected)
print_copies_table(population, copies)
winner = population[copies.index(max(copies))]
print(f"    {count_extinct(copies)} of {POPULATION_SIZE} individuals went "
      f"extinct, and {winner.name} (x={winner.gene:+.3f}) took "
      f"{max(copies)} of the {POPULATION_SIZE} slots.")
print("    Those are the same numbers Lesson 01's step 3 printed, because it")
print("    is the same population, the same seed and the same operator. This")
print("    lesson is measuring what that lesson only showed once.")
# ------------------------------------------------------------------------------

# --- NEW (3) the pressure dial ------------------------------------------------
print(f"\nThe dial: tournament size against pressure, over {TRIALS} draws each:")
dial = [pressure_report(f"tournament k={k}",
                        lambda p, k=k: select_tournament(p, k), population)
        for k in TOURNAMENT_SIZES]
print_pressure_header()
for row in dial:
    print_pressure_row(row)

print(f"\nk = {TOURNAMENT_SIZES[0]} is not selection at all: every slot is a")
print(f"single random pick, so the best individual expects "
      f"{dial[0]['copies']:.2f} copies, the same as no selection, and it is")
print(f"lost in {dial[0]['lost']:.1f}% of draws. At k = "
      f"{TOURNAMENT_SIZES[-1]} the best individual takes "
      f"{dial[-1]['copies']:.2f} of the {POPULATION_SIZE} slots and")
print(f"{dial[-1]['extinct']:.2f} individuals of {POPULATION_SIZE} die per "
      f"draw. One integer spans the whole range.")
# ------------------------------------------------------------------------------

# --- NEW (4) the whole comparison ---------------------------------------------
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
] + dial

# How wide is the band the floor alone opens up, in copies of the best?
# Computed from the two proportional rows above, before the table is sorted,
# so the sentence printed at the end cannot drift from the table.
_proporcional = [row for row in rows if row["label"].startswith("proportional")]
by_floor_range = abs(_proporcional[0]["copies"] - _proporcional[1]["copies"])

print(f"\nAll five methods, {TRIALS} draws each, sorted by pressure:")
rows.sort(key=lambda row: row["copies"])
print_pressure_header()
for row in rows:
    print_pressure_row(row)

pressure = [row["copies"] for row in rows]
diversity = [row["spread"] for row in rows]
quality = [row["fitness"] for row in rows]
print(f"\nAcross those {len(rows)} configurations:")
print(f"    pressure against quality   (mean fitness): "
      f"correlation {statistics.correlation(pressure, quality):+.2f}")
print(f"    pressure against diversity (gene spread):  "
      f"correlation {statistics.correlation(pressure, diversity):+.2f}")
print("Those two numbers are the lesson. There is no method here that buys")
print("fitness without paying in diversity, and no method that keeps")
print("diversity without giving up fitness. What the five methods differ in")
print("is not the trade - it is how much of it you get by accident:")
print(f"    proportional trades at a rate set by the floor "
      f"({by_floor_range:.2f} copies of the best, for the same population)")
print("    rank trades at a rate set by the population size, and only that")
print("    elitism removes the risk of going backwards, and adds pressure")
print("    SUS removes the sampling noise without touching the rate")
print("    tournament puts the rate in one integer you choose on purpose")
# ------------------------------------------------------------------------------

# --- NEW (5) the dial figure --------------------------------------------------
fig, (ax_dial, ax_scatter) = plt.subplots(1, 2, figsize=(12, 4.5))

ax_dial.plot(TOURNAMENT_SIZES, [row["copies"] for row in dial], "o-",
             color="tab:blue", label="copies of the best individual")
ax_dial.plot(TOURNAMENT_SIZES, [row["extinct"] for row in dial], "s--",
             color="tab:red", label="individuals that go extinct")
ax_dial.set_xlabel("tournament size k")
ax_dial.set_ylabel(f"out of {POPULATION_SIZE}")
ax_dial.set_title("the dial: one integer\nsets selection pressure", fontsize=10)
ax_dial.grid(True, linestyle=":", alpha=0.5)
ax_dial.legend()

ax_scatter.scatter(pressure, diversity, color="tab:purple", zorder=3)
chart_labels = {
    "no selection": ("no selection", (8, 5)),
    "tournament k=5": ("tournament 5", (8, 8)),
    "tournament k=10": ("tournament 10", (8, 8)),
}

# Label three representative settings; the table above lists every method.
for row in rows:
    if row["label"] not in chart_labels:
        continue
    short_label, offset = chart_labels[row["label"]]
    ax_scatter.annotate(
        short_label,
        (row["copies"], row["spread"]),
        textcoords="offset points",
        xytext=offset,
        ha="left",
        fontsize=6,
    )
ax_scatter.axhline(gene_spread(population), color="grey", linestyle="--",
                   label=f"no selection ({gene_spread(population):.2f})")
ax_scatter.set_xlabel("pressure: expected copies of the best individual")
ax_scatter.set_ylabel("diversity: standard deviation of the genes")
ax_scatter.set_title("pressure and diversity\nmove together", fontsize=10)
ax_scatter.grid(True, linestyle=":", alpha=0.5)
ax_scatter.legend()

FIGURES.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURES / "selection_pressure_06_tournament.png",
            dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved to {FIGURES}/selection_pressure_06_tournament.png")
# ------------------------------------------------------------------------------

