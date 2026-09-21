"""
Lesson 12 - Step 2: Finding the stall
======================================
NEW IN THIS STEP: average(), is_improving(), the stall trend, shade_stalls().

Step 1 spent its whole budget at one setting. This step asks where that budget
went. The algorithm cannot see how far it is from the optimum -- it has no
optimum to compare against -- but it can see its own recent history, and that
is enough to tell a generation that is still making progress from one that is
not. The signal is the book's: compare the population mean of this generation
with the mean of the previous ten. If it has not improved by at least a tenth
of a percent, the generation is stalled.

The signal is deliberately crude and it is the only sensor the adaptive
algorithm of steps 3 and 4 will have.

CHANGES FROM tsp_01_fixed_parameters.py
Introduce them in this order:
    1. average()        the trailing mean of a series, over a window
    2. is_improving()   the stall signal, one boolean per generation
    3. stall_trend      run() records that boolean alongside its other histories
    4. shade_stalls()   paint the stalled generations onto a plot

Run it:  python tsp_02_the_stall.py

The signal calls 14 of 99 generations stalled, the first at generation 74,
and 14% of the budget is spent inside them; the first half of the budget
buys 85% of the run's total improvement.
"""
from math import dist, isnan, nan
from pathlib import Path
import random
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt

SEED = 1
DATA = Path(__file__).with_name("att48_xy.txt")
FIGURES = Path(__file__).resolve().parent.parent / "figures"

BUDGET = 12_000            # route evaluations, the only currency that is fair
POPULATION_SIZE = 120
CROSSOVER_PROBABILITY = 0.9
MUTATION_PROBABILITY = 0.25
TOURNAMENT_SIZE = 3
TREND_PERIOD = 10          # how many past generations the signal looks back over
TREND_GAP = 0.001          # the improvement it demands: one tenth of a percent

Route = List[int]
Scored = Tuple[Route, float]


def load_points(path: Path = DATA) -> List[Tuple[int, int]]:
    """The 48 capitals, as integer coordinates, one city per line.

    Args:
        path: ``att48_xy.txt`` beside this script, restated locally.

    Returns:
        48 ``(x, y)`` integer pairs.
    """
    return [tuple(map(int, line.split())) for line in path.read_text().splitlines() if line.strip()]


def route_length(points: Sequence[Tuple[int, int]], route: Sequence[int]) -> float:
    """Closed-tour length. One call to this function is one evaluation.

    Args:
        points: city coordinates.
        route: a permutation of city indices.

    Returns:
        Euclidean length of the closed tour. The 12,000-evaluation budget
        counts these calls.
    """
    return sum(dist(points[a], points[b]) for a, b in zip(route, tuple(route[1:]) + tuple(route[:1])))


def random_route(size: int) -> Route:
    """A random permutation of ``0..size-1``.

    Args:
        size: number of cities, 48 here.

    Returns:
        A legal tour.
    """
    route = list(range(size))
    random.shuffle(route)
    return route


def nearest_neighbour(points: Sequence[Tuple[int, int]], start: int = 0) -> Route:
    """The cheap constructive baseline from Lesson 10; the GA has to answer to it.

    Args:
        points: city coordinates.
        start: city index to leave first. 0 here.

    Returns:
        A legal tour of length 40,526 on this instance.
    """
    route, unseen = [start], set(range(len(points))) - {start}
    while unseen:
        current = route[-1]
        closest = min(unseen, key=lambda city: dist(points[current], points[city]))
        route.append(closest)
        unseen.remove(closest)
    return route


def ordered_crossover(first: Sequence[int], second: Sequence[int]) -> Route:
    """Keeps a slice of one parent and fills the holes in the other parent's order.

    Args:
        first: the parent that donates the kept segment.
        second: the parent that donates the remaining cities, in order.

    Returns:
        A permutation.
    """
    left, right = sorted(random.sample(range(len(first)), 2))
    child = [-1] * len(first)
    child[left:right] = first[left:right]
    remaining = [city for city in second if city not in child]
    holes = list(range(right, len(first))) + list(range(left))
    for index, city in zip(holes, remaining):
        child[index] = city
    return child


def inversion_mutation(source: Sequence[int]) -> Route:
    """Reverses a segment: a legal permutation stays a legal permutation.

    Args:
        source: a legal route.

    Returns:
        A new permutation.
    """
    route = list(source)
    left, right = sorted(random.sample(range(len(route)), 2))
    route[left:right] = reversed(route[left:right])
    return route


def tournament(population: List[Scored]) -> Scored:
    """Prefer the shorter of three scored routes.

    Args:
        population: ``(route, length)`` pairs.

    Returns:
        The shortest member of a uniform sample of size 3.
    """
    return min(random.sample(population, TOURNAMENT_SIZE), key=lambda scored: scored[1])


# --- NEW (1) average() --------------------------------------------------------
def average(series: Sequence[float], period: int) -> float:
    """Mean of the `period` values before the last one, or nan if there are not enough.

    The last value is excluded on purpose: the caller compares it against this.

    Args:
        series: population-mean lengths, newest last.
        period: how many past generations to average. 10 here.

    Returns:
        The trailing mean, or ``nan`` until there is enough history.
    """
    if len(series) < period + 1:
        return nan
    return sum(series[-period - 1:-1]) / period
# ------------------------------------------------------------------------------


# --- NEW (2) is_improving() ---------------------------------------------------
def is_improving(series: Sequence[float], period: int = TREND_PERIOD, gap: float = TREND_GAP) -> bool:
    """True while the series is still falling faster than `gap` against its own past.

    Route length is minimised, so improvement means the newest value is BELOW the
    trailing average. Before there is enough history the answer is True: a run
    that has just started is not stalled, it is young.

    Args:
        series: population-mean lengths, newest last.
        period: look-back. 10 generations here.
        gap: required fractional improvement. 0.001 here.

    Returns:
        True while improving. False is a stall.

    Example:
        14 of 99 generations are stalled; the first stall is generation 74.
    """
    reference = average(series, period)
    if isnan(reference) or reference == 0:
        return True
    return series[-1] < reference * (1 - gap)
# ------------------------------------------------------------------------------


def run(points: Sequence[Tuple[int, int]], seed: int) -> Dict[str, object]:
    """One run of the fixed-parameter GA, stopped by the evaluation budget.

    A generation is only started if the budget can pay for all of it, so no run
    is cut off half way and every run reports the evaluations it really spent.

    Args:
        points: the 48 ATT cities.
        seed: 1 here.

    Returns:
        Step 1's record plus ``stall_trend``, one boolean per generation.

    Example:
        14 of 99 generations stall; the first half of the budget buys 85%
        of the total improvement.
    """
    random.seed(seed)
    evaluations = 0
    population: List[Scored] = []
    for _ in range(POPULATION_SIZE):
        route = random_route(len(points))
        population.append((route, route_length(points, route)))
        evaluations += 1

    best = min(population, key=lambda scored: scored[1])
    best_history = [best[1]]
    mean_history = [sum(scored[1] for scored in population) / len(population)]
    spent_history = [evaluations]
    # --- NEW (3) stall_trend --------------------------------------------------
    stall_trend: List[bool] = []       # one entry per generation: True while improving
    # --------------------------------------------------------------------------

    while evaluations + len(population) - 1 <= BUDGET:
        children = [best]                                  # elitism: carried, not re-evaluated
        while len(children) < len(population):
            parent_one, parent_two = tournament(population), tournament(population)
            if random.random() < CROSSOVER_PROBABILITY:
                route = ordered_crossover(parent_one[0], parent_two[0])
            else:
                route = list(parent_one[0])
            if random.random() < MUTATION_PROBABILITY:
                route = inversion_mutation(route)
            children.append((route, route_length(points, route)))
            evaluations += 1
        population = children
        best = min(population, key=lambda scored: scored[1])
        best_history.append(best[1])
        mean_history.append(sum(scored[1] for scored in population) / len(population))
        spent_history.append(evaluations)
        stall_trend.append(is_improving(mean_history))     # (3) read the sensor every generation

    return {
        "route": best[0],
        "length": best[1],
        "best_history": best_history,
        "mean_history": mean_history,
        "spent_history": spent_history,
        "stall_trend": stall_trend,
        "evaluations": evaluations,
        "generations": len(best_history) - 1,
    }


# --- NEW (4) shade_stalls() ---------------------------------------------------
def shade_stalls(axis: plt.Axes, stall_trend: Sequence[bool]) -> None:
    """Paint one red band over every generation the signal calls stalled.

    Args:
        axis: the history axes.
        stall_trend: one boolean per generation, True while improving.

    Returns:
        None. 14 red bands appear on this seed.
    """
    for index, improving in enumerate(stall_trend):
        if not improving:
            axis.axvspan(index + 1, index + 2, color="red", alpha=0.12, linewidth=0)
# ------------------------------------------------------------------------------


points = load_points()
baseline = route_length(points, nearest_neighbour(points))
record = run(points, SEED)

trend = record["stall_trend"]
best_history = record["best_history"]
spent_history = record["spent_history"]
stalled = [index for index, improving in enumerate(trend) if not improving]
warmup = min(TREND_PERIOD, len(trend))          # generations before the signal has history
per_generation = [spent_history[i + 1] - spent_history[i] for i in range(len(trend))]
stalled_evaluations = sum(per_generation[i] for i in stalled)

half = record["evaluations"] / 2
crossing = next(i for i, spent in enumerate(spent_history) if spent >= half)
total_gain = best_history[0] - best_history[-1]
first_half_gain = best_history[0] - best_history[crossing]

print("Lesson 12 - Step 2: finding the stall")
print(f"Best GA route (seed {SEED}):    {record['length']:,.0f}  ({record['evaluations']:,} evaluations)")
print(f"Signal warm-up:             first {warmup} generations answer True by construction")
print(f"Stalled generations:        {len(stalled)} of {len(trend)}")
print(f"First stall:                generation {stalled[0] + 1}" if stalled else "First stall: none")
print(f"Evaluations spent stalled:  {stalled_evaluations:,} of {record['evaluations']:,} "
      f"({stalled_evaluations / record['evaluations']:.0%})")
print(f"Total improvement:          {total_gain:,.0f} of route length")
print(f"Bought by the first half:   {first_half_gain:,.0f} ({first_half_gain / total_gain:.0%} of it) "
      f"in {spent_history[crossing]:,} evaluations")
print(f"Bought by the second half:  {total_gain - first_half_gain:,.0f} "
      f"({1 - first_half_gain / total_gain:.0%})")
print("The setting that bought the first half is still in force for the second.")

FIGURES.mkdir(exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].plot(record["mean_history"], label="population mean")
axes[0].plot(best_history, label="best of run")
shade_stalls(axes[0], trend)
axes[0].axhline(baseline, color="black", linestyle="--", label="nearest neighbour")
axes[0].set(xlabel="generation", ylabel="route length",
            title=f"Stalled generations shaded ({len(stalled)} of {len(trend)})")
axes[0].legend()
axes[1].plot(spent_history, best_history)
axes[1].axvline(spent_history[crossing], color="grey", linestyle=":")
axes[1].annotate(f"half the budget\nhas bought {first_half_gain / total_gain:.0%}\nof the improvement",
                 xy=(spent_history[crossing], best_history[crossing]),
                 xytext=(0.45, 0.65), textcoords="axes fraction",
                 arrowprops=dict(arrowstyle="->", color="grey"))
axes[1].set(xlabel="evaluations spent", ylabel="best route length", title="What the budget buys")
fig.tight_layout()
fig.savefig(FIGURES / "tsp_02_the_stall.png", dpi=160)
plt.close(fig)

