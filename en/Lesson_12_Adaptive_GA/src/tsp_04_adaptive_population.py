"""
Lesson 12 - Step 4: A population that resizes itself
=====================================================
NEW IN THIS STEP: resize_population(), and a run() that can use it.

The second half of the book's adaptive algorithm changes the population itself.
While the run improves it drops the worst individual, so each generation gets
cheaper and the budget buys more of them. When the run stalls it injects fresh
random routes -- immigrants -- so there is new material to recombine.

Both mechanisms are on the same sensor as step 3, so the same question applies:
which branch actually fires? The script runs three regimes and counts. Resizing
turns out to bring the stalls back that step 3 had made disappear -- a smaller
population improves in jerks, and the jerks trip the sensor -- so here, unlike
in step 3, both branches of both rules are exercised.

CHANGES FROM tsp_03_adaptive_probabilities.py
Introduce them in this order:
    1. population_constants   the floor, the ceiling and how many immigrants a stall buys
    2. resize_population()    cull the worst while improving, immigrate while stalled
    3. run(resize)            a second switch, and the population size becomes a history

Run it:  python tsp_04_adaptive_population.py

Resizing alone: 35 stalls, 70 immigrants, 103 culls, population 68–120,
138 generations and 49,374. With the probabilities too: 41 stalls,
population down to 40, 155 generations, and 52,295 — worse on this seed
than resizing alone.
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

MIN_CROSSOVER_PROBABILITY = 0.1    # never stop recombining altogether
MIN_MUTATION_PROBABILITY = 0.05    # never stop inventing altogether
MAX_PROBABILITY = 1.0
STALL_FACTOR = 1.1                 # stalled: open the operators up, fast
IMPROVE_FACTOR = 0.99              # improving: close them down, slowly

# --- NEW (1) population_constants ---------------------------------------------
MIN_POPULATION_SIZE = 40           # below this the tournament has nothing to choose from
MAX_POPULATION_SIZE = 240          # above this a generation costs more than it returns
IMMIGRANTS_PER_STALL = 2           # fresh random routes injected when the run is stuck
# ------------------------------------------------------------------------------

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


def average(series: Sequence[float], period: int) -> float:
    """Mean of the `period` values before the last one, or nan if there are not enough.

    Args:
        series: population-mean lengths, newest last.
        period: how many past generations to average. 10 here.

    Returns:
        The trailing mean, or ``nan`` until there is enough history.
    """
    if len(series) < period + 1:
        return nan
    return sum(series[-period - 1:-1]) / period


def is_improving(series: Sequence[float], period: int = TREND_PERIOD, gap: float = TREND_GAP) -> bool:
    """True while the series is still falling faster than `gap` against its own past.

    Args:
        series: population-mean lengths, newest last.
        period: look-back. 10 generations here.
        gap: required fractional improvement. 0.001 here.

    Returns:
        True while improving. False is a stall.

    Example:
        Resizing brings stalls back: 35 with resize alone, 41 with both.
    """
    reference = average(series, period)
    if isnan(reference) or reference == 0:
        return True
    return series[-1] < reference * (1 - gap)


def adapt_probabilities(crossover: float, mutation: float, improving: bool) -> Tuple[float, float]:
    """Move both probabilities one notch, in the direction the sensor points.

    Opening up is faster than closing down (1.1 against 0.99) on purpose: a
    stall has to be escaped quickly, while an improving run should be left alone
    for as long as it keeps improving.

    Args:
        crossover, mutation: current probabilities.
        improving: the sensor from ``is_improving``.

    Returns:
        Clipped ``(crossover, mutation)``.
    """
    if improving:
        return (max(crossover * IMPROVE_FACTOR, MIN_CROSSOVER_PROBABILITY),
                max(mutation * IMPROVE_FACTOR, MIN_MUTATION_PROBABILITY))
    return (min(crossover * STALL_FACTOR, MAX_PROBABILITY),
            min(mutation * STALL_FACTOR, MAX_PROBABILITY))


# --- NEW (2) resize_population() ----------------------------------------------
def resize_population(points: Sequence[Tuple[int, int]], population: List[Scored],
                      improving: bool) -> Tuple[List[Scored], int]:
    """Shrink while it works, grow with fresh blood while it does not.

    Returns the new population and the evaluations it cost: culling is free,
    every immigrant has to be measured before it can compete.

    Args:
        points: city coordinates, used to score immigrants.
        population: scored routes of the current generation.
        improving: the sensor from ``is_improving``.

    Returns:
        ``(population, evaluations_spent)``. Culling costs 0.

    Example:
        Resize alone: 70 immigrants, 103 culls, population 68–120.
    """
    if improving:
        if len(population) > MIN_POPULATION_SIZE:
            population = population[:]
            population.remove(max(population, key=lambda scored: scored[1]))
        return population, 0
    added = 0
    population = population[:]
    while len(population) < MAX_POPULATION_SIZE and added < IMMIGRANTS_PER_STALL:
        route = random_route(len(points))
        population.append((route, route_length(points, route)))
        added += 1
    return population, added
# ------------------------------------------------------------------------------


def run(points: Sequence[Tuple[int, int]], seed: int, adaptive: bool = False,
        resize: bool = False) -> Dict[str, object]:
    """One run, stopped by the evaluation budget; two independent adaptive switches.

    Args:
        points: the 48 ATT cities.
        seed: 1 here.
        adaptive: if True, probabilities move each generation.
        resize: if True, population shrinks or grows each generation.

    Returns:
        Step 3's record plus size history, immigrant and cull counts.

    Example:
        Resize alone 49,374; both 52,295 — worse on this seed than resize
        alone.
    """
    random.seed(seed)
    evaluations = 0
    crossover_probability = CROSSOVER_PROBABILITY   # constants became state in step 3
    mutation_probability = MUTATION_PROBABILITY
    crossover_history = [crossover_probability]
    mutation_history = [mutation_probability]
    # --- NEW (3) run(resize) --------------------------------------------------
    size_history = [POPULATION_SIZE]                # the population is state as well now
    immigrants = 0                                  # how often the stalled branch bought reach
    culls = 0                                       # how often the improving branch bought speed
    # --------------------------------------------------------------------------
    population: List[Scored] = []
    for _ in range(POPULATION_SIZE):
        route = random_route(len(points))
        population.append((route, route_length(points, route)))
        evaluations += 1

    best = min(population, key=lambda scored: scored[1])
    best_history = [best[1]]
    mean_history = [sum(scored[1] for scored in population) / len(population)]
    spent_history = [evaluations]
    stall_trend: List[bool] = []       # one entry per generation: True while improving

    while evaluations + len(population) - 1 <= BUDGET:
        children = [best]                                  # elitism: carried, not re-evaluated
        while len(children) < len(population):
            parent_one, parent_two = tournament(population), tournament(population)
            if random.random() < crossover_probability:
                route = ordered_crossover(parent_one[0], parent_two[0])
            else:
                route = list(parent_one[0])
            if random.random() < mutation_probability:
                route = inversion_mutation(route)
            children.append((route, route_length(points, route)))
            evaluations += 1
        population = children
        best = min(population, key=lambda scored: scored[1])
        best_history.append(best[1])
        mean_history.append(sum(scored[1] for scored in population) / len(population))
        spent_history.append(evaluations)
        improving = is_improving(mean_history)
        stall_trend.append(improving)
        if adaptive:
            crossover_probability, mutation_probability = adapt_probabilities(
                crossover_probability, mutation_probability, improving)
        if resize:                                         # (3) the same sensor, a second actuator
            before = len(population)
            population, cost = resize_population(points, population, improving)
            evaluations += cost
            immigrants += cost
            culls += 1 if len(population) < before else 0
        crossover_history.append(crossover_probability)
        mutation_history.append(mutation_probability)
        size_history.append(len(population))

    return {
        "route": best[0],
        "length": best[1],
        "best_history": best_history,
        "mean_history": mean_history,
        "spent_history": spent_history,
        "stall_trend": stall_trend,
        "crossover_history": crossover_history,
        "mutation_history": mutation_history,
        "size_history": size_history,
        "immigrants": immigrants,
        "culls": culls,
        "evaluations": evaluations,
        "generations": len(best_history) - 1,
    }


def shade_stalls(axis: plt.Axes, stall_trend: Sequence[bool]) -> None:
    """Paint one red band over every generation the signal calls stalled.

    Args:
        axis: the history axes.
        stall_trend: one boolean per generation, True while improving.

    Returns:
        None.
    """
    for index, improving in enumerate(stall_trend):
        if not improving:
            axis.axvspan(index + 1, index + 2, color="red", alpha=0.12, linewidth=0)


points = load_points()
fixed = run(points, SEED)
resized = run(points, SEED, resize=True)
full = run(points, SEED, adaptive=True, resize=True)

print("Lesson 12 - Step 4: a population that resizes itself")
print(f"All three runs use seed {SEED} and the same budget of {BUDGET:,} evaluations.")
header = f"{'regime':<26}{'length':>10}{'evals':>9}{'gens':>7}{'pop min-max':>14}{'immigrants':>12}{'culls':>7}"
print(header)
for name, record in (("fixed", fixed), ("resize only", resized), ("resize + probabilities", full)):
    sizes = record["size_history"]
    print(f"{name:<26}{record['length']:>10,.0f}{record['evaluations']:>9,}"
          f"{record['generations']:>7}{f'{min(sizes)}-{max(sizes)}':>14}"
          f"{record['immigrants']:>12}{record['culls']:>7}")

print(f"Resize only: the sensor stalled {resized['stall_trend'].count(False)} times and bought "
      f"{resized['immigrants']} immigrants; culling the worst {resized['culls']} times")
print(f"  made generations cheaper, so the same budget bought "
      f"{resized['generations'] - fixed['generations']} more of them "
      f"({resized['generations']} against {fixed['generations']}).")
print(f"Resize + probabilities: {full['stall_trend'].count(False)} stalls against "
      f"{fixed['stall_trend'].count(False)} for the fixed run and 0 in step 3 --")
print(f"  a shrinking population improves in jerks, so the probability rule finally takes its "
      f"stalled branch too (crossover ranges {min(full['crossover_history']):.2f}-"
      f"{max(full['crossover_history']):.2f} instead of decaying straight down).")
print(f"On this seed the two halves together ({full['length']:,.0f}) do worse than resizing "
      f"alone ({resized['length']:,.0f}).")
print("Still one seed. Step 5 runs all of it twelve times.")

FIGURES.mkdir(exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].plot(resized["size_history"], label="resize only")
axes[0].plot(full["size_history"], label="resize + probabilities")
axes[0].axhline(POPULATION_SIZE, color="black", linestyle="--", alpha=0.5, label="fixed")
shade_stalls(axes[0], resized["stall_trend"])
axes[0].set(xlabel="generation", ylabel="individuals",
            title="Population size (red bands: stalls of the resize-only run)")
axes[0].legend()
for name, record in (("fixed", fixed), ("resize only", resized), ("resize + probabilities", full)):
    axes[1].plot(record["spent_history"], record["best_history"],
                 label=f"{name}: {record['length']:,.0f}")
axes[1].set(xlabel="evaluations spent", ylabel="best route length", title="Same seed, same budget")
axes[1].legend()
fig.tight_layout()
fig.savefig(FIGURES / "tsp_04_adaptive_population.png", dpi=160)
plt.close(fig)

