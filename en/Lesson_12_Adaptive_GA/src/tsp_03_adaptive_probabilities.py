"""
Lesson 12 - Step 3: Probabilities that move
============================================
NEW IN THIS STEP: adapt_probabilities(), and a run() that can use it.

The sensor from step 2 says, every generation, whether the population is still
improving. This step wires it to the two probabilities: when the run stalls,
raise them -- more recombination, more mutation, more reach; while the run
improves, ease them down towards a floor and let selection refine what it has.
The rule is the book's, and it is deliberately blunt: multiply by 1.1 when
stalled, by 0.99 when improving, and clip.

Watch which of the two branches actually fires. The script counts them, and on
this seed the answer is not the one the story leads you to expect: the stalled
branch never runs at all, so what is advertised as adaptation behaves as a
one-way decay schedule. It still beats the fixed run here -- which says more
about the fixed values than about the rule.

The script runs both regimes on the same seed so the two curves can be put side
by side. One seed is an anecdote, not evidence -- Lesson 06 said why -- and
step 5 does the measurement.

CHANGES FROM tsp_02_the_stall.py
Introduce them in this order:
    1. adaptation_constants   the floors, the ceiling and the two multipliers
    2. adapt_probabilities()  the rule itself, on two plain floats
    3. run(adaptive)          the constants become per-run variables the rule moves

Run it:  python tsp_03_adaptive_probabilities.py

Same seed, same budget: adaptive 48,830 against fixed 57,076 (−14.4%).
The stalled branch fires 0 of 99 times — crossover walks 0.90 → 0.33 and
mutation 0.25 → 0.09 and never comes back up.
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

# --- NEW (1) adaptation_constants ---------------------------------------------
MIN_CROSSOVER_PROBABILITY = 0.1    # never stop recombining altogether
MIN_MUTATION_PROBABILITY = 0.05    # never stop inventing altogether
MAX_PROBABILITY = 1.0
STALL_FACTOR = 1.1                 # stalled: open the operators up, fast
IMPROVE_FACTOR = 0.99              # improving: close them down, slowly
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
        On this seed the stalled branch never fires: 0 of 99 generations.
    """
    reference = average(series, period)
    if isnan(reference) or reference == 0:
        return True
    return series[-1] < reference * (1 - gap)


# --- NEW (2) adapt_probabilities() --------------------------------------------
def adapt_probabilities(crossover: float, mutation: float, improving: bool) -> Tuple[float, float]:
    """Move both probabilities one notch, in the direction the sensor points.

    Opening up is faster than closing down (1.1 against 0.99) on purpose: a
    stall has to be escaped quickly, while an improving run should be left alone
    for as long as it keeps improving.

    Args:
        crossover, mutation: current probabilities.
        improving: the sensor from ``is_improving``.

    Returns:
        Clipped ``(crossover, mutation)``. Floors 0.10 and 0.05, ceiling 1.0.

    Example:
        The stalled branch fires 0 of 99 times; crossover walks 0.90 → 0.33
        and mutation 0.25 → 0.09.
    """
    if improving:
        return (max(crossover * IMPROVE_FACTOR, MIN_CROSSOVER_PROBABILITY),
                max(mutation * IMPROVE_FACTOR, MIN_MUTATION_PROBABILITY))
    return (min(crossover * STALL_FACTOR, MAX_PROBABILITY),
            min(mutation * STALL_FACTOR, MAX_PROBABILITY))
# ------------------------------------------------------------------------------


def run(points: Sequence[Tuple[int, int]], seed: int, adaptive: bool = False) -> Dict[str, object]:
    """One run, stopped by the evaluation budget; `adaptive` switches the rule on.

    Args:
        points: the 48 ATT cities.
        seed: 1 here.
        adaptive: if True, probabilities move each generation.

    Returns:
        Step 2's record plus probability histories.

    Example:
        Adaptive 48,830 against fixed 57,076 (−14.4%) on the same seed.
    """
    random.seed(seed)
    evaluations = 0
    # --- NEW (3) run(adaptive) ------------------------------------------------
    crossover_probability = CROSSOVER_PROBABILITY   # constants become state:
    mutation_probability = MUTATION_PROBABILITY     # both runs start where step 1 did
    crossover_history = [crossover_probability]
    mutation_history = [mutation_probability]
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
            if random.random() < crossover_probability:    # (3) the variable, not the constant
                route = ordered_crossover(parent_one[0], parent_two[0])
            else:
                route = list(parent_one[0])
            if random.random() < mutation_probability:     # (3) the variable, not the constant
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
        if adaptive:                                       # (3) the sensor moves the knobs
            crossover_probability, mutation_probability = adapt_probabilities(
                crossover_probability, mutation_probability, improving)
        crossover_history.append(crossover_probability)
        mutation_history.append(mutation_probability)

    return {
        "route": best[0],
        "length": best[1],
        "best_history": best_history,
        "mean_history": mean_history,
        "spent_history": spent_history,
        "stall_trend": stall_trend,
        "crossover_history": crossover_history,
        "mutation_history": mutation_history,
        "evaluations": evaluations,
        "generations": len(best_history) - 1,
    }


def shade_stalls(axis: plt.Axes, stall_trend: Sequence[bool]) -> None:
    """Paint one red band over every generation the signal calls stalled.

    Args:
        axis: the history axes.
        stall_trend: one boolean per generation, True while improving.

    Returns:
        None. On this seed the adaptive run paints no stall bands.
    """
    for index, improving in enumerate(stall_trend):
        if not improving:
            axis.axvspan(index + 1, index + 2, color="red", alpha=0.12, linewidth=0)


points = load_points()
fixed = run(points, SEED, adaptive=False)
adaptive = run(points, SEED, adaptive=True)

crossover_track = adaptive["crossover_history"]
mutation_track = adaptive["mutation_history"]
stalls = adaptive["stall_trend"].count(False)

print("Lesson 12 - Step 3: probabilities that move")
print(f"Both runs use seed {SEED} and the same budget of {BUDGET:,} evaluations.")
print(f"Fixed    : {fixed['length']:>9,.0f}   crossover {CROSSOVER_PROBABILITY:.2f} "
      f"mutation {MUTATION_PROBABILITY:.2f}   ({fixed['evaluations']:,} evaluations, "
      f"{fixed['generations']} generations)")
print(f"Adaptive : {adaptive['length']:>9,.0f}   crossover "
      f"{min(crossover_track):.2f}-{max(crossover_track):.2f} mutation "
      f"{min(mutation_track):.2f}-{max(mutation_track):.2f}   "
      f"({adaptive['evaluations']:,} evaluations, {adaptive['generations']} generations)")
print(f"Difference:{fixed['length'] - adaptive['length']:>9,.0f} shorter for the adaptive run "
      f"({(adaptive['length'] - fixed['length']) / fixed['length']:+.1%})")
print(f"Stall branch fired:  {stalls:>3} of {len(adaptive['stall_trend'])} generations of the adaptive run")
print(f"Improve branch fired:{len(adaptive['stall_trend']) - stalls:>3} of {len(adaptive['stall_trend'])}"
      f"   (the fixed run at the same seed stalled {fixed['stall_trend'].count(False)} times)")
print(f"Crossover went {CROSSOVER_PROBABILITY:.2f} -> {crossover_track[-1]:.2f}, "
      f"mutation {MUTATION_PROBABILITY:.2f} -> {mutation_track[-1]:.2f}, without ever going back up.")
print("So the rule did not adapt here: keeping the operators calm kept the mean improving,")
print("which kept the sensor happy, which kept the operators closing. It is a decay schedule.")
print("This is ONE seed. It shows the mechanism runs, not that it helps: step 5 measures.")

FIGURES.mkdir(exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].plot(crossover_track, label="crossover probability")
axes[0].plot(mutation_track, label="mutation probability")
shade_stalls(axes[0], adaptive["stall_trend"])
axes[0].axhline(CROSSOVER_PROBABILITY, color="tab:blue", linestyle="--", alpha=0.4)
axes[0].axhline(MUTATION_PROBABILITY, color="tab:orange", linestyle="--", alpha=0.4)
axes[0].set(xlabel="generation", ylabel="probability", ylim=(0, 1.05),
            title="Where the knobs went (dashed = the fixed values;\nno red band because no generation stalled)")
axes[0].legend()
axes[1].plot(fixed["spent_history"], fixed["best_history"], label=f"fixed: {fixed['length']:,.0f}")
axes[1].plot(adaptive["spent_history"], adaptive["best_history"],
             label=f"adaptive: {adaptive['length']:,.0f}")
axes[1].set(xlabel="evaluations spent", ylabel="best route length",
            title=f"Same seed, same budget")
axes[1].legend()
fig.tight_layout()
fig.savefig(FIGURES / "tsp_03_adaptive_probabilities.png", dpi=160)
plt.close(fig)

