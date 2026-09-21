"""
Lesson 12 - Step 5: Twelve runs of each
========================================
NEW IN THIS STEP: RUNS, measure(), paired_summary(), and a table instead of a run.

Steps 3 and 4 each showed one seed, and one seed decides nothing: Lesson 06
measured the spread of this kind of result and Lesson 07 showed a parameter
ranking flipping from seed to seed. So every regime is now run twelve times,
run i on seed i, which makes the comparison paired -- the same twelve starting
populations face all four regimes, and the difference can be read run by run.

The four regimes are the fixed algorithm of step 1, each half of the adaptive
machinery on its own, and both halves together. All four stop at the same
evaluation budget, and the table prints the evaluations each one actually spent
so that the reader can check the budgets really are comparable.

CHANGES FROM tsp_04_adaptive_population.py
Introduce them in this order:
    1. RUNS               twelve paired runs, run i on seed i
    2. measure()          one regime over all the seeds, returning what it found
    3. paired_summary()   the difference against the fixed regime, run by run
    4. the_comparison     the table and the box plot replace the single-run report

Run it:  python tsp_05_twelve_runs.py

Over 12 paired runs: fixed 58,195, probabilities 50,145 (−8,050 ± 1,022,
wins 12/12), resize 53,555 (−4,640 ± 926, 10/12), both 52,622
(−5,573 ± 1,127, 11/12). Every regime spends 11,901–11,946 evaluations.
"""
from math import dist, isnan, nan, sqrt
from pathlib import Path
import random
from statistics import fmean, stdev
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

MIN_POPULATION_SIZE = 40           # below this the tournament has nothing to choose from
MAX_POPULATION_SIZE = 240          # above this a generation costs more than it returns
IMMIGRANTS_PER_STALL = 2           # fresh random routes injected when the run is stuck

# --- NEW (1) RUNS -------------------------------------------------------------
RUNS = 12                          # run i uses seed i, so every regime faces the same dice
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


def run(points: Sequence[Tuple[int, int]], seed: int, adaptive: bool = False,
        resize: bool = False) -> Dict[str, object]:
    """One run, stopped by the evaluation budget; two independent adaptive switches.

    Args:
        points: the 48 ATT cities.
        seed: run i uses seed i.
        adaptive: if True, probabilities move each generation.
        resize: if True, population shrinks or grows each generation.

    Returns:
        A record with length, evaluations spent and operator histories.
    """
    random.seed(seed)
    evaluations = 0
    crossover_probability = CROSSOVER_PROBABILITY   # constants became state in step 3
    mutation_probability = MUTATION_PROBABILITY
    crossover_history = [crossover_probability]
    mutation_history = [mutation_probability]
    size_history = [POPULATION_SIZE]                # the population became state in step 4
    immigrants = 0
    culls = 0
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
        if resize:
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


# --- NEW (2) measure() --------------------------------------------------------
def measure(points: Sequence[Tuple[int, int]], adaptive: bool, resize: bool) -> Dict[str, List[float]]:
    """Run one regime once per seed and keep what each run found and spent.

    Args:
        points: the 48 ATT cities.
        adaptive, resize: which half of the machinery is on.

    Returns:
        Lists of lengths, evaluations and generations, one entry per seed.

    Example:
        Probabilities 50,145 against fixed 58,195 (−8,050 ± 1,022, 12/12).
    """
    lengths, spents, generations = [], [], []
    for seed in range(1, RUNS + 1):
        record = run(points, seed, adaptive=adaptive, resize=resize)
        lengths.append(record["length"])
        spents.append(record["evaluations"])
        generations.append(record["generations"])
    return {"lengths": lengths, "evaluations": spents, "generations": generations}
# ------------------------------------------------------------------------------


# --- NEW (3) paired_summary() -------------------------------------------------
def paired_summary(variant: Sequence[float], reference: Sequence[float]) -> Dict[str, float]:
    """Compare two regimes run by run, not mean against mean.

    Both regimes saw the same seeds, so the honest statistic is the mean of the
    per-run differences and its standard error. The mean plus or minus two
    standard errors is reported as a descriptive approximation, not as a
    significance test.

    Args:
        variant: lengths of the candidate regime, one per seed.
        reference: lengths of the fixed regime, same seeds.

    Returns:
        Wins, mean difference, standard error, and whether the approximate
        two-SE interval excludes zero.

    Example:
        Probabilities beat fixed 12/12; resize 10/12; both 11/12.
    """
    differences = [good - base for good, base in zip(variant, reference)]
    mean_difference = fmean(differences)
    error = stdev(differences) / sqrt(len(differences)) if len(differences) > 1 else 0.0
    return {"wins": sum(1 for value in differences if value < 0),
            "mean_difference": mean_difference,
            "standard_error": error,
            "decisive": abs(mean_difference) > 2 * error}
# ------------------------------------------------------------------------------


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


# --- NEW (4) the_comparison ---------------------------------------------------
points = load_points()
regimes = [("fixed", False, False),
           ("probabilities", True, False),
           ("resize", False, True),
           ("both", True, True)]
results = {name: measure(points, adaptive, resize) for name, adaptive, resize in regimes}
reference = results["fixed"]["lengths"]

print(f"Lesson 12 - Step 5: {RUNS} runs of each regime, seeds 1-{RUNS}, "
      f"budget {BUDGET:,} evaluations")
print(f"{'regime':<16}{'mean':>9}{'sd':>8}{'best':>9}{'evals':>9}{'gens':>7}"
      f"{'wins':>7}{'mean difference vs fixed':>28}")
for name, _, _ in regimes:
    lengths = results[name]["lengths"]
    summary = paired_summary(lengths, reference)
    if name == "fixed":
        difference = "reference"
        wins = "-"
    else:
        difference = (f"{summary['mean_difference']:+,.0f} +/- "
                      f"{summary['standard_error']:,.0f}")
        wins = f"{summary['wins']}/{RUNS}"
    print(f"{name:<16}{fmean(lengths):>9,.0f}{stdev(lengths):>8,.0f}{min(lengths):>9,.0f}"
          f"{fmean(results[name]['evaluations']):>9,.0f}"
          f"{fmean(results[name]['generations']):>7,.0f}{wins:>7}{difference:>28}")

print(f"(wins = runs in which that regime returned a shorter route than the fixed regime "
      f"on the same seed)")
best_name = min(results, key=lambda name: fmean(results[name]["lengths"]))
for name, _, _ in regimes[1:]:
    summary = paired_summary(results[name]["lengths"], reference)
    verdict = ("approximate two-SE interval excludes zero"
               if summary["decisive"] else "approximate two-SE interval includes zero")
    print(f"{name:<14} vs fixed: {summary['mean_difference']:+,.0f} "
          f"({summary['mean_difference'] / fmean(reference):+.1%}), "
          f"{summary['standard_error']:,.0f} standard error -- {verdict}")
print(f"Shortest mean route: {best_name} at {fmean(results[best_name]['lengths']):,.0f}.")
print(f"Every regime spent between {min(fmean(results[n]['evaluations']) for n in results):,.0f} "
      f"and {max(fmean(results[n]['evaluations']) for n in results):,.0f} evaluations on average, "
      f"so the comparison is at equal budget.")
print("The adaptive regimes beat THIS fixed setting. Step 6 asks whether they beat a tuned one.")

FIGURES.mkdir(exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].boxplot([results[name]["lengths"] for name, _, _ in regimes])
axes[0].set_xticks(range(1, len(regimes) + 1), [name for name, _, _ in regimes])
axes[0].set(ylabel="route length", title=f"{RUNS} runs of each regime, equal budget")
axes[0].tick_params(axis="x", rotation=15)
for name, _, _ in regimes[1:]:
    differences = [variant - base for variant, base in zip(results[name]["lengths"], reference)]
    axes[1].plot(range(1, RUNS + 1), differences, "o-", label=name)
axes[1].axhline(0, color="black", linewidth=1)
axes[1].set(xlabel="run (= seed)", ylabel="route length minus the fixed regime",
            title="Paired differences: below zero is better")
axes[1].legend()
fig.tight_layout()
fig.savefig(FIGURES / "tsp_05_twelve_runs.png", dpi=160)
plt.close(fig)
# ------------------------------------------------------------------------------

