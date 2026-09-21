"""
Lesson 12 - Step 6: The tuned rival
====================================
NEW IN THIS STEP: run() takes its starting rates as arguments, FIXED_SETTINGS,
and the comparison that decides the lesson.

Step 5 showed all three adaptive regimes beating the fixed algorithm, the best of
them by eight thousand units of route length. But the fixed algorithm they beat was never tuned: its
crossover and mutation probabilities were inherited from Lesson 10, where they
were chosen for a different budget and never questioned. Lesson 07 spent a whole
session on the right way to choose them, and the adaptive scheme has not yet
been asked to beat that.

So this step does Lesson 07's job on this problem -- a small sweep of fixed
settings, each measured over the same twelve seeds at the same budget -- and puts
the best of them against the adaptive regime. Four settings is a thin grid,
chosen to fit a class; the point is not to find the optimum of the sweep but to
find out whether the adaptive machinery survives contact with an opponent that
has been tuned at all.

The last two lines of the output are the lesson.

CHANGES FROM tsp_05_twelve_runs.py
Introduce them in this order:
    1. run(rates)       the starting probabilities become arguments, not constants
    2. FIXED_SETTINGS   four fixed settings, the first one being step 1's
    3. the_verdict      each setting against the adaptive regime, paired, at equal budget

Run it:  python tsp_06_a_tuned_rival.py

Against four fixed settings at the same budget: 0.90/0.25 → 58,195
(adaptive wins 12/12), 0.60/0.05 → 48,870 (4/12, +1,274 ± 727),
0.40/0.15 → 49,675 (6/12, +469 ± 480), 0.20/0.30 → 49,626 (5/12,
+519 ± 835). The adaptive regime is behind 3 of the 4 on the mean.
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

RUNS = 12                          # run i uses seed i, so every regime faces the same dice

# --- NEW (2) FIXED_SETTINGS ---------------------------------------------------
# Four fixed (crossover, mutation) settings, in the style of Lesson 07's grid.
# The first is the setting steps 1-5 inherited from Lesson 10; the other three
# are corners of the same sweep, quieter in one operator or the other.
FIXED_SETTINGS = [(0.9, 0.25), (0.6, 0.05), (0.4, 0.15), (0.2, 0.30)]
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
        Clipped ``(crossover, mutation)``. The adaptive schedule averages
        crossover 0.60 and mutation 0.18, the same crossover as the best
        fixed cell.
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


# --- NEW (1) run(rates) -------------------------------------------------------
def run(points: Sequence[Tuple[int, int]], seed: int, adaptive: bool = False,
        resize: bool = False, crossover: float = CROSSOVER_PROBABILITY,
        mutation: float = MUTATION_PROBABILITY) -> Dict[str, object]:
    """One run, stopped by the evaluation budget; the starting rates are arguments now.

    Nothing else changes: with the default arguments this is exactly the run() of
    step 5. Making the rates arguments is what lets one script try four settings
    without four copies of the constants.

    Args:
        points: the 48 ATT cities.
        seed: run i uses seed i.
        adaptive, resize: which half of the machinery is on.
        crossover, mutation: starting rates. Defaults are step 1's 0.90/0.25.

    Returns:
        A record with length, evaluations spent and operator histories.

    Example:
        Against 0.60/0.05 the adaptive regime is behind +1,274 ± 727
        (4 wins of 12).
    """
    random.seed(seed)
    evaluations = 0
    crossover_probability = crossover
    mutation_probability = mutation
# ------------------------------------------------------------------------------
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


def measure(points: Sequence[Tuple[int, int]], adaptive: bool, resize: bool,
            crossover: float = CROSSOVER_PROBABILITY,
            mutation: float = MUTATION_PROBABILITY) -> Dict[str, List[float]]:
    """Run one regime once per seed and keep what each run found and spent.

    Args:
        points: the 48 ATT cities.
        adaptive, resize: which half of the machinery is on.
        crossover, mutation: starting rates of the fixed or adaptive run.

    Returns:
        Lengths, evaluations, generations and mean operator rates per seed.
    """
    lengths, spents, generations = [], [], []
    crossover_means, mutation_means = [], []
    for seed in range(1, RUNS + 1):
        record = run(points, seed, adaptive=adaptive, resize=resize,
                     crossover=crossover, mutation=mutation)   # (1) the rates travel through
        lengths.append(record["length"])
        spents.append(record["evaluations"])
        generations.append(record["generations"])
        crossover_means.append(fmean(record["crossover_history"]))
        mutation_means.append(fmean(record["mutation_history"]))
    return {"lengths": lengths, "evaluations": spents, "generations": generations,
            "crossover": crossover_means, "mutation": mutation_means}


def paired_summary(variant: Sequence[float], reference: Sequence[float]) -> Dict[str, float]:
    """Compare two regimes run by run, not mean against mean.

    Args:
        variant: lengths of the adaptive regime, one per seed.
        reference: lengths of one fixed setting, same seeds.

    Returns:
        Wins, mean difference, standard error, and whether the approximate
        two-SE interval excludes zero. This is descriptive, not a significance test.

    Example:
        Against the best fixed cell (0.60/0.05) the gap is +1,274 ± 727,
        so the approximate two-SE interval includes zero at twelve runs.
    """
    differences = [good - base for good, base in zip(variant, reference)]
    mean_difference = fmean(differences)
    error = stdev(differences) / sqrt(len(differences)) if len(differences) > 1 else 0.0
    return {"wins": sum(1 for value in differences if value < 0),
            "mean_difference": mean_difference,
            "standard_error": error,
            "decisive": abs(mean_difference) > 2 * error}


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


# --- NEW (3) the_verdict ------------------------------------------------------
points = load_points()
adaptive_result = measure(points, adaptive=True, resize=False)
adaptive_lengths = adaptive_result["lengths"]
fixed_results = {rates: measure(points, adaptive=False, resize=False,
                                crossover=rates[0], mutation=rates[1])
                 for rates in FIXED_SETTINGS}

print(f"Lesson 12 - Step 6: the tuned rival, {RUNS} runs each, budget {BUDGET:,} evaluations")
print(f"{'regime':<28}{'mean':>9}{'sd':>8}{'best':>9}{'evals':>9}"
      f"{'adaptive wins':>15}{'adaptive minus fixed':>24}")
print(f"{'adaptive (probabilities)':<28}{fmean(adaptive_lengths):>9,.0f}"
      f"{stdev(adaptive_lengths):>8,.0f}{min(adaptive_lengths):>9,.0f}"
      f"{fmean(adaptive_result['evaluations']):>9,.0f}{'reference':>15}{'':>24}")
for rates in FIXED_SETTINGS:
    lengths = fixed_results[rates]["lengths"]
    summary = paired_summary(adaptive_lengths, lengths)
    label = f"fixed {rates[0]:.2f} / {rates[1]:.2f}"
    if rates == (CROSSOVER_PROBABILITY, MUTATION_PROBABILITY):
        label += " (step 1)"
    wins = f"{summary['wins']}/{RUNS}"
    difference = f"{summary['mean_difference']:+,.0f} +/- {summary['standard_error']:,.0f}"
    print(f"{label:<28}{fmean(lengths):>9,.0f}{stdev(lengths):>8,.0f}{min(lengths):>9,.0f}"
          f"{fmean(fixed_results[rates]['evaluations']):>9,.0f}{wins:>15}{difference:>24}")
print("(adaptive wins = paired runs in which the adaptive regime returned the shorter route;")
print(" adaptive minus fixed: a positive number means the adaptive regime came out longer)")

best_rates = min(FIXED_SETTINGS, key=lambda rates: fmean(fixed_results[rates]["lengths"]))
best_lengths = fixed_results[best_rates]["lengths"]
duel = paired_summary(adaptive_lengths, best_lengths)
interval_note = ("approximate two-SE interval excludes zero"
                 if duel["decisive"] else "approximate two-SE interval includes zero")
print(f"Best fixed setting among four on these same twelve seeds: crossover {best_rates[0]:.2f}, "
      f"mutation {best_rates[1]:.2f}, "
      f"mean {fmean(best_lengths):,.0f}.")
print(f"Adaptive mean {fmean(adaptive_lengths):,.0f} versus that setting "
      f"({duel['mean_difference']:+,.0f} +/- {duel['standard_error']:,.0f}, "
      f"winning {duel['wins']} of {RUNS} paired runs); {interval_note}.")
behind = sum(1 for rates in FIXED_SETTINGS
             if fmean(fixed_results[rates]["lengths"]) < fmean(adaptive_lengths))
print(f"On the mean it is behind {behind} of the {len(FIXED_SETTINGS)} fixed settings and ahead of "
      f"{len(FIXED_SETTINGS) - behind}.")
print(f"The schedule spends the run averaging crossover {fmean(adaptive_result['crossover']):.2f} "
      f"and mutation {fmean(adaptive_result['mutation']):.2f}; the best fixed setting is "
      f"crossover {best_rates[0]:.2f}, mutation {best_rates[1]:.2f}.")
print("This exploratory comparison selected and evaluated the fixed settings on the same seeds;")
print("it describes these runs, but it is not independent validation or a significance test.")

FIGURES.mkdir(exist_ok=True)
labels = ["adaptive"] + [f"{rates[0]:.2f}/{rates[1]:.2f}" for rates in FIXED_SETTINGS]
means = [fmean(adaptive_lengths)] + [fmean(fixed_results[rates]["lengths"]) for rates in FIXED_SETTINGS]
errors = [stdev(adaptive_lengths) / sqrt(RUNS)] + [
    stdev(fixed_results[rates]["lengths"]) / sqrt(RUNS) for rates in FIXED_SETTINGS]
colours = ["tab:orange"] + ["tab:blue"] * len(FIXED_SETTINGS)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].bar(labels, means, yerr=errors, capsize=4, color=colours)
axes[0].set(ylabel="mean route length", ylim=(0, max(means) * 1.2),
            title=f"Adaptive against four fixed settings ({RUNS} runs each)")
axes[0].tick_params(axis="x", rotation=15)
axes[1].boxplot([adaptive_lengths] + [fixed_results[rates]["lengths"] for rates in FIXED_SETTINGS])
axes[1].set_xticks(range(1, len(labels) + 1), labels)
axes[1].set(ylabel="route length", title="Same data, run by run")
axes[1].tick_params(axis="x", rotation=15)
fig.tight_layout()
fig.savefig(FIGURES / "tsp_06_a_tuned_rival.png", dpi=160)
plt.close(fig)
# ------------------------------------------------------------------------------

