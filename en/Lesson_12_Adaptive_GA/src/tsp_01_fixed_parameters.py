"""
Lesson 12 - Step 1: One run with fixed parameters
==================================================
NEW IN THIS STEP: the whole program.

Every lesson so far has written the parameters at the top of the file as
constants; Lesson 07 then tuned them by grid search and reported which fixed
values were best. This lesson asks a different question: should they be fixed
at all? A run has an early phase, when the population is spread out and the
search needs reach, and a late phase, when the population is nearly identical
and the search needs refinement. One constant has to serve both.

The problem is Lesson 10's travelling salesperson instance -- the same 48 US
capitals, the same ordered crossover and inversion mutation, restated here so
that this file runs on its own. A permutation search is used because it is long
enough for those two phases to be visible.

One thing does change from Lesson 10: the run stops when it has spent an
evaluation budget, not after a fixed number of generations. Lesson 07
established evaluations as the currency in which two settings are compared, and
the adaptive algorithm of steps 3 and 4 changes its own population size, which
makes a generation count meaningless as a budget.

Run it:  python tsp_01_fixed_parameters.py

Seed 1 spends 11,901 evaluations in 99 generations and returns a legal
route of 57,076, 40.8% longer than the nearest-neighbour baseline of
40,526.
"""
from math import dist
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

    Example:
        Seed 1's GA route of 57,076 is 40.8% longer than this baseline.
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


def run(points: Sequence[Tuple[int, int]], seed: int) -> Dict[str, object]:
    """One run of the fixed-parameter GA, stopped by the evaluation budget.

    A generation is only started if the budget can pay for all of it, so no run
    is cut off half way and every run reports the evaluations it really spent.

    Args:
        points: the 48 ATT cities.
        seed: 1 here, which is also run 1 of later tables.

    Returns:
        A record with the champion route, its length, histories and the
        evaluations actually spent.

    Example:
        Seed 1 spends 11,901 evaluations in 99 generations and returns
        57,076, 40.8% longer than nearest neighbour.
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

    return {
        "route": best[0],
        "length": best[1],
        "best_history": best_history,
        "mean_history": mean_history,
        "spent_history": spent_history,
        "evaluations": evaluations,
        "generations": len(best_history) - 1,
    }


points = load_points()
baseline = route_length(points, nearest_neighbour(points))
record = run(points, SEED)

print("Lesson 12 - Step 1: one run with fixed parameters")
print(f"Cities:                     {len(points)}")
print(f"Crossover / mutation:       {CROSSOVER_PROBABILITY} / {MUTATION_PROBABILITY} (constant all run)")
print(f"Population:                 {POPULATION_SIZE} (constant all run)")
print(f"Budget:                     {BUDGET:,} evaluations")
print(f"Spent:                      {record['evaluations']:,} in {record['generations']} generations")
print(f"Nearest-neighbour baseline: {baseline:,.0f}")
print(f"Best GA route (seed {SEED}):    {record['length']:,.0f}")
print(f"GA versus baseline:         {(record['length'] - baseline) / baseline:+.1%}")
print(f"Final route legal:          {sorted(record['route']) == list(range(len(points)))}")

FIGURES.mkdir(exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].plot(record["best_history"], label="best of run")
axes[0].plot(record["mean_history"], label="population mean")
axes[0].axhline(baseline, color="black", linestyle="--", label="nearest neighbour")
axes[0].set(xlabel="generation", ylabel="route length",
            title=f"Fixed parameters, seed {SEED}")
axes[0].legend()
closed = record["route"] + record["route"][:1]
axes[1].plot([points[city][0] for city in closed], [points[city][1] for city in closed], "o-", markersize=3)
axes[1].set(title=f"Best route: {record['length']:,.0f}", aspect="equal")
axes[1].set_xticks([])
axes[1].set_yticks([])
fig.tight_layout()
fig.savefig(FIGURES / "tsp_01_fixed_parameters.png", dpi=160)
plt.close(fig)

