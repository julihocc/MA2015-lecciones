"""
Lesson 10 - TSP 5: The full ordered search
===========================================
NEW IN THIS STEP: nearest_neighbour(), tournament(), and run().

CHANGES FROM tsp_04_mutation.py
Introduce them in this order:
    1. nearest_neighbour()   establish a cheap constructive baseline
    2. tournament()          prefer shorter legal routes
    3. run()                 combine ordered crossover, inversion and elitism

Run it:  python tsp_05_the_full_search.py

The GA route is legal but 57.2% longer than nearest neighbour after 10,100
evaluations.
"""
from math import dist
from pathlib import Path
import random
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt

SEED = 60
DATA = Path(__file__).with_name("att48_xy.txt")
FIGURES = Path(__file__).resolve().parent.parent / "figures"
POPULATION_SIZE = 100
GENERATIONS = 100
MUTATION_PROBABILITY = 0.25


def load_points(path: Path = DATA) -> List[Tuple[int, int]]:
    """Load the 48 ATT city coordinates shipped with the lesson.

    Args:
        path: ``att48_xy.txt`` beside this script.

    Returns:
        48 ``(x, y)`` integer pairs.
    """
    return [tuple(map(int, line.split())) for line in path.read_text().splitlines() if line.strip()]


def route_length(points: Sequence[Tuple[int, int]], route: Sequence[int]) -> float:
    """Close the tour by returning from the final city to the first.

    Args:
        points: city coordinates.
        route: a permutation of city indices.

    Returns:
        Euclidean length of the closed tour.
    """
    return sum(dist(points[a], points[b]) for a, b in zip(route, route[1:] + route[:1]))


def random_route(size: int) -> List[int]:
    """Shuffle the city indices without losing any city.

    Args:
        size: number of cities, 48 here.

    Returns:
        A permutation of ``0..size-1``.
    """
    route = list(range(size)); random.shuffle(route); return route


def ordered_crossover(first: Sequence[int], second: Sequence[int]) -> List[int]:
    """Keep one segment and fill holes in the other parent's order.

    Args:
        first: the parent that donates the kept segment.
        second: the parent that donates the remaining cities, in order.

    Returns:
        A permutation.
    """
    left, right = sorted(random.sample(range(len(first)), 2))
    child = [-1] * len(first); child[left:right] = first[left:right]
    remaining = [city for city in second if city not in child]
    holes = list(range(right, len(first))) + list(range(left))
    for index, city in zip(holes, remaining): child[index] = city
    return child


def inversion_mutation(source: Sequence[int]) -> List[int]:
    """Reverse one contiguous segment.

    Args:
        source: a legal route.

    Returns:
        A new permutation. Step 4 measured a mean length change of +238.9.
    """
    route = list(source)
    left, right = sorted(random.sample(range(len(route)), 2))
    route[left:right] = reversed(route[left:right])
    return route


# --- NEW (1) nearest_neighbour() ---------------------------------------------
def nearest_neighbour(points: Sequence[Tuple[int, int]], start: int = 0) -> List[int]:
    """A cheap constructive baseline: always take the nearest unused city.

    Args:
        points: city coordinates.
        start: city index to leave first. 0 here.

    Returns:
        A legal tour. Not optimal; it is the comparison the GA has to beat.

    Example:
        The GA route is legal but 57.2% longer than this baseline.
    """
    route, unseen = [start], set(range(len(points))) - {start}
    while unseen:
        current = route[-1]
        next_city = min(unseen, key=lambda city: dist(points[current], points[city]))
        route.append(next_city); unseen.remove(next_city)
    return route
# ------------------------------------------------------------------------------


# --- NEW (2) tournament() -----------------------------------------------------
def tournament(population: List[List[int]], points: Sequence[Tuple[int, int]], size: int = 3) -> List[int]:
    """Prefer shorter legal routes.

    Args:
        population: legal tours.
        points: city coordinates used to score length.
        size: tournament size. 3 is Lesson 03's default.

    Returns:
        The shortest member of a uniform sample of ``size``.
    """
    return min(random.sample(population, size), key=lambda route: route_length(points, route))
# ------------------------------------------------------------------------------


# --- NEW (3) run() ------------------------------------------------------------
def run(points: Sequence[Tuple[int, int]]) -> Tuple[List[int], List[float]]:
    """One hundred generations of ordered crossover, inversion and elitism.

    Args:
        points: the 48 ATT cities.

    Returns:
        ``(best, history)``. History is best-so-far length.

    Example:
        The GA route is legal but 57.2% longer than nearest neighbour
        after 10,100 evaluations.
    """
    random.seed(SEED)
    population = [random_route(len(points)) for _ in range(POPULATION_SIZE)]
    best = min(population, key=lambda route: route_length(points, route))
    history = [route_length(points, best)]
    for _ in range(GENERATIONS):
        children = [best]
        while len(children) < POPULATION_SIZE:
            child = ordered_crossover(tournament(population, points), tournament(population, points))
            if random.random() < MUTATION_PROBABILITY:
                child = inversion_mutation(child)
            children.append(child)
        population = children
        best = min(population, key=lambda route: route_length(points, route))
        history.append(route_length(points, best))
    return best, history
# ------------------------------------------------------------------------------


points = load_points()
baseline = nearest_neighbour(points)
best, history = run(points)
baseline_length = route_length(points, baseline)
best_length = route_length(points, best)

print("Lesson 10 - TSP 5: the full ordered search")
print(f"Nearest-neighbour baseline: {baseline_length:,.1f}")
print(f"GA route:                   {best_length:,.1f}")
print(f"GA versus baseline:         {(baseline_length - best_length) / baseline_length:+.1%}")
print(f"Final route legal:          {sorted(best) == list(range(len(points)))}")
print(f"Route evaluations:          {POPULATION_SIZE * (GENERATIONS + 1):,}")

FIGURES.mkdir(exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].plot(history)
axes[0].axhline(baseline_length, color="black", linestyle="--", label="nearest neighbour")
axes[0].set(xlabel="generation", ylabel="route length", title="Search history")
axes[0].legend()
closed = best + best[:1]
axes[1].plot([points[i][0] for i in closed], [points[i][1] for i in closed], "o-", markersize=3)
axes[1].set(title="Best GA route", aspect="equal")
fig.tight_layout()
fig.savefig(FIGURES / "tsp_05_the_full_search.png", dpi=160)
plt.close(fig)

