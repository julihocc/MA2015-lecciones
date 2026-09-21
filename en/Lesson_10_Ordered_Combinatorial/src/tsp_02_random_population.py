"""
Lesson 10 - TSP 2: Random legal routes
=======================================
NEW IN THIS STEP: random_route(), POPULATION_SIZE, and a length census.

CHANGES FROM tsp_01_the_route.py
Introduce them in this order:
    1. random_route()        shuffle the city indices without losing any city
    2. POPULATION_SIZE       define the sampling budget
    3. the census            measure how much legal routes still differ

Run it:  python tsp_02_random_population.py

All 500 shuffled routes are legal, with a wide length distribution.
"""
from math import dist
from pathlib import Path
import random
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt

SEED = 60
DATA = Path(__file__).with_name("att48_xy.txt")
FIGURES = Path(__file__).resolve().parent.parent / "figures"


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
        points: city coordinates, indexed by gene value.
        route: a permutation of ``0..len(points)-1``.

    Returns:
        Euclidean length of the closed tour.
    """
    pairs = zip(route, route[1:] + route[:1])
    return sum(dist(points[first], points[second]) for first, second in pairs)


# --- NEW (1) random_route() ---------------------------------------------------
def random_route(size: int) -> List[int]:
    """Shuffle the city indices without losing any city.

    Args:
        size: number of cities, 48 here.

    Returns:
        A permutation of ``0..size-1``. Every shuffle is legal.

    Example:
        All 500 shuffled routes are legal, with a wide length distribution.
    """
    route = list(range(size))
    random.shuffle(route)
    return route
# ------------------------------------------------------------------------------


# --- NEW (2) POPULATION_SIZE --------------------------------------------------
POPULATION_SIZE = 500
# ------------------------------------------------------------------------------


# --- NEW (3) the census -------------------------------------------------------
random.seed(SEED)
points = load_points()
population = [random_route(len(points)) for _ in range(POPULATION_SIZE)]
lengths = [route_length(points, route) for route in population]
# ------------------------------------------------------------------------------

print("Lesson 10 - TSP 2: random legal routes")
print(f"Legal permutations: {sum(sorted(r) == list(range(len(points))) for r in population)}/{POPULATION_SIZE}")
print(f"Route length: min={min(lengths):,.1f}, mean={sum(lengths) / len(lengths):,.1f}, max={max(lengths):,.1f}")

FIGURES.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(lengths, bins=25, color="tab:blue")
ax.set(xlabel="closed-route length", ylabel="routes", title="Every route is legal; quality still varies")
fig.tight_layout()
fig.savefig(FIGURES / "tsp_02_random_population.png", dpi=160)
plt.close(fig)
