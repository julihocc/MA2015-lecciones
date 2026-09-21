"""
Lesson 10 - TSP 1: A route is an ordering
==========================================
NEW IN THIS STEP: load_points(), route_length(), and a reference route.

A TSP chromosome contains every city exactly once. Its genes are not separate
choices: changing one position changes the meaning of the whole route.

Run it:  python tsp_01_the_route.py

The supplied instance contains 48 cities; the identity route is legal but
long.
"""
from math import dist
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt

SEED = 60
DATA = Path(__file__).with_name("att48_xy.txt")
FIGURES = Path(__file__).resolve().parent.parent / "figures"


def load_points(path: Path = DATA) -> List[Tuple[int, int]]:
    """Load the 48 ATT city coordinates shipped with the lesson.

    Args:
        path: ``att48_xy.txt`` beside this script. Lesson 12 restates the
            same file locally rather than importing across lessons.

    Returns:
        48 ``(x, y)`` integer pairs, one per city.

    Example:
        The identity route over these 48 cities is legal but long.
    """
    return [tuple(map(int, line.split())) for line in path.read_text().splitlines() if line.strip()]


def route_length(points: Sequence[Tuple[int, int]], route: Sequence[int]) -> float:
    """Close the tour by returning from the final city to the first.

    Args:
        points: city coordinates, indexed by gene value.
        route: a permutation of ``0..len(points)-1``.

    Returns:
        Euclidean length of the closed tour. Legality is not scored here.

    Example:
        The identity route over 48 cities is legal and still a long tour.
    """
    pairs = zip(route, route[1:] + route[:1])
    return sum(dist(points[first], points[second]) for first, second in pairs)


points = load_points()
route = list(range(len(points)))
length = route_length(points, route)

print("Lesson 10 - TSP 1: a route is an ordering")
print(f"Cities: {len(points)}; chromosome length: {len(route)}")
print(f"Identity route contains every city once: {sorted(route) == list(range(len(points)))}")
print(f"Identity-route length: {length:,.1f}")

FIGURES.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 5))
closed = route + route[:1]
ax.plot([points[i][0] for i in closed], [points[i][1] for i in closed], "o-", markersize=3)
ax.set(title="A legal route can still be very long", aspect="equal")
fig.tight_layout()
fig.savefig(FIGURES / "tsp_01_the_route.png", dpi=160)
plt.close(fig)
