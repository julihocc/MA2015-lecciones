"""
Lesson 10 - TSP 3: Crossover must preserve a permutation
=========================================================
NEW IN THIS STEP: one_point(), ordered_crossover(), and a legality experiment.

CHANGES FROM tsp_02_random_population.py
Introduce them in this order:
    1. one_point()           expose why a binary-style cut duplicates cities
    2. ordered_crossover()   keep one segment and fill holes in the other parent's order
    3. the experiment        count legal children from both operators

Run it:  python tsp_03_ordered_crossover.py

One-point crossover produces 0/1,000 legal children; ordered crossover
produces 1,000/1,000.
"""
from math import dist
from pathlib import Path
import random
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt

SEED = 60
DATA = Path(__file__).with_name("att48_xy.txt")
FIGURES = Path(__file__).resolve().parent.parent / "figures"
TRIALS = 1000


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


# --- NEW (1) one_point() ------------------------------------------------------
def one_point(first: Sequence[int], second: Sequence[int]) -> List[int]:
    """A binary-style cut: the child duplicates some cities and loses others.

    Args:
        first, second: parent permutations.

    Returns:
        One child. Almost never a permutation.

    Example:
        0 of 1,000 one-point children are legal.
    """
    cut = random.randrange(1, len(first))
    return list(first[:cut]) + list(second[cut:])
# ------------------------------------------------------------------------------


# --- NEW (2) ordered_crossover() ---------------------------------------------
def ordered_crossover(first: Sequence[int], second: Sequence[int]) -> List[int]:
    """Keep one segment and fill holes in the other parent's order.

    Args:
        first: the parent that donates the kept segment.
        second: the parent that donates the remaining cities, in order.

    Returns:
        A permutation: every city appears once.

    Example:
        1,000 of 1,000 ordered children are legal, against 0 of 1,000
        one-point children.
    """
    left, right = sorted(random.sample(range(len(first)), 2))
    child = [-1] * len(first)
    child[left:right] = first[left:right]
    remaining = [city for city in second if city not in child]
    holes = list(range(right, len(first))) + list(range(0, left))
    for index, city in zip(holes, remaining):
        child[index] = city
    return child
# ------------------------------------------------------------------------------


# --- NEW (3) the experiment ---------------------------------------------------
random.seed(SEED)
size = len(load_points())
ordinary, ordered = [], []
for _ in range(TRIALS):
    parents = random_route(size), random_route(size)
    ordinary.append(one_point(*parents))
    ordered.append(ordered_crossover(*parents))
legal = lambda route: sorted(route) == list(range(size))
# ------------------------------------------------------------------------------

print("Lesson 10 - TSP 3: ordered crossover")
print(f"Legal one-point children: {sum(map(legal, ordinary))}/{TRIALS}")
print(f"Legal ordered children:   {sum(map(legal, ordered))}/{TRIALS}")
print(f"Distinct cities in a typical broken child: {len(set(ordinary[0]))}/{size}")

FIGURES.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(["one point", "ordered"], [sum(map(legal, ordinary)), sum(map(legal, ordered))], color=["tab:red", "tab:green"])
ax.set(ylabel=f"legal children out of {TRIALS}", title="The operator determines chromosome legality")
fig.tight_layout()
fig.savefig(FIGURES / "tsp_03_ordered_crossover.png", dpi=160)
plt.close(fig)
