"""
Lesson 10 - TSP 4: Mutation changes order, not membership
=========================================================
NEW IN THIS STEP: swap_mutation(), inversion_mutation(), and a distance audit.

CHANGES FROM tsp_03_ordered_crossover.py
Introduce them in this order:
    1. swap_mutation()       exchange two positions
    2. inversion_mutation()  reverse one contiguous segment
    3. the distance audit    measure how differently the operators move

Run it:  python tsp_04_mutation.py

Swap and inversion preserve every city; their mean length changes are
+451.0 and +238.9.
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


def ordered_crossover(first: Sequence[int], second: Sequence[int]) -> List[int]:
    """Keep one segment and fill holes in the other parent's order.

    Args:
        first: the parent that donates the kept segment.
        second: the parent that donates the remaining cities, in order.

    Returns:
        A permutation. Step 3 measured 1,000/1,000 legal children.
    """
    left, right = sorted(random.sample(range(len(first)), 2))
    child = [-1] * len(first); child[left:right] = first[left:right]
    remaining = [city for city in second if city not in child]
    holes = list(range(right, len(first))) + list(range(left))
    for index, city in zip(holes, remaining): child[index] = city
    return child


# --- NEW (1) swap_mutation() --------------------------------------------------
def swap_mutation(source: Sequence[int]) -> List[int]:
    """Exchange two positions.

    Args:
        source: a legal route.

    Returns:
        A new permutation. Membership is unchanged; two edges usually are.

    Example:
        Mean length change over 1,000 trials is +451.0.
    """
    route = list(source)
    first, second = random.sample(range(len(route)), 2)
    route[first], route[second] = route[second], route[first]
    return route
# ------------------------------------------------------------------------------


# --- NEW (2) inversion_mutation() --------------------------------------------
def inversion_mutation(source: Sequence[int]) -> List[int]:
    """Reverse one contiguous segment.

    Args:
        source: a legal route.

    Returns:
        A new permutation. Only the two edges at the cut change.

    Example:
        Mean length change over 1,000 trials is +238.9, milder than swap.
    """
    route = list(source)
    left, right = sorted(random.sample(range(len(route)), 2))
    route[left:right] = reversed(route[left:right])
    return route
# ------------------------------------------------------------------------------


# --- NEW (3) the distance audit ----------------------------------------------
random.seed(SEED)
points = load_points()
base = random_route(len(points))
base_length = route_length(points, base)
swap_delta = [route_length(points, swap_mutation(base)) - base_length for _ in range(TRIALS)]
invert_delta = [route_length(points, inversion_mutation(base)) - base_length for _ in range(TRIALS)]
# ------------------------------------------------------------------------------

print("Lesson 10 - TSP 4: mutation")
print(f"All swap mutants legal: {all(sorted(swap_mutation(base)) == list(range(len(points))) for _ in range(100))}")
print(f"All inversion mutants legal: {all(sorted(inversion_mutation(base)) == list(range(len(points))) for _ in range(100))}")
print(f"Mean length change, swap: {sum(swap_delta) / TRIALS:+.1f}")
print(f"Mean length change, inversion: {sum(invert_delta) / TRIALS:+.1f}")

FIGURES.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 4))
ax.boxplot([swap_delta, invert_delta], tick_labels=["swap", "inversion"], showfliers=False)
ax.axhline(0, color="black", linewidth=1)
ax.set(ylabel="change in route length", title="Both mutations stay legal but move differently")
fig.tight_layout()
fig.savefig(FIGURES / "tsp_04_mutation.png", dpi=160)
plt.close(fig)
