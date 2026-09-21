"""
Lesson 09 - Radar 4: Binary operators
======================================
NEW IN THIS STEP: tournament(), crossover(), and mutate().

CHANGES FROM radar_03_the_penalty.py
Introduce them in this order:
    1. tournament()          select using the audited objective
    2. crossover()           exchange site decisions at one cut
    3. mutate()              flip one installation decision

Run it:  python radar_04_operators.py

One generation lowers mean objective from 86.0 to 42.6.
"""
from dataclasses import dataclass
from math import dist
from pathlib import Path
import random
from typing import List, Sequence, Set

import matplotlib.pyplot as plt

SEED = 3
RADIUS = 3.2
POPULATION_SIZE = 100
SITES = [(x, y) for y in (1, 5, 9) for x in (1, 5, 9)]
TARGETS = [(x, y) for y in range(0, 11, 2) for x in range(0, 11, 2)]
FIGURES = Path(__file__).resolve().parent.parent / "figures"


def covered_targets(bits: Sequence[int]) -> Set[int]:
    """Targets within radius 3.2 of at least one active site.

    Args:
        bits: length-9 installation plan.

    Returns:
        Indices of covered targets.
    """
    covered: Set[int] = set()
    for site_index, active in enumerate(bits):
        if active:
            covered.update(i for i, target in enumerate(TARGETS)
                           if dist(SITES[site_index], target) <= RADIUS)
    return covered


@dataclass(frozen=True)
class Candidate:
    """One installation plan, with radar count and uncovered targets kept together.

    Args:
        bits: length-9 chromosome.
        radar_count: number of sites switched on.
        uncovered: how many of the 36 targets sit outside every radius.
    """
    bits: List[int]
    radar_count: int
    uncovered: int


def make_candidate(bits: List[int]) -> Candidate:
    """Score an installation plan without repairing it.

    Args:
        bits: length-9 vector of 0/1 site decisions.

    Returns:
        A ``Candidate`` carrying radar count and uncovered-target count.
    """
    return Candidate(bits, sum(bits), len(TARGETS) - len(covered_targets(bits)))


def random_candidate() -> Candidate:
    """Sample one installation plan independently per site.

    Returns:
        A ``Candidate`` that may leave targets uncovered.
    """
    return make_candidate([random.choice((0, 1)) for _ in SITES])


def objective(candidate: Candidate) -> int:
    """Lower is better; coverage dominates radar count by construction.

    Args:
        candidate: a scored installation plan.

    Returns:
        ``uncovered * 10 + radar_count``.

    Example:
        One generation lowers the mean from 86.0 to 42.6.
    """
    return candidate.uncovered * (len(SITES) + 1) + candidate.radar_count


# --- NEW (1) tournament() -----------------------------------------------------
def tournament(population: List[Candidate], size: int = 3) -> Candidate:
    """Select using the audited objective.

    Args:
        population: scored installation plans.
        size: tournament size. 3 is Lesson 03's default.

    Returns:
        The lowest-objective member of a uniform sample of ``size``.
    """
    return min(random.sample(population, size), key=objective)
# ------------------------------------------------------------------------------


# --- NEW (2) crossover() ------------------------------------------------------
def crossover(first: Candidate, second: Candidate) -> Candidate:
    """Exchange site decisions at one cut.

    Args:
        first, second: parent plans.

    Returns:
        One child. Coverage is re-scored, not repaired.
    """
    cut = random.randrange(1, len(SITES))
    return make_candidate(first.bits[:cut] + second.bits[cut:])
# ------------------------------------------------------------------------------


# --- NEW (3) mutate() ---------------------------------------------------------
def mutate(candidate: Candidate) -> Candidate:
    """Flip one installation decision.

    Args:
        candidate: a child of crossover.

    Returns:
        A new ``Candidate`` with exactly one bit inverted.

    Example:
        Together with crossover this lowers mean objective from 86.0 to
        42.6 in one generation.
    """
    bits = candidate.bits.copy()
    index = random.randrange(len(bits))
    bits[index] = 1 - bits[index]
    return make_candidate(bits)
# ------------------------------------------------------------------------------


random.seed(SEED)
population = [random_candidate() for _ in range(POPULATION_SIZE)]
children = [mutate(crossover(tournament(population), tournament(population)))
            for _ in range(POPULATION_SIZE)]

print("Lesson 09 - Radar 4: binary operators")
print(f"Mean parent objective: {sum(map(objective, population)) / POPULATION_SIZE:.1f}")
print(f"Mean child objective:  {sum(map(objective, children)) / POPULATION_SIZE:.1f}")
print(f"Fully covering parents: {sum(c.uncovered == 0 for c in population)}")
print(f"Fully covering children: {sum(c.uncovered == 0 for c in children)}")

FIGURES.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(7, 5))
ax.boxplot([[objective(c) for c in population], [objective(c) for c in children]],
           tick_labels=["parents", "children"])
ax.set(ylabel="objective", title="Selection shifts the objective in one generation")
fig.tight_layout()
fig.savefig(FIGURES / "radar_04_operators.png", dpi=160)
plt.close(fig)
