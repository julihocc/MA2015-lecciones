"""
Lesson 09 - Radar 2: The chromosome
====================================
NEW IN THIS STEP: Candidate, random_candidate(), and the coverage census.

CHANGES FROM radar_01_the_landscape.py
Introduce them in this order:
    1. Candidate             keep radar count and uncovered targets together
    2. random_candidate()    sample one installation plan
    3. the census            measure the tradeoff before assigning a fitness

Run it:  python radar_02_the_chromosome.py

21 of 500 random plans cover every target.
"""
from dataclasses import dataclass
from math import dist
from pathlib import Path
import random
from typing import List, Sequence, Set

import matplotlib.pyplot as plt

SEED = 3
RADIUS = 3.2
POPULATION_SIZE = 500
SITES = [(x, y) for y in (1, 5, 9) for x in (1, 5, 9)]
TARGETS = [(x, y) for y in range(0, 11, 2) for x in range(0, 11, 2)]
FIGURES = Path(__file__).resolve().parent.parent / "figures"


def covered_targets(bits: Sequence[int]) -> Set[int]:
    """Targets within radius 3.2 of at least one active site.

    Args:
        bits: length-9 installation plan.

    Returns:
        Indices of covered targets. Full coverage is a set of size 36.
    """
    covered: Set[int] = set()
    for site_index, active in enumerate(bits):
        if active:
            covered.update(i for i, target in enumerate(TARGETS)
                           if dist(SITES[site_index], target) <= RADIUS)
    return covered


# --- NEW (1) Candidate --------------------------------------------------------
@dataclass(frozen=True)
class Candidate:
    """One installation plan, with radar count and uncovered targets kept together.

    Args:
        bits: length-9 chromosome.
        radar_count: number of sites switched on.
        uncovered: how many of the 36 targets sit outside every radius.

    Example:
        21 of 500 random plans have ``uncovered == 0``.
    """
    bits: List[int]
    radar_count: int
    uncovered: int

    @property
    def feasible(self) -> bool:
        """Whether every target is covered.

        Returns:
            True iff ``uncovered == 0``.
        """
        return self.uncovered == 0
# ------------------------------------------------------------------------------


# --- NEW (2) random_candidate() -----------------------------------------------
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

    Example:
        21 of 500 such plans cover every target.
    """
    return make_candidate([random.choice((0, 1)) for _ in SITES])
# ------------------------------------------------------------------------------


# --- NEW (3) the census -------------------------------------------------------
random.seed(SEED)
population = [random_candidate() for _ in range(POPULATION_SIZE)]
feasible = [candidate for candidate in population if candidate.feasible]
# ------------------------------------------------------------------------------

print("Lesson 09 - Radar 2: the chromosome")
print(f"Fully covering random plans: {len(feasible)}/{POPULATION_SIZE}")
print(f"Radar counts sampled: {min(c.radar_count for c in population)}.."
      f"{max(c.radar_count for c in population)}")
print(f"Best coverage miss: {min(c.uncovered for c in population)} targets")

FIGURES.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter([c.radar_count for c in population], [c.uncovered for c in population], alpha=0.35)
ax.set(xlabel="radars installed", ylabel="uncovered targets",
       title="Coverage and installation count are separate facts")
fig.tight_layout()
fig.savefig(FIGURES / "radar_02_the_chromosome.png", dpi=160)
plt.close(fig)
