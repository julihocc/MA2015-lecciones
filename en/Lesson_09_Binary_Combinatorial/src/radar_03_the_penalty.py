"""
Lesson 09 - Radar 3: A lexicographic penalty
=============================================
NEW IN THIS STEP: objective() and a ranking audit.

CHANGES FROM radar_02_the_chromosome.py
Introduce them in this order:
    1. objective()           one missed target must outweigh every radar saving
    2. the ranking audit     verify feasibility really comes first

Run it:  python radar_03_the_penalty.py

The audited penalty ranks every feasible plan ahead of every infeasible
plan.
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

    @property
    def feasible(self) -> bool:
        """Whether every target is covered.

        Returns:
            True iff ``uncovered == 0``.
        """
        return self.uncovered == 0


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


# --- NEW (1) objective() ------------------------------------------------------
def objective(candidate: Candidate) -> int:
    """Lower is better; coverage dominates radar count by construction.

    Args:
        candidate: a scored installation plan.

    Returns:
        ``uncovered * 10 + radar_count``. One missed target outweighs
        turning every remaining radar off.

    Example:
        The ranking audit puts every feasible plan ahead of every
        infeasible plan.
    """
    return candidate.uncovered * (len(SITES) + 1) + candidate.radar_count
# ------------------------------------------------------------------------------


# --- NEW (2) the ranking audit ------------------------------------------------
random.seed(SEED)
population = [random_candidate() for _ in range(POPULATION_SIZE)]
ranked = sorted(population, key=objective)
best = ranked[0]
best_infeasible = min((c for c in population if not c.feasible), key=objective)
# ------------------------------------------------------------------------------

print("Lesson 09 - Radar 3: the penalty")
print(f"Best plan: radars={best.radar_count}, uncovered={best.uncovered}, objective={objective(best)}")
print(f"Best infeasible plan: radars={best_infeasible.radar_count}, "
      f"uncovered={best_infeasible.uncovered}, objective={objective(best_infeasible)}")
print(f"Every feasible plan outranks every infeasible plan: "
      f"{max(objective(c) for c in population if c.feasible) < min(objective(c) for c in population if not c.feasible)}")

FIGURES.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter([c.radar_count for c in population], [objective(c) for c in population],
           c=[c.uncovered for c in population], cmap="viridis", alpha=0.45)
ax.set(xlabel="radars installed", ylabel="objective",
       title="The penalty orders coverage before economy")
fig.tight_layout()
fig.savefig(FIGURES / "radar_03_the_penalty.png", dpi=160)
plt.close(fig)
