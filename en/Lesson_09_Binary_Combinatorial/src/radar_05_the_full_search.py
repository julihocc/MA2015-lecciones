"""
Lesson 09 - Radar 5: The full search
=====================================
NEW IN THIS STEP: exact_optimum(), run(), and the coverage map.

CHANGES FROM radar_04_operators.py
Introduce them in this order:
    1. exact_optimum()       enumerate this small instance for a reference
    2. run()                 repeat selected crossover and mutation with elitism
    3. the coverage map      show the final geometry, not only its score

Run it:  python radar_05_the_full_search.py

GA and exhaustive enumeration both find five radars with full coverage.
"""
from dataclasses import dataclass
from itertools import product
from math import dist
from pathlib import Path
import random
from typing import List, Sequence, Set, Tuple

import matplotlib.pyplot as plt

SEED = 3
RADIUS = 3.2
POPULATION_SIZE = 100
GENERATIONS = 30
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
        ``uncovered * 10 + radar_count``. Five radars with full coverage
        score 5.
    """
    return candidate.uncovered * (len(SITES) + 1) + candidate.radar_count


def tournament(population: List[Candidate], size: int = 3) -> Candidate:
    """Select using the audited objective.

    Args:
        population: scored installation plans.
        size: tournament size. 3 is Lesson 03's default.

    Returns:
        The lowest-objective member of a uniform sample of ``size``.
    """
    return min(random.sample(population, size), key=objective)


def crossover(first: Candidate, second: Candidate) -> Candidate:
    """Exchange site decisions at one cut.

    Args:
        first, second: parent plans.

    Returns:
        One child. Coverage is re-scored, not repaired.
    """
    cut = random.randrange(1, len(SITES))
    return make_candidate(first.bits[:cut] + second.bits[cut:])


def mutate(candidate: Candidate) -> Candidate:
    """Flip one installation decision with probability 0.35.

    Args:
        candidate: a child of crossover.

    Returns:
        A new ``Candidate``, possibly unchanged.
    """
    bits = candidate.bits.copy()
    if random.random() < 0.35:
        index = random.randrange(len(bits)); bits[index] = 1 - bits[index]
    return make_candidate(bits)


# --- NEW (1) exact_optimum() --------------------------------------------------
def exact_optimum() -> Candidate:
    """Enumerate all 512 plans; this instance is small enough.

    Returns:
        The lexicographic minimum: fewest uncovered, then fewest radars.

    Example:
        Five radars with full coverage, matching the GA.
    """
    candidates = (make_candidate(list(bits)) for bits in product((0, 1), repeat=len(SITES)))
    return min(candidates, key=objective)
# ------------------------------------------------------------------------------


# --- NEW (2) run() ------------------------------------------------------------
def run() -> Tuple[Candidate, List[int]]:
    """Thirty generations from seed 3, preserving the best plan.

    Returns:
        ``(best, history)``. History is best-so-far objective.

    Example:
        The GA finds five radars with full coverage, matching enumeration.
    """
    random.seed(SEED)
    population = [random_candidate() for _ in range(POPULATION_SIZE)]
    best = min(population, key=objective)
    history = [objective(best)]
    for _ in range(GENERATIONS):
        children = [best]
        while len(children) < POPULATION_SIZE:
            child = mutate(crossover(tournament(population), tournament(population)))
            children.append(child)
        population = children
        best = min((best, min(population, key=objective)), key=objective)
        history.append(objective(best))
    return best, history
# ------------------------------------------------------------------------------


best, history = run()
reference = exact_optimum()
print("Lesson 09 - Radar 5: the full search")
print(f"GA result: radars={best.radar_count}, uncovered={best.uncovered}, objective={objective(best)}")
print(f"Exact reference: radars={reference.radar_count}, uncovered={reference.uncovered}, "
      f"objective={objective(reference)}")
print(f"GA matches the exact objective: {objective(best) == objective(reference)}")
print(f"Candidate evaluations: {POPULATION_SIZE * (GENERATIONS + 1):,}")

# --- NEW (3) the coverage map -------------------------------------------------
FIGURES.mkdir(exist_ok=True)
fig, (left, right) = plt.subplots(1, 2, figsize=(11, 5))
left.step(range(len(history)), history, where="post")
left.axhline(objective(reference), color="black", linestyle="--")
left.set(xlabel="generation", ylabel="objective", title="Search history")
active_sites = [site for bit, site in zip(best.bits, SITES) if bit]
right.scatter(*zip(*TARGETS), marker="x", s=45, label="targets")
right.scatter(*zip(*active_sites), marker="^", s=100, label="selected radars")
for x, y in active_sites:
    right.add_patch(plt.Circle((x, y), RADIUS, fill=False, alpha=0.25))
right.set(xlim=(-1, 11), ylim=(-1, 11), aspect="equal", title="Final coverage geometry")
right.legend()
fig.tight_layout()
fig.savefig(FIGURES / "radar_05_the_full_search.png", dpi=160)
plt.close(fig)
# ------------------------------------------------------------------------------

