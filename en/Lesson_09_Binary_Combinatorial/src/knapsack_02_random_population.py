"""
Lesson 09 - Knapsack 2: A random population
============================================
NEW IN THIS STEP: Candidate, random_candidate(), and a feasibility census.

CHANGES FROM knapsack_01_the_instance.py
Introduce them in this order:
    1. Candidate             keep chromosome, weight, value and feasibility together
    2. random_candidate()    sample every decision independently
    3. POPULATION_SIZE       make the size of the experiment explicit
    4. the census            measure feasibility before designing selection

Run it:  python knapsack_02_random_population.py

233 of 1,000 random chromosomes are feasible; the best random value is 47.
"""
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import random
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt

SEED = 63
CAPACITY = 20
FIGURES = Path(__file__).resolve().parent.parent / "figures"


@dataclass(frozen=True)
class Item:
    """One knapsack object: a name, a weight, and a value.

    Args:
        name: label used only in printed solutions.
        weight: capacity consumed if the corresponding bit is 1.
        value: objective gained if the corresponding bit is 1.
    """
    name: str
    weight: int
    value: int


ITEMS = [
    Item("A", 2, 6), Item("B", 3, 8), Item("C", 4, 9),
    Item("D", 5, 12), Item("E", 6, 13), Item("F", 7, 14),
    Item("G", 3, 7), Item("H", 8, 16), Item("I", 1, 2),
    Item("J", 9, 18), Item("K", 4, 10), Item("L", 2, 5),
]


def totals(bits: Sequence[int]) -> Tuple[int, int]:
    """Return weight and value for one binary chromosome.

    Args:
        bits: length-12 vector of 0/1 decisions.

    Returns:
        ``(weight, value)``. Feasibility is ``weight <= 20``.

    Example:
        The exact optimum is still value 50 at weight 20; a random draw
        need not be feasible.
    """
    weight = sum(bit * item.weight for bit, item in zip(bits, ITEMS))
    value = sum(bit * item.value for bit, item in zip(bits, ITEMS))
    return weight, value


def exact_optimum() -> Tuple[List[int], int, int]:
    """Enumerate every feasible packing; the GA is not running yet.

    Returns:
        ``(bits, weight, value)`` of the exact optimum: value 50 at weight 20.
    """
    feasible = []
    for bits in product((0, 1), repeat=len(ITEMS)):
        weight, value = totals(bits)
        if weight <= CAPACITY:
            feasible.append((value, -weight, list(bits)))
    value, negative_weight, bits = max(feasible)
    return bits, -negative_weight, value


# --- NEW (1) Candidate --------------------------------------------------------
@dataclass(frozen=True)
class Candidate:
    """One packing, with weight, value and a feasibility test kept together.

    Args:
        bits: length-12 chromosome.
        weight: sum of selected item weights.
        value: sum of selected item values.

    Example:
        233 of 1,000 random candidates are feasible; the best of those
        scores 47 against the exact 50.
    """
    bits: List[int]
    weight: int
    value: int

    @property
    def feasible(self) -> bool:
        """Whether this packing respects the capacity of 20.

        Returns:
            True iff ``weight <= 20``.
        """
        return self.weight <= CAPACITY
# ------------------------------------------------------------------------------


# --- NEW (2) random_candidate() -----------------------------------------------
def random_candidate() -> Candidate:
    """Sample every decision independently, with no capacity check.

    Returns:
        A ``Candidate`` that may or may not be feasible.

    Example:
        233 of 1,000 such draws fit in the knapsack; the best feasible
        random value is 47.
    """
    bits = [random.choice((0, 1)) for _ in ITEMS]
    weight, value = totals(bits)
    return Candidate(bits, weight, value)
# ------------------------------------------------------------------------------


# --- NEW (3) POPULATION_SIZE --------------------------------------------------
POPULATION_SIZE = 1000
# ------------------------------------------------------------------------------


# --- NEW (4) the census -------------------------------------------------------
random.seed(SEED)
population = [random_candidate() for _ in range(POPULATION_SIZE)]
feasible = [candidate for candidate in population if candidate.feasible]
best_random = max(feasible, key=lambda candidate: candidate.value)
# ------------------------------------------------------------------------------

_, exact_weight, exact_value = exact_optimum()
print("Lesson 09 - Knapsack 2: a random population")
print(f"Feasible random chromosomes: {len(feasible)}/{POPULATION_SIZE} "
      f"({len(feasible) / POPULATION_SIZE:.1%})")
print(f"Best feasible random value: {best_random.value} at weight {best_random.weight}")
print(f"Exact reference: value={exact_value}, weight={exact_weight}")

FIGURES.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist([candidate.weight for candidate in population], bins=range(0, 61, 3),
        color="tab:blue", alpha=0.8)
ax.axvline(CAPACITY, color="tab:red", linewidth=2, label="capacity")
ax.set(xlabel="chromosome weight", ylabel="count",
       title="Independent random bits usually exceed the capacity")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "knapsack_02_random_population.png", dpi=160)
plt.close(fig)
