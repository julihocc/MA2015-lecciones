"""
Lesson 09 - Knapsack 3: Repair creates selection pressure
==========================================================
NEW IN THIS STEP: repair() and a before/after experiment.

CHANGES FROM knapsack_02_random_population.py
Introduce them in this order:
    1. repair()              remove weak value-per-weight choices until legal
    2. the comparison        measure what repair fixes and what it discards

Run it:  python knapsack_03_repair.py

Repair makes 1,000/1,000 feasible and changes 767 chromosomes.
"""
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import random
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt

SEED = 63
CAPACITY = 20
POPULATION_SIZE = 1000
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
    """
    return (sum(bit * item.weight for bit, item in zip(bits, ITEMS)),
            sum(bit * item.value for bit, item in zip(bits, ITEMS)))


def exact_optimum() -> Tuple[List[int], int, int]:
    """Enumerate every feasible packing.

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


@dataclass(frozen=True)
class Candidate:
    """One packing, with weight, value and a feasibility test kept together.

    Args:
        bits: length-12 chromosome.
        weight: sum of selected item weights.
        value: sum of selected item values.

    Example:
        Repair makes 1,000/1,000 of these feasible and changes 767 of them.
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


def make_candidate(bits: List[int]) -> Candidate:
    """Score a chromosome without repairing it.

    Args:
        bits: length-12 vector of 0/1 decisions.

    Returns:
        A ``Candidate`` whose weight may still exceed capacity.
    """
    weight, value = totals(bits)
    return Candidate(bits, weight, value)


def random_candidate() -> Candidate:
    """Sample every decision independently, with no capacity check.

    Returns:
        A ``Candidate`` that may or may not be feasible.

    Example:
        Step 2 found 233 of 1,000 such draws feasible; repair below
        raises that to 1,000/1,000.
    """
    return make_candidate([random.choice((0, 1)) for _ in ITEMS])


# --- NEW (1) repair() ---------------------------------------------------------
def repair(candidate: Candidate) -> Candidate:
    """Drop selected items with the weakest value per unit of weight.

    Args:
        candidate: a packing that may exceed capacity.

    Returns:
        A feasible packing. Already-legal chromosomes are returned unchanged.

    Example:
        1,000/1,000 become feasible; 767 chromosomes actually change.
    """
    bits = candidate.bits.copy()
    order = sorted((i for i, bit in enumerate(bits) if bit),
                   key=lambda i: (ITEMS[i].value / ITEMS[i].weight, ITEMS[i].value))
    for index in order:
        if totals(bits)[0] <= CAPACITY:
            break
        bits[index] = 0
    return make_candidate(bits)
# ------------------------------------------------------------------------------


# --- NEW (2) the comparison ---------------------------------------------------
random.seed(SEED)
raw = [random_candidate() for _ in range(POPULATION_SIZE)]
repaired = [repair(candidate) for candidate in raw]
changed = sum(before.bits != after.bits for before, after in zip(raw, repaired))
value_removed = [before.value - after.value for before, after in zip(raw, repaired)]
# ------------------------------------------------------------------------------

_, exact_weight, exact_value = exact_optimum()
print("Lesson 09 - Knapsack 3: repair")
print(f"Feasible before repair: {sum(c.feasible for c in raw)}/{POPULATION_SIZE}")
print(f"Feasible after repair:  {sum(c.feasible for c in repaired)}/{POPULATION_SIZE}")
print(f"Chromosomes changed:    {changed}/{POPULATION_SIZE}")
print(f"Mean value removed:     {sum(value_removed) / POPULATION_SIZE:.2f}")
print(f"Best repaired value:    {max(c.value for c in repaired)}")
print(f"Exact reference:        value={exact_value}, weight={exact_weight}")

FIGURES.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter([c.weight for c in raw], [c.value for c in raw], alpha=0.25,
           s=18, label="before repair")
ax.scatter([c.weight for c in repaired], [c.value for c in repaired], alpha=0.35,
           s=18, label="after repair")
ax.axvline(CAPACITY, color="tab:red", linewidth=2, label="capacity")
ax.set(xlabel="weight", ylabel="value", title="Repair restores feasibility but changes the sample")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "knapsack_03_repair.png", dpi=160)
plt.close(fig)
