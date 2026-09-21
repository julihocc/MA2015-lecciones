"""
Lesson 09 - Knapsack 1: The instance
=====================================
NEW IN THIS STEP: a fixed binary decision problem and its exact reference.

Each bit answers one question: take this item or leave it. Before introducing a
GA, enumerate this deliberately small instance so later runs have an honest
answer to compare against.

Run it:  python knapsack_01_the_instance.py

The 12-item instance has 4,096 chromosomes; the exact optimum is value 50
at weight 20.
"""
from dataclasses import dataclass
from itertools import product
from pathlib import Path
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

    Example:
        Twelve items; the exact optimum is value 50 at weight 20.
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
        bits: length-12 vector of 0/1 decisions, one per item.

    Returns:
        ``(weight, value)``. Feasibility is ``weight <= 20``, not a return.

    Example:
        The enumerated optimum scores value 50 at weight 20.
    """
    weight = sum(bit * item.weight for bit, item in zip(bits, ITEMS))
    value = sum(bit * item.value for bit, item in zip(bits, ITEMS))
    return weight, value


def exact_optimum() -> Tuple[List[int], int, int]:
    """Enumerate 2^12 chromosomes; this is a reference, not the GA.

    Returns:
        ``(bits, weight, value)`` of the feasible chromosome with the
        highest value, breaking ties toward lower weight.

    Example:
        4,096 chromosomes; the exact optimum is value 50 at weight 20.
    """
    feasible = []
    for bits in product((0, 1), repeat=len(ITEMS)):
        weight, value = totals(bits)
        if weight <= CAPACITY:
            feasible.append((value, -weight, list(bits)))
    value, negative_weight, bits = max(feasible)
    return bits, -negative_weight, value


best_bits, best_weight, best_value = exact_optimum()
chosen = [item.name for bit, item in zip(best_bits, ITEMS) if bit]

print("Lesson 09 - Knapsack 1: the instance")
print(f"Items: {len(ITEMS)}; binary search space: 2^{len(ITEMS)} = {2 ** len(ITEMS):,}")
print(f"Capacity: {CAPACITY}")
print(f"Exact reference: value={best_value}, weight={best_weight}, items={chosen}")

FIGURES.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter([item.weight for item in ITEMS], [item.value for item in ITEMS], s=70)
for item in ITEMS:
    ax.annotate(item.name, (item.weight, item.value), xytext=(4, 4),
                textcoords="offset points")
ax.set(xlabel="weight", ylabel="value", title="Knapsack items: value versus weight")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(FIGURES / "knapsack_01_the_instance.png", dpi=160)
plt.close(fig)
