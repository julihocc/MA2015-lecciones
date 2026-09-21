"""
Lesson 09 - Knapsack 4: Selection and crossover
================================================
NEW IN THIS STEP: tournament(), crossover(), and one_generation().

CHANGES FROM knapsack_03_repair.py
Introduce them in this order:
    1. tournament()          prefer value only after feasibility is guaranteed
    2. crossover()           recombine two legal binary decisions at one cut
    3. one_generation()      expose the select-cross-repair pipeline

Run it:  python knapsack_04_selection_and_crossover.py

One generation raises mean value from 39.00 to 40.73; all stored children
remain feasible.
"""
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import random
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt

SEED = 63
CAPACITY = 20
POPULATION_SIZE = 100
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
    """One packing, scored and stored after repair.

    Args:
        bits: length-12 chromosome.
        weight: sum of selected item weights.
        value: sum of selected item values.

    Example:
        After one generation the mean value is 40.73 and every stored
        child is feasible.
    """
    bits: List[int]
    weight: int
    value: int


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
    """Sample every decision independently, then let ``repair`` legalise it.

    Returns:
        An unrepaired ``Candidate``.
    """
    return make_candidate([random.choice((0, 1)) for _ in ITEMS])


def repair(candidate: Candidate) -> Candidate:
    """Drop weakest value-per-weight items until the packing is legal.

    Args:
        candidate: a packing that may exceed capacity.

    Returns:
        A feasible packing. Already-legal chromosomes are unchanged.

    Example:
        Step 3 changed 767 of 1,000 chromosomes; here every stored child
        is repaired before it enters the next generation.
    """
    bits = candidate.bits.copy()
    order = sorted((i for i, bit in enumerate(bits) if bit),
                   key=lambda i: (ITEMS[i].value / ITEMS[i].weight, ITEMS[i].value))
    for index in order:
        if totals(bits)[0] <= CAPACITY:
            break
        bits[index] = 0
    return make_candidate(bits)


# --- NEW (1) tournament() -----------------------------------------------------
def tournament(population: List[Candidate], size: int = 3) -> Candidate:
    """Prefer value only after feasibility is already guaranteed.

    Args:
        population: repaired candidates; every member is legal.
        size: tournament size. 3 is Lesson 03's default.

    Returns:
        The highest-value member of a uniform sample of ``size``.
    """
    return max(random.sample(population, size), key=lambda candidate: candidate.value)
# ------------------------------------------------------------------------------


# --- NEW (2) crossover() ------------------------------------------------------
def crossover(parent_a: Candidate, parent_b: Candidate) -> Tuple[Candidate, Candidate]:
    """One-point swap of two legal binary decisions.

    Args:
        parent_a, parent_b: repaired parents.

    Returns:
        Two children, not yet repaired. One-point crossover can break
        the capacity that repair had restored.

    Example:
        One generation still stores only feasible children after repair,
        and the mean value rises from 39.00 to 40.73.
    """
    cut = random.randrange(1, len(ITEMS))
    first = make_candidate(parent_a.bits[:cut] + parent_b.bits[cut:])
    second = make_candidate(parent_b.bits[:cut] + parent_a.bits[cut:])
    return first, second
# ------------------------------------------------------------------------------


# --- NEW (3) one_generation() -------------------------------------------------
def one_generation(population: List[Candidate]) -> Tuple[List[Candidate], int]:
    """Select, cross, repair; return the new population and a repair census.

    Args:
        population: the current feasible generation.

    Returns:
        ``(children, infeasible_before_repair)``. Stored children are all
        feasible; the census counts how many raw crossovers were not.

    Example:
        Mean value rises from 39.00 to 40.73; all stored children remain
        feasible.
    """
    children: List[Candidate] = []
    infeasible_before_repair = 0
    while len(children) < len(population):
        raw_children = crossover(tournament(population), tournament(population))
        infeasible_before_repair += sum(child.weight > CAPACITY for child in raw_children)
        children.extend(repair(child) for child in raw_children)
    return children[:len(population)], infeasible_before_repair
# ------------------------------------------------------------------------------


random.seed(SEED)
population = [repair(random_candidate()) for _ in range(POPULATION_SIZE)]
before = [candidate.value for candidate in population]
population, illegal_children = one_generation(population)
after = [candidate.value for candidate in population]
_, _, exact_value = exact_optimum()

print("Lesson 09 - Knapsack 4: selection and crossover")
print(f"Mean value before: {sum(before) / len(before):.2f}")
print(f"Mean value after:  {sum(after) / len(after):.2f}")
print(f"Crossover produced {illegal_children} overweight children before repair.")
print(f"All {len(population)} stored children are feasible after repair.")
print(f"Best after one generation: {max(after)}; exact reference: {exact_value}")

FIGURES.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(before, bins=range(0, exact_value + 3, 3), alpha=0.55, label="before")
ax.hist(after, bins=range(0, exact_value + 3, 3), alpha=0.55, label="after")
ax.axvline(exact_value, color="black", linestyle="--", label="exact optimum")
ax.set(xlabel="feasible value", ylabel="count",
       title="One selected generation shifts value without losing legality")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "knapsack_04_selection_and_crossover.png", dpi=160)
plt.close(fig)
