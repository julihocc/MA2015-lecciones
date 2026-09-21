"""
Lesson 09 - Knapsack 5: The full search
========================================
NEW IN THIS STEP: mutate(), run(), and the exact-gap report.

CHANGES FROM knapsack_04_selection_and_crossover.py
Introduce them in this order:
    1. mutate()              bit flips restore decisions crossover may have lost
    2. run()                 repeat the generation and preserve the best chromosome
    3. the exact-gap report  compare the heuristic against the enumerated reference

Run it:  python knapsack_05_the_full_search.py

The GA reaches the exact value 50 at weight 20 with zero gap in 4,060
evaluations.
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
GENERATIONS = 40
MUTATION_RATE = 1 / 12
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

    Example:
        The GA below matches this reference with zero gap in 4,060 evaluations.
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
    """Sample every decision independently.

    Returns:
        An unrepaired ``Candidate``.
    """
    return make_candidate([random.choice((0, 1)) for _ in ITEMS])


def repair(candidate: Candidate) -> Candidate:
    """Drop weakest value-per-weight items until the packing is legal.

    Args:
        candidate: a packing that may exceed capacity.

    Returns:
        A feasible packing.
    """
    bits = candidate.bits.copy()
    order = sorted((i for i, bit in enumerate(bits) if bit),
                   key=lambda i: (ITEMS[i].value / ITEMS[i].weight, ITEMS[i].value))
    for index in order:
        if totals(bits)[0] <= CAPACITY:
            break
        bits[index] = 0
    return make_candidate(bits)


def tournament(population: List[Candidate], size: int = 3) -> Candidate:
    """Prefer value only after feasibility is already guaranteed.

    Args:
        population: repaired candidates; every member is legal.
        size: tournament size. 3 is Lesson 03's default.

    Returns:
        The highest-value member of a uniform sample of ``size``.
    """
    return max(random.sample(population, size), key=lambda candidate: candidate.value)


def crossover(parent_a: Candidate, parent_b: Candidate) -> Tuple[Candidate, Candidate]:
    """One-point swap of two legal binary decisions.

    Args:
        parent_a, parent_b: repaired parents.

    Returns:
        Two children, not yet repaired or mutated.
    """
    cut = random.randrange(1, len(ITEMS))
    return (make_candidate(parent_a.bits[:cut] + parent_b.bits[cut:]),
            make_candidate(parent_b.bits[:cut] + parent_a.bits[cut:]))


# --- NEW (1) mutate() ---------------------------------------------------------
def mutate(candidate: Candidate) -> Candidate:
    """Flip each bit independently at rate 1/12, then re-score.

    Args:
        candidate: a child, possibly already illegal from crossover.

    Returns:
        A new ``Candidate``, not yet repaired.

    Example:
        Mutation plus repair still reaches the exact value 50 at weight 20.
    """
    bits = [1 - bit if random.random() < MUTATION_RATE else bit
            for bit in candidate.bits]
    return make_candidate(bits)
# ------------------------------------------------------------------------------


def one_generation(population: List[Candidate], elite: Candidate) -> List[Candidate]:
    """Keep the elite, then fill the rest by crossover, mutation and repair.

    Args:
        population: the current feasible generation.
        elite: the best-ever packing, copied through unchanged.

    Returns:
        A new feasible population of the same size.
    """
    children = [elite]
    while len(children) < len(population):
        pair = crossover(tournament(population), tournament(population))
        children.extend(repair(mutate(child)) for child in pair)
    return children[:len(population)]


# --- NEW (2) run() ------------------------------------------------------------
def run() -> Tuple[Candidate, List[int], int]:
    """Forty generations from seed 63, preserving the best chromosome.

    Returns:
        ``(best, history, evaluations)``. History is best-so-far value.

    Example:
        Value 50 at weight 20, zero gap, in 4,060 evaluations.
    """
    random.seed(SEED)
    population = [repair(random_candidate()) for _ in range(POPULATION_SIZE)]
    best = max(population, key=lambda candidate: candidate.value)
    history = [best.value]
    evaluations = POPULATION_SIZE
    for _ in range(GENERATIONS):
        population = one_generation(population, best)
        evaluations += POPULATION_SIZE - 1
        candidate = max(population, key=lambda individual: individual.value)
        best = max((best, candidate), key=lambda individual: individual.value)
        history.append(best.value)
    return best, history, evaluations
# ------------------------------------------------------------------------------


# --- NEW (3) the exact-gap report --------------------------------------------
best, history, evaluations = run()
exact_bits, exact_weight, exact_value = exact_optimum()
gap = exact_value - best.value
chosen = [item.name for bit, item in zip(best.bits, ITEMS) if bit]
# ------------------------------------------------------------------------------

print("Lesson 09 - Knapsack 5: the full search")
print(f"Best GA solution: value={best.value}, weight={best.weight}, items={chosen}")
print(f"Exact reference: value={exact_value}, weight={exact_weight}")
print(f"Optimality gap: {gap} ({gap / exact_value:.1%})")
print(f"Candidate evaluations: {evaluations:,}")
print(f"Feasible final answer: {best.weight <= CAPACITY}")

FIGURES.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(9, 5))
ax.step(range(len(history)), history, where="post", label="best GA value")
ax.axhline(exact_value, color="black", linestyle="--", label="exact optimum")
ax.set(xlabel="generation", ylabel="value", title="Knapsack search against an exact reference")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "knapsack_05_the_full_search.png", dpi=160)
plt.close(fig)
