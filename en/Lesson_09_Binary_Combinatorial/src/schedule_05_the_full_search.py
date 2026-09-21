"""
Lesson 09 - Schedule 5: The full search
========================================
NEW IN THIS STEP: one_generation(), run(), and an honest baseline comparison.

CHANGES FROM schedule_04_operators.py
Introduce them in this order:
    1. one_generation()      keep the best roster and produce repaired children
    2. run()                 repeat the process under a fixed evaluation budget
    3. the baseline report   compare the result with repaired random rosters

Run it:  python schedule_05_the_full_search.py

GA cost 41 beats the best of 5,100 repaired-random rosters, cost 52, at
equal evaluation count.
"""
from pathlib import Path
import random
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt

SEED = 15
DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
SHIFTS = ("morning", "day", "evening")
EMPLOYEES = tuple(f"E{i}" for i in range(1, 9))
DEMAND = (2, 2, 1)
MAX_SHIFTS = 5
POPULATION_SIZE = 100
GENERATIONS = 50
FIGURES = Path(__file__).resolve().parent.parent / "figures"
PREFERENCE = tuple(1 + ((employee * 7 + day * 3 + shift * 5) % 5)
                   for employee in range(len(EMPLOYEES))
                   for day in range(len(DAYS)) for shift in range(len(SHIFTS)))


def gene_index(employee: int, day: int, shift: int) -> int:
    """Map a roster coordinate to one binary chromosome position.

    Args:
        employee: 0..7.
        day: 0..6.
        shift: 0..2.

    Returns:
        An index in ``0..167``.
    """
    return employee * len(DAYS) * len(SHIFTS) + day * len(SHIFTS) + shift


def random_roster() -> List[int]:
    """Sample every assignment independently at the mean demand rate.

    Returns:
        A 168-bit roster, repaired before it enters the population.
    """
    probability = sum(DEMAND) / (len(EMPLOYEES) * len(SHIFTS))
    return [int(random.random() < probability) for _ in PREFERENCE]


def violations(bits: Sequence[int]) -> Tuple[int, int]:
    """Keep coverage and workload failures as two visible numbers.

    Args:
        bits: a 168-bit roster.

    Returns:
        ``(coverage, workload)``. The GA champion reports both zero.
    """
    coverage = sum(abs(sum(bits[gene_index(e, day, shift)] for e in range(len(EMPLOYEES))) - required)
                   for day in range(len(DAYS)) for shift, required in enumerate(DEMAND))
    workloads = [sum(bits[gene_index(e, day, shift)] for day in range(len(DAYS))
                     for shift in range(len(SHIFTS))) for e in range(len(EMPLOYEES))]
    return coverage, sum(max(0, work - MAX_SHIFTS) for work in workloads)


def preference_cost(bits: Sequence[int]) -> int:
    """Lower is better; constraints are deliberately absent from this value.

    Args:
        bits: a roster.

    Returns:
        Sum of per-assignment preference costs.

    Example:
        The GA reaches cost 41 against a repaired-random best of 52.
    """
    return sum(bit * cost for bit, cost in zip(bits, PREFERENCE))


def repair(source: Sequence[int]) -> List[int]:
    """Remove excess work, then fill every shortage.

    Args:
        source: a raw 168-bit roster.

    Returns:
        A feasible roster.
    """
    bits = list(source)
    workloads = [sum(bits[gene_index(e, day, shift)] for day in range(len(DAYS))
                     for shift in range(len(SHIFTS))) for e in range(len(EMPLOYEES))]
    for day in range(len(DAYS)):
        for shift, required in enumerate(DEMAND):
            assigned = [e for e in range(len(EMPLOYEES)) if bits[gene_index(e, day, shift)]]
            while len(assigned) > required:
                e = max(assigned, key=lambda x: (PREFERENCE[gene_index(x, day, shift)], workloads[x]))
                bits[gene_index(e, day, shift)] = 0; workloads[e] -= 1; assigned.remove(e)
    for e in range(len(EMPLOYEES)):
        while workloads[e] > MAX_SHIFTS:
            positions = [(day, shift) for day in range(len(DAYS)) for shift in range(len(SHIFTS))
                         if bits[gene_index(e, day, shift)]]
            day, shift = max(positions, key=lambda ds: PREFERENCE[gene_index(e, *ds)])
            bits[gene_index(e, day, shift)] = 0; workloads[e] -= 1
    for day in range(len(DAYS)):
        for shift, required in enumerate(DEMAND):
            assigned = sum(bits[gene_index(e, day, shift)] for e in range(len(EMPLOYEES)))
            while assigned < required:
                choices = [e for e in range(len(EMPLOYEES)) if workloads[e] < MAX_SHIFTS
                           and not bits[gene_index(e, day, shift)]]
                e = min(choices, key=lambda x: (PREFERENCE[gene_index(x, day, shift)], workloads[x]))
                bits[gene_index(e, day, shift)] = 1; workloads[e] += 1; assigned += 1
    return bits


def tournament(population: List[List[int]], size: int = 3) -> List[int]:
    """Prefer lower cost among already legal schedules.

    Args:
        population: repaired rosters.
        size: tournament size. 3 is Lesson 03's default.

    Returns:
        The lowest-cost member of a uniform sample of ``size``.
    """
    return min(random.sample(population, size), key=preference_cost)


def crossover(first: Sequence[int], second: Sequence[int]) -> List[int]:
    """Exchange whole employee rosters, not arbitrary bits.

    Args:
        first, second: repaired parents.

    Returns:
        One child cut on an employee boundary, not yet repaired.
    """
    cut = random.randrange(1, len(EMPLOYEES)) * len(DAYS) * len(SHIFTS)
    return list(first[:cut]) + list(second[cut:])


def mutate(source: Sequence[int]) -> List[int]:
    """Swap one employee assignment before repair.

    Args:
        source: a roster, usually a child of crossover.

    Returns:
        A copy with one of that employee's on-bits moved to an off-bit.
    """
    bits = list(source)
    employee = random.randrange(len(EMPLOYEES))
    assigned = [gene_index(employee, day, shift) for day in range(len(DAYS))
                for shift in range(len(SHIFTS)) if bits[gene_index(employee, day, shift)]]
    unassigned = [gene_index(employee, day, shift) for day in range(len(DAYS))
                  for shift in range(len(SHIFTS)) if not bits[gene_index(employee, day, shift)]]
    bits[random.choice(assigned)] = 0; bits[random.choice(unassigned)] = 1
    return bits


# --- NEW (1) one_generation() -------------------------------------------------
def one_generation(population: List[List[int]], elite: List[int]) -> List[List[int]]:
    """Keep the best roster and produce repaired children.

    Args:
        population: the current feasible generation.
        elite: the best-ever roster, copied through unchanged.

    Returns:
        A new feasible population of the same size.
    """
    children = [elite]
    while len(children) < POPULATION_SIZE:
        child = crossover(tournament(population), tournament(population))
        children.append(repair(mutate(child)))
    return children
# ------------------------------------------------------------------------------


# --- NEW (2) run() ------------------------------------------------------------
def run() -> Tuple[List[int], List[int]]:
    """Fifty generations from seed 15 under a fixed evaluation budget.

    Returns:
        ``(best, history)``. History is best-so-far preference cost.

    Example:
        Cost 41 beats the best of 5,100 repaired-random rosters (cost 52)
        at equal evaluation count.
    """
    random.seed(SEED)
    population = [repair(random_roster()) for _ in range(POPULATION_SIZE)]
    best = min(population, key=preference_cost)
    history = [preference_cost(best)]
    for _ in range(GENERATIONS):
        population = one_generation(population, best)
        best = min((best, min(population, key=preference_cost)), key=preference_cost)
        history.append(preference_cost(best))
    return best, history
# ------------------------------------------------------------------------------


# --- NEW (3) the baseline report ---------------------------------------------
best, history = run()
random.seed(SEED + 1)
baseline = [preference_cost(repair(random_roster()))
            for _ in range(POPULATION_SIZE * (GENERATIONS + 1))]
# ------------------------------------------------------------------------------

print("Lesson 09 - Schedule 5: the full search")
print(f"Best GA preference cost: {preference_cost(best)}")
print(f"Best of {len(baseline):,} repaired random rosters: {min(baseline)}")
print(f"Final roster violations: coverage={violations(best)[0]}, workload={violations(best)[1]}")
print(f"Roster evaluations: {POPULATION_SIZE * (GENERATIONS + 1):,}")

FIGURES.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 4))
ax.step(range(len(history)), history, where="post", label="best GA cost")
ax.axhline(min(baseline), color="black", linestyle="--", label="best repaired random")
ax.set(xlabel="generation", ylabel="preference cost",
       title="Scheduling quality at a visible feasibility boundary")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "schedule_05_the_full_search.png", dpi=160)
plt.close(fig)
