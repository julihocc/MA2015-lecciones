"""
Lesson 09 - Schedule 3: Repair the constraints
===============================================
NEW IN THIS STEP: preference_cost(), repair(), and a before/after census.

CHANGES FROM schedule_02_random_rosters.py
Introduce them in this order:
    1. preference_cost()     separate schedule quality from legality
    2. repair()              remove excess work, then fill every shortage
    3. the comparison        prove the stored population is feasible

Run it:  python schedule_03_repair.py

Repair makes 500/500 rosters feasible; repaired costs range from 58
upward.
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
POPULATION_SIZE = 500
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
        A 168-bit roster that is almost never legal.

    Example:
        Step 2 found 0 of 500 feasible; repair below raises that to
        500/500.
    """
    probability = sum(DEMAND) / (len(EMPLOYEES) * len(SHIFTS))
    return [int(random.random() < probability) for _ in PREFERENCE]


def violations(bits: Sequence[int]) -> Tuple[int, int]:
    """Keep coverage and workload failures as two visible numbers.

    Args:
        bits: a 168-bit roster.

    Returns:
        ``(coverage, workload)``. Both zero after a successful repair.
    """
    coverage = 0
    for day in range(len(DAYS)):
        for shift, required in enumerate(DEMAND):
            assigned = sum(bits[gene_index(employee, day, shift)]
                           for employee in range(len(EMPLOYEES)))
            coverage += abs(assigned - required)
    workloads = [sum(bits[gene_index(employee, day, shift)]
                     for day in range(len(DAYS)) for shift in range(len(SHIFTS)))
                 for employee in range(len(EMPLOYEES))]
    return coverage, sum(max(0, work - MAX_SHIFTS) for work in workloads)


# --- NEW (1) preference_cost() ------------------------------------------------
def preference_cost(bits: Sequence[int]) -> int:
    """Lower is better; constraints are deliberately absent from this value.

    Args:
        bits: a roster. Quality is scored whether or not it is legal.

    Returns:
        Sum of per-assignment preference costs. Independent of demand.

    Example:
        After repair the stored costs range from 58 upward; legality is
        no longer mixed into this number.
    """
    return sum(bit * cost for bit, cost in zip(bits, PREFERENCE))
# ------------------------------------------------------------------------------


# --- NEW (2) repair() ---------------------------------------------------------
def repair(source: Sequence[int]) -> List[int]:
    """Remove excess work, then fill every shortage, preferring cheaper slots.

    Args:
        source: a raw 168-bit roster.

    Returns:
        A feasible roster. Demand is met and no employee exceeds five shifts.

    Example:
        500/500 become feasible; repaired costs range from 58 upward.
    """
    bits = list(source)
    workloads = [sum(bits[gene_index(employee, day, shift)]
                     for day in range(len(DAYS)) for shift in range(len(SHIFTS)))
                 for employee in range(len(EMPLOYEES))]
    for day in range(len(DAYS)):
        for shift, required in enumerate(DEMAND):
            assigned = [e for e in range(len(EMPLOYEES)) if bits[gene_index(e, day, shift)]]
            while len(assigned) > required:
                employee = max(assigned, key=lambda e: (PREFERENCE[gene_index(e, day, shift)], workloads[e]))
                bits[gene_index(employee, day, shift)] = 0
                workloads[employee] -= 1
                assigned.remove(employee)
    for employee in range(len(EMPLOYEES)):
        while workloads[employee] > MAX_SHIFTS:
            positions = [(day, shift) for day in range(len(DAYS)) for shift in range(len(SHIFTS))
                         if bits[gene_index(employee, day, shift)]]
            day, shift = max(positions, key=lambda ds: PREFERENCE[gene_index(employee, *ds)])
            bits[gene_index(employee, day, shift)] = 0
            workloads[employee] -= 1
    for day in range(len(DAYS)):
        for shift, required in enumerate(DEMAND):
            assigned = sum(bits[gene_index(e, day, shift)] for e in range(len(EMPLOYEES)))
            while assigned < required:
                choices = [e for e in range(len(EMPLOYEES))
                           if workloads[e] < MAX_SHIFTS and not bits[gene_index(e, day, shift)]]
                employee = min(choices, key=lambda e: (PREFERENCE[gene_index(e, day, shift)], workloads[e]))
                bits[gene_index(employee, day, shift)] = 1
                workloads[employee] += 1
                assigned += 1
    return bits
# ------------------------------------------------------------------------------


# --- NEW (3) the comparison ---------------------------------------------------
random.seed(SEED)
raw = [random_roster() for _ in range(POPULATION_SIZE)]
repaired = [repair(bits) for bits in raw]
before = [sum(violations(bits)) for bits in raw]
after = [sum(violations(bits)) for bits in repaired]
# ------------------------------------------------------------------------------

print("Lesson 09 - Schedule 3: repair")
print(f"Feasible before repair: {sum(v == 0 for v in before)}/{POPULATION_SIZE}")
print(f"Feasible after repair:  {sum(v == 0 for v in after)}/{POPULATION_SIZE}")
print(f"Preference cost after repair: min={min(map(preference_cost, repaired))}, "
      f"mean={sum(map(preference_cost, repaired)) / POPULATION_SIZE:.1f}")

FIGURES.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist([preference_cost(bits) for bits in repaired], color="tab:green")
ax.set(xlabel="preference cost", ylabel="feasible rosters",
       title="Repair makes legality an invariant; cost remains to optimise")
fig.tight_layout()
fig.savefig(FIGURES / "schedule_03_repair.png", dpi=160)
plt.close(fig)
