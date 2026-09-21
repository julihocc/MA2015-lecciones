"""
Lesson 09 - Schedule 2: Random rosters
=======================================
NEW IN THIS STEP: random_roster(), violations(), and a feasibility census.

CHANGES FROM schedule_01_the_requirements.py
Introduce them in this order:
    1. random_roster()       sample every assignment independently
    2. violations()          keep coverage and workload failures visible
    3. the census            measure the failure before assigning a fitness

Run it:  python schedule_02_random_rosters.py

0 of 500 random rosters are feasible; the best still has 14 violations.
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


def gene_index(employee: int, day: int, shift: int) -> int:
    """Map a roster coordinate to one binary chromosome position.

    Args:
        employee: 0..7.
        day: 0..6.
        shift: 0..2.

    Returns:
        An index in ``0..167``. Employee is the slowest-moving axis.
    """
    return employee * len(DAYS) * len(SHIFTS) + day * len(SHIFTS) + shift


# --- NEW (1) random_roster() --------------------------------------------------
def random_roster() -> List[int]:
    """Sample every assignment independently at the mean demand rate.

    Returns:
        A 168-bit roster. Independence does not respect coverage or
        workload, so feasibility is not expected.

    Example:
        0 of 500 such rosters are feasible; the best still has 14
        violations.
    """
    probability = sum(DEMAND) / (len(EMPLOYEES) * len(SHIFTS))
    return [int(random.random() < probability)
            for _ in range(len(EMPLOYEES) * len(DAYS) * len(SHIFTS))]
# ------------------------------------------------------------------------------


# --- NEW (2) violations() -----------------------------------------------------
def violations(bits: Sequence[int]) -> Tuple[int, int]:
    """Keep coverage and workload failures as two visible numbers.

    Args:
        bits: a 168-bit roster, legal or not.

    Returns:
        ``(coverage, workload)``. Coverage is the L1 gap to demand;
        workload is excess shifts beyond five per employee. Zero and
        zero is the only legal roster.

    Example:
        The best of 500 random rosters still has 14 violations; none
        are feasible.
    """
    coverage = 0
    for day in range(len(DAYS)):
        for shift, required in enumerate(DEMAND):
            assigned = sum(bits[gene_index(employee, day, shift)]
                           for employee in range(len(EMPLOYEES)))
            coverage += abs(assigned - required)
    workload = 0
    for employee in range(len(EMPLOYEES)):
        assigned = sum(bits[gene_index(employee, day, shift)]
                       for day in range(len(DAYS)) for shift in range(len(SHIFTS)))
        workload += max(0, assigned - MAX_SHIFTS)
    return coverage, workload
# ------------------------------------------------------------------------------


# --- NEW (3) the census -------------------------------------------------------
random.seed(SEED)
population = [random_roster() for _ in range(POPULATION_SIZE)]
counts = [sum(violations(bits)) for bits in population]
feasible = sum(count == 0 for count in counts)
# ------------------------------------------------------------------------------

print("Lesson 09 - Schedule 2: random rosters")
print(f"Feasible random rosters: {feasible}/{POPULATION_SIZE}")
print(f"Median violations: {sorted(counts)[len(counts) // 2]}")
print(f"Best random violation count: {min(counts)}")

FIGURES.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(counts, bins=range(min(counts), max(counts) + 2), color="tab:blue")
ax.set(xlabel="coverage plus workload violations", ylabel="rosters",
       title="Independent bits rarely form a legal roster")
fig.tight_layout()
fig.savefig(FIGURES / "schedule_02_random_rosters.png", dpi=160)
plt.close(fig)
