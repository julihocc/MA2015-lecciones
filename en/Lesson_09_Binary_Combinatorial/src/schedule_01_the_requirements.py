"""
Lesson 09 - Schedule 1: The requirements
=========================================
NEW IN THIS STEP: a binary roster, explicit demand, and employee capacity.

One bit assigns one employee to one shift. The instance needs 35 assignments
across 21 shifts, while eight employees can cover at most five shifts each.

Run it:  python schedule_01_the_requirements.py

168 bits encode 35 required assignments under capacity 40.
"""
from pathlib import Path

import matplotlib.pyplot as plt

SEED = 15
DAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
SHIFTS = ("morning", "day", "evening")
EMPLOYEES = tuple(f"E{i}" for i in range(1, 9))
DEMAND = (2, 2, 1)
MAX_SHIFTS = 5
FIGURES = Path(__file__).resolve().parent.parent / "figures"


def gene_index(employee: int, day: int, shift: int) -> int:
    """Map a roster coordinate to one binary chromosome position.

    Args:
        employee: 0..7, the eight people on the roster.
        day: 0..6, Monday through Sunday.
        shift: 0..2, morning / day / evening.

    Returns:
        An index in ``0..167``. Employee is the slowest-moving axis, so a
        whole person is a contiguous block of 21 bits.

    Example:
        8 x 7 x 3 = 168 bits encode 35 required assignments under
        capacity 40.
    """
    return employee * len(DAYS) * len(SHIFTS) + day * len(SHIFTS) + shift


chromosome_length = len(EMPLOYEES) * len(DAYS) * len(SHIFTS)
required_assignments = len(DAYS) * sum(DEMAND)
available_capacity = len(EMPLOYEES) * MAX_SHIFTS

print("Lesson 09 - Schedule 1: the requirements")
print(f"Chromosome: {len(EMPLOYEES)} employees x {len(DAYS)} days x "
      f"{len(SHIFTS)} shifts = {chromosome_length} bits")
print(f"Required assignments: {required_assignments}")
print(f"Available capacity:    {available_capacity}")
print(f"Capacity margin:       {available_capacity - required_assignments}")

FIGURES.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 4))
matrix = [list(DEMAND) for _ in DAYS]
image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(DEMAND))
ax.set_xticks(range(len(SHIFTS)), SHIFTS)
ax.set_yticks(range(len(DAYS)), DAYS)
for day in range(len(DAYS)):
    for shift in range(len(SHIFTS)):
        ax.text(shift, day, DEMAND[shift], ha="center", va="center")
ax.set_title("Employees required in each shift")
fig.colorbar(image, ax=ax, label="employees")
fig.tight_layout()
fig.savefig(FIGURES / "schedule_01_the_requirements.png", dpi=160)
plt.close(fig)
