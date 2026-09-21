"""
Lesson 11 - Equations 2: What a random sample is worth here
============================================================
NEW IN THIS STEP: random_triple() and a 400-draw sample of the box.

Before running a genetic algorithm, find out what luck alone buys. In lesson 09
random sampling was a respectable baseline. Here it is not, and the reason is
visible in one histogram: the residual is measured in decimal digits, so a draw
that looks close is still astronomically far away.

CHANGES FROM equations_01_the_system.py
Introduce them in this order:
    1. random_triple()  draw one integer candidate uniformly from the box
    2. the sample       400 draws, and what the best of them is actually worth

Run it:  python equations_02_random_population.py

400 uniform draws (0.58% of the box) find 0 solutions; the best is
(0, 1, 2) with residual 453, while the median draw has a 957-digit residual
and the worst 14,187 digits.
"""
from math import factorial
from pathlib import Path
import random
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 3
FIGURES = Path(__file__).resolve().parent.parent / "figures"
SAMPLE_SIZE = 400

BOX_LOW, BOX_HIGH = -20, 20


def f(x: int, y: int, z: int) -> int:
    """First equation of the system; a solution makes it zero.

    Args:
        x, y, z: integers in [-20, 20].

    Returns:
        One integer residual.
    """
    return (x * y + 2) ** 2 + (x + y) ** (abs(z - 2) ** 3 + 2) + x * y * z


def g(x: int, y: int, z: int) -> int:
    """Second equation. The factorial is why z has to stay small.

    Args:
        x, y, z: integers in [-20, 20].

    Returns:
        One integer residual.
    """
    return x * (y * z + 10) - factorial(abs(z - 3)) + y ** abs(x) + 11 * z


def w(x: int, y: int, z: int) -> int:
    """Third equation.

    Args:
        x, y, z: integers in [-20, 20].

    Returns:
        One integer residual.
    """
    return (x + 7 * y) ** abs(z + x) - (z + 16) ** 2 - 151


def total_error(x: int, y: int, z: int) -> int:
    """Sum of absolute residuals. Exactly zero means an exact solution.

    Args:
        x, y, z: integers in [-20, 20].

    Returns:
        ``|f|+|g|+|w|``, or ``10**100`` if an evaluation overflows.

    Example:
        The best of 400 random draws is residual 453 at ``(0, 1, 2)``.
    """
    try:
        return abs(f(x, y, z)) + abs(g(x, y, z)) + abs(w(x, y, z))
    except (ValueError, ZeroDivisionError):
        return 10 ** 100


def digits(error: int) -> int:
    """Decimal length of the residual, computed from the bit length because
    these integers are routinely too long for str() to convert.

    Args:
        error: a non-negative integer residual.

    Returns:
        Number of decimal digits.

    Example:
        The median of 400 draws is 957 digits; the worst is 14,187.
    """
    if error == 0:
        return 1
    estimate = int(error.bit_length() * 0.30103) + 1
    while 10 ** (estimate - 1) > error:
        estimate -= 1
    while 10 ** estimate <= error:
        estimate += 1
    return estimate


def clamp(value: float) -> int:
    """Genes are integers inside the box; crossover and mutation are not.

    Args:
        value: a proposed real gene.

    Returns:
        The nearest integer in [-20, 20].
    """
    return max(BOX_LOW, min(BOX_HIGH, round(value)))


# --- NEW (1) random_triple() --------------------------------------------------
def random_triple() -> Tuple[int, int, int]:
    """One uniform draw from the box. This is the whole of 'random search'.

    Returns:
        An ``(x, y, z)`` triple with each gene in [-20, 20].

    Example:
        400 draws (0.58% of the box) find 0 solutions.
    """
    return tuple(random.randint(BOX_LOW, BOX_HIGH) for _ in range(3))
# ------------------------------------------------------------------------------


# --- NEW (2) the sample -------------------------------------------------------
random.seed(SEED)
sample: List[Tuple[Tuple[int, int, int], int]] = [
    (triple, total_error(*triple)) for triple in (random_triple() for _ in range(SAMPLE_SIZE))
]
exact = [triple for triple, error in sample if error == 0]
best_triple, best_error = min(sample, key=lambda pair: pair[1])
sizes = sorted(digits(error) for _, error in sample)
median = sizes[len(sizes) // 2]
under_ten = [error for _, error in sample if error < 10]
# ------------------------------------------------------------------------------

side = BOX_HIGH - BOX_LOW + 1
print("Lesson 11 - Equations 2: what a random sample is worth here")
print(f"Seed {SEED}; {SAMPLE_SIZE} uniform draws from a box of {side ** 3:,} triples")
print(f"Exact solutions found:      {len(exact)}")
print(f"Best draw:                  {best_triple} with residual {best_error:,}")
print(f"Residual size, digits:      min {sizes[0]}, median {median}, max {sizes[-1]}")
print(f"Draws with residual < 10:   {len(under_ten)}")
print(f"Fraction of the box seen:   {SAMPLE_SIZE / side ** 3:.2%}")
print("A residual with a median of "
      f"{median} digits is not 'nearly solved'; it is a different number entirely.")

FIGURES.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(sizes, bins=40, color="#4c72b0", edgecolor="white")
ax.axvline(digits(best_error), color="crimson", linestyle="--",
           label=f"best draw: {digits(best_error)} digits")
ax.set(xlabel="decimal digits of the residual", ylabel="draws",
       title=f"{SAMPLE_SIZE} random candidates, none of them a solution")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "equations_02_random_population.png", dpi=160)
plt.close(fig)

