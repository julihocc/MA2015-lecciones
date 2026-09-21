"""
Lesson 11 - Equations 1: A problem whose answer can be checked
===============================================================
NEW IN THIS STEP: f(), g(), w(), total_error(), digits() and the search box.

Every problem so far has been an optimisation: we produced an answer and had no
way of telling whether it was the best one. Lesson 08 made that explicit. A
system of equations is the rare exception. A candidate (x, y, z) is a solution
if and only if all three equations evaluate to exactly zero, and that is a
one-line check, not an opinion.

The genes are integers, so the whole search space is a finite box. Remember that
number: it comes back in step 4.

Run it:  python equations_01_the_system.py

Genes are integers in [-20, 20], so the whole space is 68,921 triples.
Residuals along (y, z) = (1, 1) run from 3 digits at x = 2 to 31 digits at
x = 20.
"""
from math import factorial
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path

FIGURES = Path(__file__).resolve().parent.parent / "figures"

# The genes are clamped to this integer box. The bound is not decoration: the
# exponents below grow with |z| and |x|, so an unbounded gene makes a single
# fitness evaluation take longer than the whole lesson.
BOX_LOW, BOX_HIGH = -20, 20


def f(x: int, y: int, z: int) -> int:
    """First equation of the system; a solution makes it zero.

    Args:
        x, y, z: integers in [-20, 20].

    Returns:
        One integer residual. Zero is exact, not approximate.
    """
    return (x * y + 2) ** 2 + (x + y) ** (abs(z - 2) ** 3 + 2) + x * y * z


def g(x: int, y: int, z: int) -> int:
    """Second equation. The factorial is why z has to stay small.

    Args:
        x, y, z: integers in [-20, 20]. ``factorial(|z-3|)`` is why the
            box cannot be unbounded.

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
        Along ``(y, z) = (1, 1)`` the residual runs from 3 digits at
        ``x = 2`` to 31 digits at ``x = 20``.
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
        Number of decimal digits. Zero residual is reported as 1 digit.

    Example:
        The line ``(y, z) = (1, 1)`` runs from 3 digits to 31.
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


def describe(point: Tuple[int, int, int]) -> str:
    """One line per candidate, in the units the problem is actually stated in.

    Args:
        point: an ``(x, y, z)`` triple.

    Returns:
        A printable residual line, with huge values shown as ``~10^k``.
    """
    error = total_error(*point)
    shown = str(error) if error < 10 ** 9 else f"~10^{digits(error) - 1}"
    return f"  {str(point):<14} |f|+|g|+|w| = {shown:<12} solution: {error == 0}"


side = BOX_HIGH - BOX_LOW + 1
print("Lesson 11 - Equations 1: a problem whose answer can be checked")
print(f"Search box: integers in [{BOX_LOW}, {BOX_HIGH}] for each of x, y, z")
print(f"Box size:   {side} x {side} x {side} = {side ** 3:,} candidate triples")
print("A candidate is a solution exactly when the total residual is 0.")
print("Four candidates picked by hand:")
for candidate in [(0, 0, 0), (1, 1, 1), (-3, 4, 2), (5, -2, 7)]:
    print(describe(candidate))

scan = [(x, total_error(x, 1, 1)) for x in range(BOX_LOW, BOX_HIGH + 1)]
worst = max(scan, key=lambda pair: pair[1])
best = min(scan, key=lambda pair: pair[1])
print(f"Scanning x with (y, z) = (1, 1): the residual ranges from "
      f"{digits(best[1])} digits at x={best[0]} to {digits(worst[1])} digits at x={worst[0]}.")
print("The objective is not smooth and its values are not floats. That matters.")

FIGURES.mkdir(exist_ok=True)
grid = [[digits(total_error(x, y, 0)) for x in range(BOX_LOW, BOX_HIGH + 1)]
        for y in range(BOX_LOW, BOX_HIGH + 1)]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
image = axes[0].imshow(grid, origin="lower", extent=(BOX_LOW, BOX_HIGH, BOX_LOW, BOX_HIGH),
                       cmap="magma")
axes[0].set(xlabel="x", ylabel="y", title="Digits of the residual on the slice z = 0")
fig.colorbar(image, ax=axes[0], label="decimal digits")
axes[1].plot([x for x, _ in scan], [digits(e) for _, e in scan], "o-")
axes[1].set(xlabel="x", ylabel="decimal digits of the residual",
            title="One line through the box, (y, z) = (1, 1)")
fig.tight_layout()
fig.savefig(FIGURES / "equations_01_the_system.png", dpi=160)
plt.close(fig)

