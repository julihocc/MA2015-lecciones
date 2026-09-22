"""
Lesson 01 - Step 1: The problem
================================
Before writing any algorithm, look at what we are searching.

    f(x) = sin(x) - 0.2 * |x|,   x in [-10, 10]

Run it:  python first_example_01_the_landscape.py

objective() is the only function here: it scores a point on the sine
landscape. The rest of the file plots that landscape so the four peaks are
visible before any search is written.
"""
# Path builds operating-system-independent file paths.
from pathlib import Path

# pyplot supplies the plotting commands; the short name plt is conventional.
import matplotlib.pyplot as plt

# NumPy supplies arrays and mathematical functions that act on whole arrays.
import numpy as np

# Uppercase names mark settings that should stay fixed during this run.
# Tuple unpacking assigns the two interval endpoints in one statement.
GENE_MIN, GENE_MAX = -10.0, 10.0

# __file__ is this script. resolve() makes its path absolute; one .parent
# reaches src/ and the second reaches the lesson folder. Path / "figures"
# appends a folder name without assuming Windows or POSIX separators.
FIGURES = Path(__file__).resolve().parent.parent / "figures"


def objective(x: float | np.ndarray) -> float | np.ndarray:
    """Score a candidate on the one-variable landscape this lesson maximises.

    The four peaks are why a climber that only walks uphill can miss the
    global one: it stops at whichever hill it started on.

    Args:
        x: a real number, typically in [-10, 10].

    Returns:
        sin(x) - 0.2 * |x|. Larger is better.

    Example:
        The global peak this grid reports is near x = +1.378, f = +0.706;
        three other peaks sit in the same interval.
    """
    # NumPy applies sin() element by element when x is an array. Python's
    # abs() likewise works for both one number and a NumPy array here.
    return np.sin(x) - 0.2 * abs(x)


# linspace includes 400 evenly spaced values from -10 through +10.
# objective() accepts the whole array, so y contains one score per x.
x = np.linspace(GENE_MIN, GENE_MAX, 400)
y = objective(x)

# Start an 8-by-4-inch , then draw y against x as a blue line.
plt.figure(figsize=(8, 4))
plt.plot(x, y, color="tab:blue")
# The title and axis labels say what the picture measures. The dotted
# grid is drawn at 50% opacity; this visual alpha is unrelated to BLX-alpha.
plt.title("The search space")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.grid(True, linestyle=":", alpha=0.5)
# Create figures/ if needed. exist_ok=True avoids an error when it exists.
# dpi controls image resolution; tight removes unused outside whitespace.
# close() releases the figure because this unattended script never shows it.
FIGURES.mkdir(exist_ok=True)
plt.savefig(
    FIGURES / "first_example_01_the_landscape.png",
    dpi=150,
    bbox_inches="tight",
)
plt.close()

# argmax returns the index of the largest sampled score; x[index] gets
# the matching coordinate. This is a grid approximation, not an exact proof.
best_x = x[np.argmax(y)]
print(f"Global maximum is near x = {best_x:+.3f}, f(x) = {objective(best_x):+.3f}")
print("Note the three other peaks. Any method that only climbs uphill")
print("will stop at whichever one it happens to start on.")
print(f"\nFigure saved to {FIGURES}/first_example_01_the_landscape.png")
