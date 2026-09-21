"""
Lesson 01 - Step 4: Crossover
==============================
NEW IN THIS STEP: crossover_blend() and crossover().

Selection alone can only copy. Crossover is the first operator that creates
values that were not in the population before, by combining two parents.

CHANGES FROM first_example_03_selection.py
Introduce them in this order:
    1. clamp()                extrapolated proposals still need legal search-space bounds
    2. crossover_blend()      the operator itself, working on raw gene values
    3. crossover()            wrap it so it takes and returns Individual objects

Run it:  python first_example_04_crossover.py

clamp() projects a raw gene back onto [-10, 10] because, for fixed parents and
alpha, blend crossover uses a finite extended interval that may exceed the
legal domain. crossover_blend() draws a shift from [-alpha, 1+alpha] and
mixes two parent genes; crossover() wraps that mix so it takes and returns
Individual objects. With alpha = 1.0, 14 of 20 children of the parents at -2.0
and +3.0 fall outside that parental interval.
"""
# random supplies seeded pseudorandom draws.
import random
# NumPy evaluates and plots the objective on a dense grid.
import numpy as np
# Matplotlib draws and saves the evidence figure.
import matplotlib.pyplot as plt
# Path builds portable output paths.
from pathlib import Path

# These uppercase names are run settings. BLEND_ALPHA controls crossover's
# numeric reach; it is unrelated to Matplotlib's visual-opacity argument.
SEED = 52
GENE_MIN, GENE_MAX = -10.0, 10.0
BLEND_ALPHA = 1.0
# __file__ is this script. resolve() makes its path absolute; one .parent
# reaches src/ and the second reaches the lesson folder. Path / "figures"
# appends a folder name without assuming Windows or POSIX separators.
FIGURES = Path(__file__).resolve().parent.parent / "figures"


def objective(x):
    """Score a candidate on the sine landscape of step 1.

    Args:
        x: a real gene, typically in [-10, 10].

    Returns:
        sin(x) - 0.2 * |x|. Larger is better.

    Example:
        The global peak is near x = +1.372, f = +0.706.
    """
    # NumPy applies sin() element by element when x is an array. Python's
    # abs() likewise works for both one number and a NumPy array here.
    return np.sin(x) - 0.2 * abs(x)


# --- NEW (1) clamp() ----------------------------------------------------------
def clamp(gene, low=GENE_MIN, high=GENE_MAX):
    """Keep a gene inside the landscape after blend crossover.

    BLX-alpha can propose a value outside its parents' interval. This
    landscape only exists on [-10, 10], so every proposal must remain legal.

    Args:
        gene: raw gene (float).
        low, high: closed bounds of the search interval.

    Returns:
        gene projected onto [low, high].

    Example:
        After ten crossovers of parents at -2.0 and +3.0 (SEED = 52,
        BLEND_ALPHA = 1.0), 14 of 20 children fall outside [-2, +3];
        clamp is why any of those that would also leave [-10, 10]
        still score as a legal gene.
    """
    # min(high, gene) caps the upper side; max(low, ...) then raises any
    # value below the lower side. Legal values pass through unchanged.
    return max(low, min(high, gene))
# ------------------------------------------------------------------------------


class Individual:
    """One candidate solution: a chromosome plus its fitness.

    Example:
        The demonstration parents are Individual([-2.0]) and
        Individual([+3.0]).

        """

    def __init__(self, gene_list):
        """Build an individual and score it immediately.

        Args:
            gene_list: a one-element list holding the real gene x.
        """
        # ``self`` is the new object. Store the one-element chromosome,
        # then cache its score. The operators create new Individuals rather
        # than editing this list, so the cached fitness stays consistent.
        self.gene_list = gene_list
        self.fitness = objective(gene_list[0])

    # @property lets callers write ``individual.gene`` instead of calling
    # ``individual.gene()``. It hides the one-element list representation.
    @property
    def gene(self):
        """The single real gene as a float.

        Returns:
            gene_list[0].
        """
        return self.gene_list[0]

    def __repr__(self):
        """Compact snapshot used in the child table.

        Returns:
            A string such as 'x=+1.372 f=+0.706'.
        """
        # In this f-string, ``+`` always shows the sign and ``.3f`` keeps
        # three digits after the decimal point.
        return f"x={self.gene:+.3f} f={self.fitness:+.3f}"


# --- NEW (2) crossover_blend() ------------------------------------------------
def crossover_blend(gene1, gene2, alpha):
    """Complementary blend-crossover variant for real-valued genes.

    A random `shift` is drawn from [-alpha, 1+alpha]:

        child1 = (1 - shift) * gene1 + shift * gene2
        child2 = shift * gene1 + (1 - shift) * gene2

    With alpha = 0 the children always land BETWEEN the parents, so crossover
    alone cannot expand that parental interval. With alpha > 0 the sampling
    interval reaches past both parents, although it is still finite and depends
    on their separation. That is why alpha exists.

    Args:
        gene1, gene2: parent gene values (floats).
        alpha: interval extension. 0.0 interpolates only; 1.0 is this step's
            default.

    Returns:
        Two clamped child genes.

    Example:
        With parents at -2.0 and +3.0 and alpha = 1.0, 14 of 20 children
        fall outside [-2, +3]; with alpha = 0 that count would be zero.
    """
    # random() is uniform on [0, 1). Scaling and shifting maps that draw
    # to [-alpha, 1 + alpha), the finite extended mixing interval.
    shift = (1 + 2 * alpha) * random.random() - alpha
    # The weights in each expression sum to one. Swapping them makes two
    # complementary children from the same random shift.
    child1 = (1 - shift) * gene1 + shift * gene2
    child2 = shift * gene1 + (1 - shift) * gene2
    # A comma returns a two-value tuple. Each raw proposal is bounded first.
    return clamp(child1), clamp(child2)
# ------------------------------------------------------------------------------


# --- NEW (3) crossover() ------------------------------------------------------
def crossover(parent1, parent2):
    """Blend two Individuals and return two scored children.

    Args:
        parent1, parent2: Individuals whose `.gene` values are blended.

    Returns:
        A pair of new Individual objects.

    Example:
        Ten calls on the parents at -2.0 and +3.0 (SEED = 52) produce
        the 14-of-20-outside count this step is built around.
    """
    # Tuple unpacking gives a name to each returned gene. Constructing new
    # Individuals immediately computes the two matching fitness values.
    g1, g2 = crossover_blend(parent1.gene, parent2.gene, BLEND_ALPHA)
    return Individual([g1]), Individual([g2])
# ------------------------------------------------------------------------------


# Reset the pseudorandom sequence immediately before the first crossover draw.
random.seed(SEED)
# Fixed parents isolate crossover from population selection. Unary + on 3.0
# changes no value; it visually emphasizes the positive endpoint.
mother = Individual([-2.0])
father = Individual([+3.0])

print(f"Parents:   {mother}   and   {father}")
print(f"\nTen crossovers of the SAME two parents (alpha = {BLEND_ALPHA}):\n")
outside = 0
children = []
for i in range(1, 11):
    # Unpack the returned pair. extend adds both objects individually;
    # append([c1, c2]) would instead add one nested list.
    c1, c2 = crossover(mother, father)
    children.extend([c1, c2])
    for child in (c1, c2):
        # Python supports chained comparisons. ``not`` flips inside to
        # outside. Booleans count as 1/0, so += increments only when True.
        outside_parent_interval = not (mother.gene <= child.gene <= father.gene)
        outside += outside_parent_interval
        marker = "   <- outside the parents" if outside_parent_interval else ""
        print(f"  {i:2d}. {child}{marker}")

print(f"\n{outside} of 20 children fell outside the interval [-2, +3].")
print("With alpha = 0 that number would be zero: crossover alone could not")
print("expand this parental interval. Try it: set BLEND_ALPHA = 0.0")

lo, hi = mother.gene, father.gene
# Complementary filtering comprehensions separate the same children for
# two marker styles; neither comprehension changes a child.
inside = [c for c in children if lo <= c.gene <= hi]
outside_children = [c for c in children if not (lo <= c.gene <= hi)]

# The dense grid is only the background curve; the child objects hold the data
# produced by crossover.
x = np.linspace(GENE_MIN, GENE_MAX, 400)
y = objective(x)
# Draw the objective first, shade the parent interval, and then add parents and
# both child groups with different marker shapes.
plt.figure(figsize=(8, 4))
plt.plot(x, y, color="tab:blue", alpha=0.35)
# axvspan shades an x interval. Here alpha means 20% visual opacity; it is
# unrelated to the numeric BLEND_ALPHA used by crossover.
plt.axvspan(lo, hi, color="gray", alpha=0.2, label="parents' interval [-2, +3]")
plt.plot([mother.gene, father.gene], [mother.fitness, father.fitness],
         "D", color="black", label="parents", markersize=8, zorder=4)
plt.plot([c.gene for c in inside], [c.fitness for c in inside],
         "o", color="tab:orange", label="children inside", zorder=3)
plt.plot([c.gene for c in outside_children], [c.fitness for c in outside_children],
         "o", color="tab:red", label="children outside", zorder=3)
plt.title(f"Blend crossover (alpha = {BLEND_ALPHA})")
plt.xlabel("x"); plt.ylabel("f(x)")
plt.legend(); plt.grid(True, linestyle=":", alpha=0.5)
# Save the completed layered plot and close its .
FIGURES.mkdir(exist_ok=True)
plt.savefig(FIGURES / "first_example_04_crossover.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"\nFigure saved to {FIGURES}/first_example_04_crossover.png")
