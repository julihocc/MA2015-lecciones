"""
Lesson 01 - Step 2: A population
=================================
NEW IN THIS STEP: the Individual class and a random population.

A genetic algorithm does not track one candidate solution, it tracks many.
Here we simply scatter ten of them at random and look at where they land.

CHANGES FROM first_example_01_the_landscape.py
Introduce them in this order:
    1. Individual             a candidate solution: genes plus the fitness they produce
    2. create_random()        draw one individual uniformly from the search space
    3. the population         build ten of them and plot where they landed

Run it:  python first_example_02_random_population.py

Individual stores one x and the f(x) scored at birth. create_random() draws
that x uniformly from [-10, 10]; ten of them are the starting population, and
none of them is expected to sit on the global peak.
"""
# random supplies seeded pseudorandom draws.
import random
# Path builds portable output paths.
from pathlib import Path
# Matplotlib draws and saves the evidence figure.
import matplotlib.pyplot as plt
# NumPy evaluates and plots the objective on a dense grid.
import numpy as np

# Reusing the seed resets Python's pseudorandom sequence, which makes the
# same program produce the same teaching example on every run.
SEED = 52
POPULATION_SIZE = 10
GENE_MIN, GENE_MAX = -10.0, 10.0
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


# --- NEW (1) Individual -------------------------------------------------------
class Individual:
    """One candidate solution: a chromosome plus its fitness.

    Fitness is computed once when the individual is created. This lesson's
    operators build new objects, so gene and cached fitness stay aligned.

    Example:
        Ten Individuals drawn with SEED = 52 are the starting population;
        the best of those ten is still nowhere near x = +1.372.

        """

    def __init__(self, gene_list):
        """Build an individual and score it immediately.

        Args:
            gene_list: a one-element list holding the real gene x.

        Returns:
            None. Stores gene_list and fitness on the instance.
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
        """The single real gene, as a float rather than a one-element list.

        Returns:
            gene_list[0].
        """
        return self.gene_list[0]

    def __repr__(self):
        """Compact snapshot used in every printed table of this lesson.

        Returns:
            A string such as 'x=+1.372 f=+0.706'.
        """
        # In this f-string, ``+`` always shows the sign and ``.3f`` keeps
        # three digits after the decimal point.
        return f"x={self.gene:+.3f} f={self.fitness:+.3f}"
# ------------------------------------------------------------------------------


# --- NEW (2) create_random() --------------------------------------------------
def create_random():
    """Draw one individual uniformly from the closed search interval.

    Uniform is the honest prior: before any fitness is seen, every x in
    [-10, 10] is equally plausible.

    Returns:
        An Individual whose gene is Uniform[GENE_MIN, GENE_MAX].

    Example:
        Ten calls with SEED = 52 produce the generation-0 cloud; none of
        those ten sits on the global peak at x = +1.372.
    """
    # uniform(low, high) draws one real number. Square brackets turn it
    # into this example's one-gene chromosome before it is scored.
    return Individual([random.uniform(GENE_MIN, GENE_MAX)])
# ------------------------------------------------------------------------------


# --- NEW (3) the population ---------------------------------------------------
# Reset before the first draw. The list comprehension calls create_random()
# ten times; _ names a loop counter whose value is intentionally unused.
random.seed(SEED)
population = [create_random() for _ in range(POPULATION_SIZE)]

print("Initial population (ten blind guesses):")
for ind in population:
    print("   ", ind)

# max(..., key=...) compares the value returned by the key function.
# The short lambda returns each individual's fitness for that comparison.
best = max(population, key=lambda i: i.fitness)
print(f"\nBest of the initial population: {best}")

# Build a smooth reference curve; the GA does not search these grid points.
grid = np.linspace(GENE_MIN, GENE_MAX, 400)
# Create the . The next calls add one layer at a time.
plt.figure(figsize=(8, 4))
plt.plot(grid, objective(grid), "--", color="tab:blue", alpha=0.6)
# The two comprehensions extract matching x and y coordinates. "o" asks
# for unconnected circle markers, and label supplies the legend text.
plt.plot([i.gene for i in population], [i.fitness for i in population],
         "o", color="tab:orange", label="population")
plt.plot([best.gene], [best.fitness], "s", color="tab:green", markersize=9, label="best")
plt.title("Generation 0: random population")
plt.xlabel("x"); plt.ylabel("f(x)")
# Semicolons put two ordinary statements on one line. legend() displays
# the labels above; visual alpha=0.5 makes the dotted grid lighter.
plt.legend(); plt.grid(True, linestyle=":", alpha=0.5)
# Save reproducible evidence and release the .
FIGURES.mkdir(exist_ok=True)
plt.savefig(FIGURES / "first_example_02_random_population.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"\nFigure saved to {FIGURES}/first_example_02_random_population.png")

# ------------------------------------------------------------------------------
