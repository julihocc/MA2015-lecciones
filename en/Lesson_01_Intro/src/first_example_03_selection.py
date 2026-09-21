"""
Lesson 01 - Step 3: Selection
==============================
NEW IN THIS STEP: select_tournament().

The population from step 2 is random and mostly bad. Selection is the first
force of evolution: it decides who gets to reproduce. Nothing new is created
here - we only choose, with repetition, from what already exists.

CHANGES FROM first_example_02_random_population.py
Introduce them in this order:
    1. select_tournament()    the first force of evolution: who gets to reproduce

Run it:  python first_example_03_selection.py

select_tournament() fills each slot of the next generation by drawing
TOURNAMENT_SIZE = 3 individuals at random and keeping the fittest. It copies
references, so a strong individual can occupy several slots and a weak one can
occupy none — 4 of these 10 go extinct.
"""
# random supplies seeded pseudorandom draws.
import random
# Counter counts how often each selected object occurs.
from collections import Counter
# Path builds portable output paths.
from pathlib import Path
# Matplotlib draws and saves the evidence figure.
import matplotlib.pyplot as plt
# NumPy evaluates and plots the objective on a dense grid.
import numpy as np

# These uppercase names are run settings. Keep them together so an experiment
# can change a parameter without hunting through the algorithm.
SEED = 52
POPULATION_SIZE = 10
GENE_MIN, GENE_MAX = -10.0, 10.0
TOURNAMENT_SIZE = 3
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


class Individual:
    """One candidate solution: a chromosome plus its fitness.

    Fitness is computed once, at construction, so it cannot fall out of
    step with the genes.

    Example:
        The ten Individuals of SEED = 52 are the same starting population
        as step 2; selection will copy 6 of them and drop 4.

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
        """Compact snapshot used in the before/after table.

        Returns:
            A string such as 'x=+1.372 f=+0.706'.
        """
        # In this f-string, ``+`` always shows the sign and ``.3f`` keeps
        # three digits after the decimal point.
        return f"x={self.gene:+.3f} f={self.fitness:+.3f}"


def create_random():
    """Draw one individual uniformly from [-10, 10].

    Returns:
        An Individual whose gene is Uniform[GENE_MIN, GENE_MAX].

    Example:
        Ten calls with SEED = 52 rebuild the generation-0 cloud of step 2.
    """
    # uniform(low, high) draws one real number. Square brackets turn it
    # into this example's one-gene chromosome before it is scored.
    return Individual([random.uniform(GENE_MIN, GENE_MAX)])


# --- NEW (1) select_tournament() ----------------------------------------------
def select_tournament(population, size):
    """Run one tournament per slot in the new generation.

    Each tournament draws `size` individuals at random and keeps the fittest.
    A strong individual wins several tournaments and appears several times;
    a weak one probably never appears. `size` alone controls how greedy the
    selection is.

    Args:
        population: the current list of Individuals.
        size: candidates per tournament. 3 is the course default.

    Returns:
        A new list of the same length, holding references into population.
        No new x value is created.

    Example:
        On the SEED = 52 population of ten, 4 of 10 individuals are
        selected zero times — they are extinct.
    """
    # Start empty, then append one tournament winner for every output slot.
    offspring = []
    for _ in range(len(population)):
        # choice() samples with replacement. This comprehension runs it
        # ``size`` times, so one object may enter the tournament repeatedly.
        candidates = [random.choice(population) for _ in range(size)]
        # max uses fitness as its comparison key. append adds that one
        # existing object as the next selected slot; it does not copy it.
        offspring.append(max(candidates, key=lambda ind: ind.fitness))
    return offspring
# ------------------------------------------------------------------------------


# Resetting before both initialization and selection reproduces the exact
# sequence of population draws and tournament draws used in step 2.
random.seed(SEED)
population = [create_random() for _ in range(POPULATION_SIZE)]
offspring = select_tournament(population, TOURNAMENT_SIZE)

# sorted normally goes low to high. Negating fitness makes high values sort
# first without changing the individuals themselves.
print("BEFORE selection:")
for ind in sorted(population, key=lambda i: -i.fitness):
    print("   ", ind)

print("\nAFTER selection (same individuals, different multiplicities):")
# id(ind) identifies the object itself. A generator expression supplies one
# identity per selected slot; Counter turns them into identity -> count pairs.
counts = Counter(id(ind) for ind in offspring)
# Reuse the same high-to-low ordering so the before/after tables align.
for ind in sorted(population, key=lambda i: -i.fitness):
    # get(key, 0) returns zero when an identity never won a tournament.
    n = counts.get(id(ind), 0)
    print(f"    {ind}   selected {n} time(s)  {'#' * n}")

# This filtering comprehension keeps only original individuals with no wins.
extinct = [ind for ind in population if counts.get(id(ind), 0) == 0]
print("\nNotice: no new x value appeared. Selection only copies.")
print(f"{len(extinct)} of {POPULATION_SIZE} individuals were selected zero times.")
print("They are extinct here: no selected copy remains to reproduce directly.")

# Sort low-to-high so the horizontal order is meaningful, then build one
# fitness, multiplicity, and conditional color for every plotted individual.
ranked = sorted(population, key=lambda i: i.fitness)
xs = [ind.fitness for ind in ranked]
ys = [counts.get(id(ind), 0) for ind in ranked]
colors = ["tab:red" if n == 0 else "tab:blue" for n in ys]

# Build the figure in layers: vertical stems, observed markers, then two empty
# marker sets that exist only to explain the colors in the legend.
plt.figure(figsize=(8, 4))
# vlines draws each count from zero upward. zorder puts these stems behind
# the markers drawn next. The two empty scatter calls create legend keys.
plt.vlines(xs, 0, ys, colors=colors, linewidth=2, zorder=2)
plt.scatter(xs, ys, c=colors, s=90, zorder=3, edgecolors="white", linewidths=0.6)
plt.scatter([], [], color="tab:red", s=90, label="extinct (0 copies)")
plt.scatter([], [], color="tab:blue", s=90, label="selected")
# Labels make the stored PNG understandable without reading the program.
plt.title("Selection only copies; the red dots are gone")
plt.xlabel("Fitness f(x)")
plt.ylabel("Times selected (multiplicity)")
plt.yticks(range(0, max(ys) + 2))
plt.legend()
plt.grid(True, linestyle=":", alpha=0.5, zorder=0)
# Save at 150 dots per inch with tight outside margins, then release memory.
FIGURES.mkdir(exist_ok=True)
plt.savefig(FIGURES / "first_example_03_selection.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"\nFigure saved to {FIGURES}/first_example_03_selection.png")

