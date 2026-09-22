"""
Lesson 01 - Step 7: When it fails
=================================
NEW IN THIS STEP: nothing. This is step 6 with ONE character changed.

    SEED = 52  ->  SEED = 16

Same algorithm, same parameters, same number of generations. The seed changes
the entire random sequence, and the resulting run settles
near a local maximum within this ten-generation run.

This is not a bug to be fixed. It is the honest behaviour of the method, and
it is the reason Lessons 03, 04, 05 and 07 exist: selection pressure, operator
choice and parameter tuning are all attempts to make this failure less likely.

CHANGES FROM first_example_06_the_full_loop.py
Introduce them in this order:
    1. SEED                   change 52 to 16; nothing else in the file moves

Run it:  python first_example_07_local_optimum.py
Compare: python first_example_06_the_full_loop.py

Same functions as step 6. Only SEED changes, from 52 to 16. That changes
initialization and every later random decision. The run settles on
x = -4.417, f = +0.073, near a local peak after ten generations.
"""
# random supplies seeded pseudorandom draws.
import random
# Path builds portable output paths.
from pathlib import Path
# Matplotlib draws and saves the evidence figure.
import matplotlib.pyplot as plt
# NumPy evaluates and plots the objective on a dense grid.
import numpy as np

# These settings match step 6 except for SEED. Probabilities are in [0, 1],
# and the population size stays even so adjacent pairing loses no slot.
SEED = 16   # --- CHANGED --- the one and only difference from step 6
POPULATION_SIZE = 10
CROSSOVER_PROBABILITY = 0.8
MUTATION_PROBABILITY = 0.1
MAX_GENERATIONS = 10
GENE_MIN, GENE_MAX = -10.0, 10.0
TOURNAMENT_SIZE, BLEND_ALPHA = 3, 1.0
MUTATION_MU, MUTATION_SIGMA = 0.0, 1.0
# __file__ is this script. resolve() makes its path absolute; one .parent
# reaches src/ and the second reaches the lesson folder. Path / "figures"
# appends a folder name without assuming Windows or POSIX separators.
FIGURES = Path(__file__).resolve().parent.parent / "figures"


def objective(x: float | np.ndarray) -> float | np.ndarray:
    """Score a candidate on the sine landscape of step 1.

    Args:
        x: a real gene, typically in [-10, 10].

    Returns:
        sin(x) - 0.2 * |x|. Larger is better.

    Example:
        This seed's solution is x = -4.417, f = +0.073, not the global
        peak at f = +0.706.
    """
    # NumPy applies sin() element by element when x is an array. Python's
    # abs() likewise works for both one number and a NumPy array here.
    return np.sin(x) - 0.2 * abs(x)


def clamp(g: float, low: float = GENE_MIN, high: float = GENE_MAX) -> float:
    """Keep a gene inside the landscape after blend or Gaussian noise.

    Args:
        g: raw gene (float).
        low, high: closed bounds of the search interval.

    Returns:
        g projected onto [low, high].
    """
    # Nested min/max projects an out-of-bounds proposal to the nearest wall.
    return max(low, min(high, g))


class Individual:
    """One candidate solution: a chromosome plus its fitness.

    Example:
        After ten generations with SEED = 16 the best Individual is
        x = -4.417, f = +0.073.

        """

    def __init__(self, gene_list: list[float]) -> None:
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
    def gene(self) -> float:
        """The single real gene as a float.

        Returns:
            gene_list[0].
        """
        return self.gene_list[0]

    def __repr__(self) -> str:
        """Compact snapshot printed as the run's (failed) solution.

        Returns:
            A string such as 'x=-4.417 f=+0.073'.
        """
        # In this f-string, ``+`` always shows the sign and ``.3f`` keeps
        # three digits after the decimal point.
        return f"x={self.gene:+.3f} f={self.fitness:+.3f}"


def create_random() -> Individual:
    """Draw one individual uniformly from [-10, 10].

    Returns:
        An Individual whose gene is Uniform[GENE_MIN, GENE_MAX].

    Example:
        SEED = 16 produces this run's complete random sequence; the
        resulting run finishes on the local hill.
    """
    # uniform(low, high) draws one real number. Square brackets turn it
    # into this example's one-gene chromosome before it is scored.
    return Individual([random.uniform(GENE_MIN, GENE_MAX)])


def select_tournament(population: list[Individual], size: int) -> list[Individual]:
    """Fill each next-generation slot with the winner of a size-tournament.

    Args:
        population: current Individuals.
        size: candidates per tournament. 3 here.

    Returns:
        A list of the same length, holding references into population.
    """
    # The inner comprehension samples one tournament with replacement.
    # max(..., key=lambda ...) returns its fittest object. The outer
    # comprehension repeats that process once per output slot.
    return [
        max(
            [random.choice(population) for _ in range(size)],
            key=lambda i: i.fitness,
        )
        for _ in range(len(population))
    ]


def crossover(p1: Individual, p2: Individual) -> tuple[Individual, Individual]:
    """Use a complementary blend variant and return two clamped children.

    Args:
        p1, p2: parent Individuals.

    Returns:
        Two new Individuals.

    Example:
        Once parent genes cluster on the local hill, their mixing intervals
        become narrow. In this sampled run, crossover does not create
        an escape.
    """
    # Map a uniform [0, 1) draw to the finite extended mixing interval.
    shift = (1 + 2 * BLEND_ALPHA) * random.random() - BLEND_ALPHA
    g1 = clamp((1 - shift) * p1.gene + shift * p2.gene)
    g2 = clamp(shift * p1.gene + (1 - shift) * p2.gene)
    # Return a tuple of newly scored children; the parents are unchanged.
    return Individual([g1]), Individual([g2])


def mutate(ind: Individual) -> Individual:
    """Add N(0, 1.0) noise to one gene and wrap it as a new Individual.

    Args:
        ind: the parent Individual.

    Returns:
        A new Individual with a clamped mutant gene.

    Example:
        The step-5 sample observed 0 of 2000 arrivals with sigma = 1.0.
        This ten-generation run also finishes near x = -4.417; neither
        observation proves that a longer Gaussian jump is impossible.
    """
    # Draw one perturbation, add it to the gene, enforce the bounds, and
    # construct a new object whose cached fitness matches the gene.
    return Individual([clamp(
        ind.gene + random.gauss(MUTATION_MU, MUTATION_SIGMA)
    )])


# seed() fixes the entire subsequent pseudorandom sequence: initialization,
# tournament samples, crossover decisions, blend shifts, and mutations.
random.seed(SEED)
# The comprehension creates ten independently sampled Individuals.
population = [create_random() for _ in range(POPULATION_SIZE)]
# key= tells max to compare fitness. Store generation 0 as a one-item list.
history = [max(population, key=lambda i: i.fitness).fitness]
print(f"Generation  0 | best {history[0]:+.4f}")

# range stops before its second argument, so +1 includes generation 10.
for generation in range(1, MAX_GENERATIONS + 1):

    offspring = select_tournament(population, TOURNAMENT_SIZE)      # SELECT

    crossed = []                                                     # CROSSOVER
    # [::2] selects slots 0,2,4,... and [1::2] selects 1,3,5,... .
    # zip aligns them as adjacent pairs and unpacking names each parent.
    # The population size is even, so zip does not discard a leftover.
    for p1, p2 in zip(offspring[::2], offspring[1::2]):
        # A fresh uniform draw decides whether this pair crosses. extend
        # adds two separate objects; append would create a nested pair.
        if random.random() < CROSSOVER_PROBABILITY:
            crossed.extend(crossover(p1, p2))
        else:
            crossed.extend([p1, p2])

    # This conditional comprehension makes one mutation decision per
    # candidate. It returns a new Individual when True and the existing
    # reference when False. Assignment replaces the old population list.
    population = [
        mutate(i) if random.random() < MUTATION_PROBABILITY else i
        for i in crossed
    ]                                                               # MUTATE + replace

    # Recompute this generation's best. append adds one scalar fitness
    # to the end of history; there is no protected best-ever archive.
    best = max(population, key=lambda i: i.fitness)
    history.append(best.fitness)
    print(f"Generation {generation:2d} | best {best.fitness:+.4f}")

best = max(population, key=lambda i: i.fitness)
# Leading underscores mark private reporting helpers, not GA state.
# argmax returns the best grid index; float converts the NumPy scalar.
# This is a 400-point reference approximation, not the exact optimum.
_grid = np.linspace(GENE_MIN, GENE_MAX, 400)
_true_x = float(_grid[np.argmax(objective(_grid))])
print(f"\nSolution: {best}")
print(f"Grid reference: x={_true_x:+.3f} f={objective(_true_x):+.3f}   <- never found")
print(f"\nThe run converged in a few generations and then stopped improving.")
print("Every individual sits on the same hill, crossover has little reach, and")
print("this ten-generation random sequence did not produce an escape.")

# range(len(history)) supplies generation numbers 0 through 10. "o-"
# draws both circular observations and lines between them.
plt.figure(figsize=(8, 4))
plt.plot(range(len(history)), history, "o-", color="tab:green")
plt.title("Best fitness per generation")
# The semicolon separates two ordinary statements on one physical line.
plt.xlabel("generation"); plt.ylabel("best f(x)")
# Here alpha is grid-line opacity, not the BLEND_ALPHA crossover parameter.
plt.grid(True, linestyle=":", alpha=0.5)
# Create the folder if needed, save the evidence, and release the figure.
FIGURES.mkdir(exist_ok=True)
plt.savefig(FIGURES / "first_example_07_local_optimum.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Figure saved to {FIGURES}/first_example_07_local_optimum.png")
