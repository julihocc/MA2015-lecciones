"""
Lesson 01 - Step 5: Mutation
=============================
NEW IN THIS STEP: mutate_gaussian() and mutate().

Crossover's reach shrinks when parent genes cluster closely on one hill.
Mutation adds an independent displacement, so it can still propose a distant
gene even when both parents are identical.

CHANGES FROM first_example_04_crossover.py
Introduce them in this order:
    1. mutate_gaussian()      add an independent random displacement to one gene
    2. mutate()               wrap it for Individual objects

Run it:  python first_example_05_mutation.py

mutate_gaussian() adds N(mu, sigma) noise to a gene and clamp() keeps the
result on [-10, 10]. mutate() wraps that for Individual objects. From a
population trapped at x = -4.6, this seeded sample observes 0 arrivals in
2000 jumps with sigma = 1.0 and 110 arrivals with sigma = 3.0. Zero
observations do not mean that a Gaussian jump is mathematically impossible.
"""
# random supplies seeded pseudorandom draws.
import random
# Path builds portable output paths.
from pathlib import Path
# Matplotlib draws and saves the evidence figure.
import matplotlib.pyplot as plt
# NumPy evaluates and plots the objective on a dense grid.
import numpy as np

# These uppercase names are settings shared by every part of this run.
SEED = 52
GENE_MIN, GENE_MAX = -10.0, 10.0
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
        The local hill this step is trapped on peaks near f = +0.073;
        the global hill peaks near f = +0.706.
    """
    # NumPy applies sin() element by element when x is an array. Python's
    # abs() likewise works for both one number and a NumPy array here.
    return np.sin(x) - 0.2 * abs(x)


def clamp(gene: float, low: float = GENE_MIN, high: float = GENE_MAX) -> float:
    """Keep a mutated gene inside the landscape.

    Gaussian noise has unbounded support, so a jump can leave [-10, 10].

    Args:
        gene: raw gene (float).
        low, high: closed bounds of the search interval.

    Returns:
        gene projected onto [low, high].

    Example:
        At sigma = 6.0 later lessons show the walls eating proposals;
        here sigma is 1.0 or 3.0, still large enough to need the clip.
    """
    # Clamp upper values first and lower values second. Gaussian noise has
    # unbounded support, so this boundary rule is part of the algorithm.
    return max(low, min(high, gene))


class Individual:
    """One candidate solution: a chromosome plus its fitness.

    Example:
        The trapped population is six Individuals clustered around x = -4.6.

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
        """Compact snapshot used in the trapped-population listing.

        Returns:
            A string such as 'x=-4.417 f=+0.073'.
        """
        # In this f-string, ``+`` always shows the sign and ``.3f`` keeps
        # three digits after the decimal point.
        return f"x={self.gene:+.3f} f={self.fitness:+.3f}"


# --- NEW (1) mutate_gaussian() ------------------------------------------------
def mutate_gaussian(gene: float, mu: float, sigma: float) -> float:
    """Add noise drawn from N(mu, sigma) to a gene.

    sigma controls the distribution's typical step size. A small sigma makes
    a long jump rare, not impossible; a very large sigma makes local refinement
    harder and sends more proposals to the boundary rule.

    Args:
        gene: current gene value.
        mu: mean of the noise. 0.0 here, so the jump is unbiased.
        sigma: standard deviation of the jump.

    Returns:
        The clamped mutant gene.

    Example:
        From x = -4.6, sigma = 1.0 reaches the global hill 0 of 2000
        times; sigma = 3.0 reaches it 110 of 2000 times.
    """
    # gauss draws noise with the requested mean and standard deviation.
    # Add it to the current gene, then enforce the legal interval.
    return clamp(gene + random.gauss(mu, sigma))
# ------------------------------------------------------------------------------


# --- NEW (2) mutate() ---------------------------------------------------------
def mutate(individual: Individual) -> Individual:
    """Apply Gaussian mutation to an Individual and return a new one.

    Args:
        individual: the parent Individual.

    Returns:
        A new Individual whose gene is the mutated, clamped value.

    Example:
        mutate() is the wrapper the generational loop of step 6 will call;
        the 0-of-2000 / 110-of-2000 counts below call mutate_gaussian()
        directly so the experiment is one jump, not a whole generation.
    """
    # Wrap the changed gene in a one-element chromosome; construction scores it.
    return Individual([mutate_gaussian(
        individual.gene, MUTATION_MU, MUTATION_SIGMA
    )])
# ------------------------------------------------------------------------------


# Reset before building the trapped population. Later resets deliberately make
# the two sigma regimes use corresponding underlying Gaussian draws.
random.seed(SEED)

# A population trapped on the same local hill, as happens late in a run.
# The hill near x = -4.6 peaks at f = +0.073; the global hill is at x = +1.38,
# six units away, and reaches f = +0.706.
TRAPPED_AT = -4.6
GLOBAL_HILL = 1.38

# Build six separate candidates in a narrow band around -4.6. The list
# comprehension repeats the draw and constructor six times.
trapped = [
    Individual([TRAPPED_AT + random.uniform(-0.05, 0.05)])
    for _ in range(6)
]
print("A converged population, all on the same local hill:")
for ind in trapped:
    print("   ", ind)

# Index 0 and 1 select the first two candidates; tuple unpacking names them.
# Their arithmetic mean is a simple example of recombination between them.
a, b = trapped[0], trapped[1]
mid = (a.gene + b.gene) / 2
print(f"\nAveraging two of them stays on the hill:")
print(f"    a child near the average lands at x={mid:+.3f}, f={objective(mid):+.3f}")
print("    Closely clustered parents give crossover little reach.")

# ----- experiment: how far can one mutation reach? ----------------------------
print(f"\nThe global hill is {abs(GLOBAL_HILL - TRAPPED_AT):.1f} units away.")
print("How often does a SINGLE mutation get there?\n")

# A tuple supplies the two regimes. Resetting inside the outer loop makes
# both regimes begin from the same underlying standard-normal draws.
for sigma in (1.0, 3.0):
    random.seed(SEED)
    arrivals = 0
    trials = 2000
    for _ in range(trials):
        m = mutate_gaussian(TRAPPED_AT, MUTATION_MU, sigma)
        # This operational definition counts a target neighbourhood, not
        # an exact optimum. abs measures distance along the x-axis.
        if abs(m - GLOBAL_HILL) < 1.5:
            arrivals += 1
    print(f"    sigma = {sigma}:  {arrivals:4d} of {trials} mutations "
          f"({100*arrivals/trials:5.1f}%) land on the global hill")

print("\nsigma controls the typical reach of mutation. In this finite sample,")
print("sigma = 1.0 produced no arrivals; that does not make a longer Gaussian")
print("jump impossible. A very large sigma also makes local refinement harder.")
print("\nOne attempt does not determine the whole run. A GA creates thousands")
print("of mutation attempts across many generations, increasing its chances")
print("of observing a rare jump. Step 6 supplies that repeated process.")

def gaussian_pdf(grid: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Density of N(mu, sigma), used only to shade the reach in the figure.

    Args:
        grid: x values to evaluate on.
        mu, sigma: mean and standard deviation of the mutation kernel.

    Returns:
        The Gaussian density at each point of grid.

    Example:
        The two kernels drawn from TRAPPED_AT = -4.6 with sigma 1.0 and
        3.0; only the wider one overlaps the global-hill band enough to
        produce the 110-of-2000 arrivals.
    """
    # This is the normal-distribution formula evaluated element by element.
    # ** 2 squares each standardized distance; exp makes the bell shape.
    # Dividing by sigma*sqrt(2*pi) normalizes the total area to one.
    return (
        np.exp(-0.5 * ((grid - mu) / sigma) ** 2)
        / (sigma * np.sqrt(2.0 * np.pi))
    )


# Prepare three aligned arrays: objective scores plus the two probability
# densities. These arrays are for explanation, not inputs to the mutation test.
x = np.linspace(GENE_MIN, GENE_MAX, 400)
y = objective(x)
pdf1 = gaussian_pdf(x, TRAPPED_AT, 1.0)
pdf3 = gaussian_pdf(x, TRAPPED_AT, 3.0)

# subplots returns two objects: fig owns the  and ax owns the axes.
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, y, color="tab:blue", label="objective f(x)")
ax.plot([TRAPPED_AT], [objective(TRAPPED_AT)], "D", color="black",
        markersize=8, zorder=4, label="trapped here")
ax.axvline(TRAPPED_AT, color="gray", linestyle="--", alpha=0.5)
ax.axvspan(GLOBAL_HILL - 1.5, GLOBAL_HILL + 1.5, color="tab:green", alpha=0.12)
ax.plot([GLOBAL_HILL], [objective(GLOBAL_HILL)], "*", color="tab:green",
        markersize=14, zorder=4, label="global hill")

# twinx adds a second y-axis sharing the same x-axis. Fitness uses ax; the
# probability densities use ax2 because their vertical units differ.
ax2 = ax.twinx()
ax2.fill_between(x, pdf1, color="#A65300", alpha=0.35, label="sigma = 1.0")
ax2.plot(x, pdf1, color="#A65300", linewidth=1.5)
ax2.fill_between(x, pdf3, color="tab:red", alpha=0.18, label="sigma = 3.0")
ax2.plot(x, pdf3, color="tab:red", linewidth=1.5)
# Leave headroom above the taller density and hide its numeric ticks because
# the plot compares reach and overlap rather than exact density values.
ax2.set_ylim(0, pdf1.max() * 2.2)
ax2.set_yticks([])

# Each axis tracks only its own legend entries. Retrieve both pairs, join
# the lists with +, and display one combined legend on the first axis.
handles, labels = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(handles + h2, labels + l2, loc="upper left")
ax.set_title("Mutation reach: escaping a local optimum")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.grid(True, linestyle=":", alpha=0.5)
# Save after both axes are complete. close() closes the current figure.
FIGURES.mkdir(exist_ok=True)
plt.savefig(FIGURES / "first_example_05_mutation.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"\nFigure saved to {FIGURES}/first_example_05_mutation.png")
