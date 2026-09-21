"""
Lesson 02 - Step 1: The flow, named
====================================
NEW IN THIS STEP: evolve_one_generation(), run(), and a per-phase trace.

Lesson 01 finished with a genetic algorithm that worked. Its loop body, though,
was a run of unnamed comment blocks, and you cannot discuss "the flow" when the
flow has no name and no boundary. This step names it.

Nothing below changes the algorithm. One generation of evolution becomes one
function, the stopping condition moves to one place, and the driver prints what
each phase handed to the next. The proof that it is the same algorithm is in the
output: same seed, same answer as Lesson 01's step 6, to the digit.

CHANGES FROM Lesson 01's first_example_06_the_full_loop.py
Introduce them in this order:
    1. evolve_one_generation()  one generation as one function; it also counts what it did
    2. run()                    the driver: INITIALISE, then evolve until the stop condition
    3. the phase trace          print the census each generation, and check against Lesson 01

Run it:  python ga_flow_01_the_five_phases.py

evolve_one_generation() is one SELECT-CROSSOVER-MUTATE-replace cycle and
returns a census of what it created. run() initialises and repeats that cycle
for MAX_GENERATIONS. Same seed as Lesson 01 step 6, same answer: x = +1.372, f
= +0.706, at a true cost of 107 evaluations rather than the naive 110.
"""
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

SEED = 52
POPULATION_SIZE = 10
CROSSOVER_PROBABILITY = 0.8
MUTATION_PROBABILITY = 0.1
MAX_GENERATIONS = 10
GENE_MIN, GENE_MAX = -10.0, 10.0
TOURNAMENT_SIZE, BLEND_ALPHA = 3, 1.0
MUTATION_MU, MUTATION_SIGMA = 0.0, 1.0
FIGURES = Path(__file__).resolve().parent.parent / "figures"

# Lesson 01's step 6 printed this, with this seed. It is the reference value
# this refactor has to reproduce; it is not used for anything else.
LESSON_01_STEP_6_GENE = +1.372


def objective(x: float) -> float:
    """The function we want to MAXIMISE.

    Args:
        x: a real gene, typically in [-10, 10].

    Returns:
        sin(x) - 0.2 * |x|. Larger is better.

    Example:
        The global peak of this landscape is near x = +1.372, f = +0.706.
    """
    return np.sin(x) - 0.2 * abs(x)


def clamp(g: float, low: float = GENE_MIN, high: float = GENE_MAX) -> float:
    """Keep a gene inside the landscape after blend or Gaussian noise.

    Args:
        g: raw gene (float).
        low, high: closed bounds of the search interval.

    Returns:
        g projected onto [low, high].

    Example:
        BLX-alpha with BLEND_ALPHA = 1.0 can emit any real; this
        landscape only exists on [-10, 10].
    """
    return max(low, min(high, g))


class Individual:
    """One candidate solution: a chromosome plus the fitness it produces.

    Fitness is evaluated here, in the constructor, and nowhere else. That is
    worth noticing: EVALUATE is a phase of the flow with no code of its own,
    and the only way to count evaluations is to count individuals created.


    Example:
        After ten generations with SEED = 52 the sine run's best is
        x = +1.372, f = +0.706.
    """

    def __init__(self, gene_list: List[float]) -> None:
        """Build an individual and score it immediately.

        Args:
            gene_list: the chromosome. One gene in the scalar lessons;
                ten genes once mutation gets a rate dial.

        Example:
            Fitness is scored at construction so it cannot fall out of
            step with the genes.
        """
        self.gene_list = gene_list
        self.fitness = objective(gene_list[0])

    @property
    def gene(self) -> float:
        """The first gene as a float rather than a one-element list.

        Returns:
            The first (or only) gene as a float, so plots and tables do
            not keep writing gene_list[0].
        """
        return self.gene_list[0]

    def __repr__(self) -> str:
        """Compact snapshot used in every printed table of this lesson.

        Returns:
            A one-line snapshot such as 'x=+1.372 f=+0.706' or
            '(+1.372) f=+0.7060', used in every printed table.
        """
        return f"x={self.gene:+.3f} f={self.fitness:+.3f}"


def create_random() -> Individual:
    """Draw one individual uniformly from the search interval.

    Returns:
        An Individual whose gene is Uniform[GENE_MIN, GENE_MAX].

    Example:
        Ten calls with SEED = 52 rebuild Lesson 01's generation 0.
    """
    return Individual([random.uniform(GENE_MIN, GENE_MAX)])


def select_tournament(population: List[Individual], size: int) -> List[Individual]:
    """Fill each next-generation slot with a tournament winner.

    Args:
        population: current Individuals.
        size: candidates per tournament. 3 is Lesson 01's default.

    Returns:
        A list of the same length, holding references into population.

    Example:
        On Lesson 01's SEED = 52 population, 4 of 10 individuals go
        extinct. In Lesson 03, k = 3 gives 2.74 copies of the best,
        k = 5 gives 4.13, k = 10 gives 6.52.
    """
    return [max([random.choice(population) for _ in range(size)],
                key=lambda i: i.fitness) for _ in range(len(population))]


def crossover(p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
    """Blend two parents and return two scored children.

    Args:
        p1, p2: parent Individuals.

    Returns:
        Two children, judged by the same fitness function as p1.

    Example:
        Blend with alpha = 1.0 is the operator that, in Lesson 01, put
        14 of 20 children outside the parents' interval.
    """
    shift = (1 + 2 * BLEND_ALPHA) * random.random() - BLEND_ALPHA
    g1 = clamp((1 - shift) * p1.gene + shift * p2.gene)
    g2 = clamp(shift * p1.gene + (1 - shift) * p2.gene)
    return Individual([g1]), Individual([g2])


def mutate(ind: Individual) -> Individual:
    """Add Gaussian noise and wrap the result as a new Individual.

    Args:
        ind: the parent Individual.

    Returns:
        A new Individual with a clamped Gaussian-mutant gene.

    Example:
        sigma = 1.0 reached the global hill 0 of 2000 times from x = -4.6
        in Lesson 01; the loop still finds x = +1.372 on seed 52.
    """
    return Individual([clamp(ind.gene + random.gauss(MUTATION_MU, MUTATION_SIGMA))])


# --- NEW (1) evolve_one_generation() ------------------------------------------
def evolve_one_generation(
        population: List[Individual]) -> Tuple[List[Individual], Dict[str, int]]:
    """SELECT -> CROSSOVER -> MUTATE -> replace, once.

    Returning the census alongside the new population costs three counters and
    buys the whole lesson: from here on the flow can be watched rather than
    believed. Nothing here decides when to stop - that is the driver's job.

    Args:
        population: current generation.

    Returns:
        The next generation, and in Lesson 02 a census of pairs / crossed
        / mutated / created.

    Example:
        Created counts the true evaluation cost: 107 rather than 110 in
        the Lesson 01-reproducing run, because an un-crossed, un-mutated
        selected individual is the same object.
    """
    census = {"pairs": 0, "crossed": 0, "mutated": 0, "created": 0}

    selected = select_tournament(population, TOURNAMENT_SIZE)            # SELECT

    crossed: List[Individual] = []                                   # CROSSOVER
    for p1, p2 in zip(selected[::2], selected[1::2]):  # consecutive selected pairs
        census["pairs"] += 1
        if random.random() < CROSSOVER_PROBABILITY:
            crossed.extend(crossover(p1, p2))
            census["crossed"] += 1
            census["created"] += 2
        else:
            crossed.extend([p1, p2])

    mutated: List[Individual] = []                                      # MUTATE
    for ind in crossed:
        if random.random() < MUTATION_PROBABILITY:
            mutated.append(mutate(ind))
            census["mutated"] += 1
            census["created"] += 1
        else:
            mutated.append(ind)

    return mutated, census                                             # replace
# ------------------------------------------------------------------------------


# --- NEW (2) run() ------------------------------------------------------------
def run() -> Tuple[List[List[Individual]], List[Dict[str, int]]]:
    """INITIALISE, then evolve until the stopping condition says stop.

    The stopping condition lives here and only here. Right now it is the
    crudest one the book lists - a fixed number of generations - and putting it
    in one visible place is what makes it replaceable later.

    Returns:
        (generations, censuses) for MAX_GENERATIONS of the hard-wired
        sine landscape.

    Example:
        SEED = 52, 10 generations, solution x = +1.372, f = +0.706,
        107 evaluations.
    """
    random.seed(SEED)
    population = [create_random() for _ in range(POPULATION_SIZE)]

    generations = [population]
    censuses: List[Dict[str, int]] = []
    for _ in range(MAX_GENERATIONS):
        population, census = evolve_one_generation(population)
        generations.append(population)
        censuses.append(census)
    return generations, censuses
# ------------------------------------------------------------------------------


generations, censuses = run()

# --- NEW (3) the phase trace --------------------------------------------------
print("The flow, one line per generation. SELECT never creates anything: it")
print("copies references, so only CROSSOVER and MUTATE add new individuals,")
print("and every new individual is exactly one fitness evaluation.\n")
print(" gen | pairs crossed | mutated | new individuals | best fitness")
print("-----+---------------+---------+-----------------+-------------")
best0 = max(generations[0], key=lambda i: i.fitness)
print(f"   0 |             - |       - |"
      f" {POPULATION_SIZE:15d} | {best0.fitness:+.4f}")
for g, census in enumerate(censuses, start=1):
    best = max(generations[g], key=lambda i: i.fitness)
    print(f" {g:3d} |"
          f" {census['crossed']:8d} / {census['pairs']:2d} |"
          f" {census['mutated']:4d}/{POPULATION_SIZE:2d} |"
          f" {census['created']:15d} | {best.fitness:+.4f}")

evaluations = POPULATION_SIZE + sum(c["created"] for c in censuses)
naive = POPULATION_SIZE * (MAX_GENERATIONS + 1)
print(f"\nThis run cost {evaluations} fitness evaluations: one per individual ever")
print(f"created. The obvious guess is {naive} = {POPULATION_SIZE} x "
      f"{MAX_GENERATIONS + 1}, and it is wrong, because an")
print("individual that is selected and then neither crossed nor mutated is the")
print("same object and is never re-evaluated. The true cost of a run is data,")
print("not arithmetic - which is why the census above has to be counted.")

best = max(generations[-1], key=lambda i: i.fitness)
print(f"\nSolution: {best}")
unchanged = abs(best.gene - LESSON_01_STEP_6_GENE) < 5e-4
print(f"Lesson 01 step 6, same seed, reported x={LESSON_01_STEP_6_GENE:+.3f}.")
print("Same seed, same answer: naming the flow changed nothing." if unchanged
      else "The answer moved: this refactor was NOT behaviour-preserving.")
# ------------------------------------------------------------------------------

grid = np.linspace(GENE_MIN, GENE_MAX, 400)
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, (label, pop) in zip(axes, [("generation 0", generations[0]),
                                   (f"generation {MAX_GENERATIONS}", generations[-1])]):
    ax.plot(grid, objective(grid), "--", color="tab:blue", alpha=0.6)
    ax.plot([i.gene for i in pop], [i.fitness for i in pop],
            "o", color="tab:orange", label="population")
    b = max(pop, key=lambda i: i.fitness)
    ax.plot([b.gene], [b.fitness], "s", color="tab:green", markersize=9, label="best")
    ax.set_title(label)
    ax.set_xlabel("x")
    ax.grid(True, linestyle=":", alpha=0.5)
axes[0].set_ylabel("f(x)")
axes[0].legend()
fig.suptitle("The flow takes the population from scattered to settled")
FIGURES.mkdir(exist_ok=True)
fig.savefig(FIGURES / "ga_flow_01_the_five_phases.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved to {FIGURES}/ga_flow_01_the_five_phases.png")

