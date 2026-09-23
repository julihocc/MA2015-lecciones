"""
Lesson 02 - Step 2: The problem becomes an argument
====================================================
NEW IN THIS STEP: the fitness function is injected instead of hard-wired.

In step 1 the word `objective` appears inside Individual. That single reference
welds the flow to one problem: to solve a second one you would copy the file.
The book splits `fitness.py` from `individual.py` for exactly this reason.

So the function moves out and becomes an argument. Each individual carries the
function that judges it, the operators pass it on, and the driver takes the
problem it is asked to solve. Then the same driver is handed two different
problems - one multimodal maximisation, one minimisation written as a
maximisation - and the loop body is not touched between them.

CHANGES FROM ga_flow_01_the_five_phases.py
Introduce them in this order:
    1. the two problems       sine_landscape() and closeness_to_target(), side by side
    2. Individual(genes, f)   an individual carries the function that judges it
    3. the building operators create_random() takes f; crossover() and mutate() inherit it
    4. run(fitness_function)  the driver takes the problem; its body never names one
    5. the two runs           same driver, both problems, nothing in between changed

Run it:  python ga_flow_02_injected_fitness.py

sine_landscape() is Lesson 01's objective; closeness_to_target() is a
minimisation written as a maximisation by flipping the sign. Individual now
carries the function that judges it, so run() can take the problem as an
argument. The sine run still lands at x = +1.372; the target run lands 0.0038
from +4.200.
"""
import random
from pathlib import Path
from typing import Callable, Dict, List, Tuple

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
TARGET = 4.2
FIGURES = Path(__file__).resolve().parent.parent / "figures"

Fitness = Callable[[float], float]


# --- NEW (1) the two problems -------------------------------------------------
def sine_landscape(x: float) -> float:
    """Lesson 01's objective, now just one problem among others.

    Args:
        x: a real gene (or, in later lessons, a one-element gene list).

    Returns:
        sin(x) - 0.2 * |x|. Larger is better.

    Example:
        Lesson 01's global peak is near x = +1.372, f = +0.706; this lesson
        reuses that number as a check that the driver did not change.
    """
    return np.sin(x) - 0.2 * abs(x)


def closeness_to_target(x: float) -> float:
    """MINIMISE the distance to TARGET - written as a maximisation.

    A genetic algorithm always maximises; there is no minimising variant. A
    problem that is naturally a minimisation is injected with its sign flipped,
    and that is the whole of the trick. The best reachable fitness here is 0.

    Args:
        x: a real gene in [-10, 10].

    Returns:
        -(x - 4.2)^2. Best reachable fitness is 0.

    Example:
        The injected-fitness run lands 0.0038 from +4.200.
    """
    return -(x - TARGET) ** 2
# ------------------------------------------------------------------------------


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
    # --- NEW (2) Individual(genes, f) -----------------------------------------
    """One candidate solution: a chromosome, its judge, and the verdict.

    Keeping `fitness_function` on the individual means the operators do not
    need it passed down to them: a child is judged by whatever judged its
    parent. Hard-wiring the function here, as step 1 did, is what made the
    whole file single-purpose.


    Example:
        After ten generations with SEED = 52 the sine run's best is
        x = +1.372, f = +0.706.
    """

    def __init__(self, gene_list: List[float], fitness_function: Fitness) -> None:
        """Build an individual and score it immediately.

        Args:
            gene_list: the chromosome.
            fitness_function: the injected judge. Children inherit it
                from the parent, so operators never name a landscape.
        """
        self.gene_list = gene_list
        self.fitness_function = fitness_function
        self.fitness = fitness_function(gene_list[0])
    # --------------------------------------------------------------------------

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


def select_tournament(population: List[Individual], size: int) -> List[Individual]:
    """Unchanged, and it is worth saying why: selection compares fitness values
    that already exist. It never needs to know where they came from.

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


# --- NEW (3) the building operators -------------------------------------------
def create_random(fitness_function: Fitness) -> Individual:
    """Draw one individual uniformly from the search interval.

    Args:
        fitness_function: the injected judge, sine_landscape or
            closeness_to_target.

    Returns:
        An Individual scored by that function.

    Example:
        The only difference between the two runs of this step is which
        function is passed in here.
    """
    return Individual([random.uniform(GENE_MIN, GENE_MAX)], fitness_function)


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
    return (Individual([g1], p1.fitness_function),
            Individual([g2], p1.fitness_function))


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
    return Individual([clamp(ind.gene + random.gauss(MUTATION_MU, MUTATION_SIGMA))],
                      ind.fitness_function)
# ------------------------------------------------------------------------------


def evolve_one_generation(
        population: List[Individual]) -> Tuple[List[Individual], Dict[str, int]]:
    """SELECT -> CROSSOVER -> MUTATE -> replace, once.

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
    for p1, p2 in zip(selected[::2], selected[1::2]):
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


# --- NEW (4) run(fitness_function) --------------------------------------------
def run(fitness_function: Fitness) -> Tuple[List[List[Individual]],
                                            List[Dict[str, int]]]:
    """INITIALISE, then evolve until the stopping condition says stop.

    Read the body and notice what is missing: no sine, no target, no mention of
    what is being optimised. That absence is the point of the step.

    Args:
        fitness_function: the injected landscape.

    Returns:
        (generations, censuses).

    Example:
        sine_landscape reproduces step 1; closeness_to_target lands
        0.0038 from +4.200.
    """
    random.seed(SEED)
    population = [create_random(fitness_function) for _ in range(POPULATION_SIZE)]

    generations = [population]
    censuses: List[Dict[str, int]] = []
    for _ in range(MAX_GENERATIONS):
        population, census = evolve_one_generation(population)
        generations.append(population)
        censuses.append(census)
    return generations, censuses
# ------------------------------------------------------------------------------


# --- NEW (5) the two runs -----------------------------------------------------
PROBLEMS = [("sine landscape", sine_landscape),
            ("closeness to target", closeness_to_target)]

results = []
for name, fitness_function in PROBLEMS:
    generations, censuses = run(fitness_function)
    best = max(generations[-1], key=lambda i: i.fitness)
    results.append((name, generations, best))
    print(f"{name:>20s}: {MAX_GENERATIONS} generations -> {best}")

sine_best = results[0][2]
target_best = results[1][2]
gap = abs(target_best.gene - TARGET)
print(f"\nThe first run found the sine peak at x={sine_best.gene:+.3f}; the second")
print(f"landed {gap:.4f} away from the target {TARGET:+.3f}. Same seed, same")
print("population size, same operators, same ten generations - and the only")
print("difference between the two runs is which function was passed in.")
print("\nThe sine run also reproduces step 1 exactly: moving the function out")
print("changed the code, not the algorithm. And note what the two landscapes")
print("are: the second has one peak, so every comparison selection makes points")
print("the same way; the first has four. The flow cannot tell them apart.")
# ------------------------------------------------------------------------------

grid = np.linspace(GENE_MIN, GENE_MAX, 400)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, (name, generations, best) in zip(axes, results):
    _, fitness_function = next(p for p in PROBLEMS if p[0] == name)
    pop = generations[-1]
    ax.plot(grid, [fitness_function(v) for v in grid], "--",
            color="tab:blue", alpha=0.6)
    ax.plot([i.gene for i in pop], [i.fitness for i in pop],
            "o", color="tab:orange", label="final population")
    ax.plot([best.gene], [best.fitness], "s", color="tab:green",
            markersize=9, label="best")
    ax.set_title(f"{name}\nbest x={best.gene:+.3f}")
    ax.set_xlabel("x")
    ax.set_ylabel("fitness")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="lower center")
fig.suptitle("One driver, two problems, no change to the loop")
fig.tight_layout(rect=(0, 0, 1, 0.90))
FIGURES.mkdir(exist_ok=True)
fig.savefig(FIGURES / "ga_flow_02_injected_fitness.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved to {FIGURES}/ga_flow_02_injected_fitness.png")

