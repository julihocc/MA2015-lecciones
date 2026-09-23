"""
Lesson 02 - Step 3: Measuring the population, not the leader
=============================================================
NEW IN THIS STEP: population_metrics(), the history table and its figure.

Every trace so far reports the best individual. That is the one number a GA is
least honest about: the best can sit still for five generations while the
population underneath it changes completely, or it can be a single lucky outlier
in a population that has learned nothing.

The book lists three characteristics of a population - best individual, best
fitness, average fitness. A fourth is added here, the spread of the genes,
because it answers the question the other three cannot: how much different
genetic material is still in the room. Step 5 is built entirely on that number.

Nothing in the flow changes. run() already keeps every generation it produced,
so the measurement is done afterwards, from the outside.

CHANGES FROM ga_flow_02_injected_fitness.py
Introduce them in this order:
    1. population_metrics()   best, best fitness, average fitness, gene spread
    2. the history table       the four numbers per generation, for both problems
    3. the metrics figure      best against average, with spread on a second axis

Run it:  python ga_flow_03_population_metrics.py

population_metrics() reports best gene, best fitness, average fitness and gene
spread. On the sine problem average fitness climbs from -1.5588 to +0.6228
while gene spread falls from 6.665 to 0.064 by generation 8: the champion
stands still while the population collapses.
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


# --- NEW (1) population_metrics() ---------------------------------------------
def population_metrics(population: List["Individual"]) -> Dict[str, float]:
    """The four numbers that describe a population rather than its champion.

    `spread` is the standard deviation of the genes. It is not in the book, and
    it is the only one of the four that can tell a converged population from a
    merely lucky one: two populations can share a best and an average and have
    nothing else in common.

    Args:
        population: one generation of Individuals.

    Returns:
        Dict with best_gene, best_fitness, average_fitness, spread
        (standard deviation of the genes).

    Example:
        On sine, spread falls from 6.665 to 0.064 by generation 8 while
        best fitness has already arrived.
    """
    genes = [ind.gene for ind in population]
    fitnesses = [ind.fitness for ind in population]
    best = max(population, key=lambda i: i.fitness)
    return {"best_gene": best.gene,
            "best_fitness": best.fitness,
            "average_fitness": sum(fitnesses) / len(fitnesses),
            "spread": float(np.std(genes))}
# ------------------------------------------------------------------------------


class Individual:
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


PROBLEMS = [("sine landscape", sine_landscape),
            ("closeness to target", closeness_to_target)]

results = []
for name, fitness_function in PROBLEMS:
    generations, censuses = run(fitness_function)
    results.append((name, fitness_function, generations))

# --- NEW (2) the history table ------------------------------------------------
for name, fitness_function, generations in results:
    history = [population_metrics(pop) for pop in generations]

    print(f"=== {name} " + "=" * (56 - len(name)))
    print(" gen | best gene | best fitness | average fitness | gene spread")
    print("-----+-----------+--------------+-----------------+------------")
    for g, m in enumerate(history):
        print(f" {g:3d} |  {m['best_gene']:+8.3f} |"
              f"     {m['best_fitness']:+8.4f} |"
              f"        {m['average_fitness']:+8.4f} |"
              f"     {m['spread']:7.3f}")

    first, last = history[0], history[-1]
    improved = [g for g in range(1, len(history))
                if history[g]["best_fitness"] > history[g - 1]["best_fitness"]]
    gap_first = first["best_fitness"] - first["average_fitness"]
    gap_last = last["best_fitness"] - last["average_fitness"]
    spreads = [m["spread"] for m in history]
    thinnest = min(range(len(spreads)), key=lambda g: spreads[g])
    after = [g for g in improved if g > thinnest]

    print(f"\nBest fitness went from {first['best_fitness']:+.4f} to "
          f"{last['best_fitness']:+.4f}, improving in "
          f"{len(improved)} of the {len(history) - 1} generations;")
    print(f"the last improvement was generation "
          f"{max(improved) if improved else 0}.")
    print(f"Average fitness went from {first['average_fitness']:+.4f} to "
          f"{last['average_fitness']:+.4f}, so the gap between the")
    print(f"champion and the crowd closed from {gap_first:.4f} to {gap_last:.4f}.")
    print(f"Gene spread started at {spreads[0]:.3f}, bottomed out at "
          f"{spreads[thinnest]:.3f} in generation {thinnest},")
    print(f"and ended at {spreads[-1]:.3f}.")
    print(f"After generation {thinnest} the best fitness improved "
          f"{len(after)} more time(s).")
    print("Read the spread column on its own: selection and crossover spend it,")
    print("mutation is the only operator that puts any back, and the two are not")
    print("in balance. A population whose spread is near zero has stopped")
    print("searching - it is one individual, copied, and selection has nothing")
    print("left to compare. Lesson 03 measures the pressure that spends it, and")
    print("Lesson 05 is about the operator that can put it back.")
    print()
# ------------------------------------------------------------------------------

# --- NEW (3) the metrics figure -----------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, (name, fitness_function, generations) in zip(axes, results):
    history = [population_metrics(pop) for pop in generations]
    gens = range(len(history))
    ax.plot(gens, [m["best_fitness"] for m in history], "o-",
            color="tab:green", label="best fitness")
    ax.plot(gens, [m["average_fitness"] for m in history], "o-",
            color="tab:orange", label="average fitness")
    ax.set_title(name)
    ax.set_xlabel("generation")
    ax.set_ylabel("fitness")
    ax.grid(True, linestyle=":", alpha=0.5)
    twin = ax.twinx()
    twin.plot(gens, [m["spread"] for m in history], "s--",
              color="tab:grey", label="gene spread")
    twin.set_ylabel("gene spread (std dev)")
    twin.set_ylim(bottom=0)
    lines = ax.get_lines() + twin.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="center right", fontsize=8)
fig.suptitle("Average catches up with best, while the spread collapses - and partly recovers")
fig.tight_layout(rect=(0, 0, 1, 0.90))
FIGURES.mkdir(exist_ok=True)
fig.savefig(FIGURES / "ga_flow_03_population_metrics.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Figure saved to {FIGURES}/ga_flow_03_population_metrics.png")
# ------------------------------------------------------------------------------

