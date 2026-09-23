"""
Lesson 02 - Step 5: Stepped landscapes, and a guess that turned out wrong
=========================================================================
NEW IN THIS STEP: a third problem, and a second instrument that can see what the
cost column of step 4 cannot.

Step 4 priced the stopping rule on two smooth landscapes and found it free. The
natural suspicion is that smoothness was doing the work: on a landscape where
progress arrives in jumps instead of slopes, a flat stretch would mean the
population has not yet found the next step, not that the search is over - and a
patience rule cannot tell those apart.

This step builds that landscape to catch the rule out. It does not catch it out.
The run climbs the staircase, reaches its top step, and the rule stops it six
generations later having lost nothing. The suspicion was wrong, and the reason is
worth more than the suspicion was: these steps RISE. Every step boundary is a
real comparison, so selection has a direction to follow even though the surface
is flat between boundaries. Stepped is not the same as directionless.

What this step does deliver is the instrument. "The rule cost nothing" is not the
same claim as "the run succeeded", and a run cannot tell you the difference about
itself. Comparing each run against the landscape's dense-grid reference - available here
only because these problems are one-dimensional and can be brute-forced - is what
separates a search that finished from a search that stalled. Lessons 08 and 09
work on landscapes where that check is not available, which is exactly when the
distinction starts to matter.

CHANGES FROM ga_flow_04_stopping_condition.py
Introduce them in this order:
    1. staircase()            a landscape whose progress comes in jumps, not slopes
    2. PROBLEMS               the third problem joins the other two, nothing else moves
    3. the verdict            measure every run against the dense-grid reference, not against itself

Run it:  python ga_flow_05_stepped_landscapes.py

staircase() is a landscape whose progress comes in jumps; TENT = 10.5 so the
top step is 0.5 wide rather than a single point. The verdict compares each run
to a dense-grid reference: all three runs match that sampled reference,
shortfall +0.0000. This does not prove global optimality in the continuum.
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
MAX_GENERATIONS = 60   # the cap, unchanged since step 4
GENE_MIN, GENE_MAX = -10.0, 10.0
TOURNAMENT_SIZE, BLEND_ALPHA = 3, 1.0
MUTATION_MU, MUTATION_SIGMA = 0.0, 1.0
TARGET = 4.2
PEAK = 4.0             # where the staircase has its top step
SLOPE = 2.0            # how fast the steps narrow as they rise
TENT = 10.5            # tent height: makes the top step 0.5 wide, not a single point
# How long we wait, and what we are willing to call progress. A rule with no
# MIN_IMPROVEMENT is fooled by the fifth decimal place and never fires; a rule
# with no PATIENCE fires on the first unlucky generation. PATIENCE = 0 switches
# the whole rule off, which is how the control runs below are made.
PATIENCE = 5
MIN_IMPROVEMENT = 1e-4
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


# --- NEW (1) staircase() ------------------------------------------------------
def staircase(x: float, step: float = 1.0, rise: float = 0.2) -> float:
    """A landscape whose progress comes in jumps instead of slopes.

    Within one step the value is constant, so every individual standing on the
    same step has exactly the same fitness and selection has nothing to compare.
    The only way up is a mutation large enough to cross a step boundary - and
    until one happens, the best fitness does not move at all.

    This is not a contrived shape. Any fitness built from a count, a category or
    a rounded measurement looks like this, and the combinatorial problems of
    Lessons 09 to 11 are all of that kind.

    The steps rise towards PEAK and fall away after it, so the top step sits in
    the middle of the interval. That detail matters: if the best step were at a
    border, clamp() would deliver it for free to any mutation that overshoots,
    and the landscape would be easy for a reason that has nothing to do with the
    algorithm.

    TENT matters for a second reason, found the hard way. With TENT exactly 10.0
    the condition for the top step, TENT - SLOPE*|x - PEAK| >= 10, holds at the
    single point x = PEAK and nowhere else: the best step has ZERO WIDTH. A
    brute-force optimum taken on a grid that happens to contain that point then
    reports a maximum no search could ever land on, and every "shortfall" measured
    against it is an artefact. TENT = 10.5 gives the top step a width of 0.5.

    The general rule this is an instance of: when you compare a run against a
    brute-forced optimum, check that the optimum occupies some volume. An optimum
    the grid can see and the search cannot is not a benchmark, it is a bug.

    Args:
        x: a real gene in [-10, 10].
        step: width of one floor-band. 1.0 here.
        rise: fitness gained per step. 0.2 here.

    Returns:
        A multiple of `rise`. With TENT = 10.5 the top step is 0.5 wide
        and the dense-grid reference is +2.0000.

    Example:
        The run of this step reaches +2.0000 and the stopping rule costs
        a shortfall of +0.0000.
    """
    return float(np.floor((TENT - SLOPE * abs(x - PEAK)) / step) * rise)
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


def run(fitness_function: Fitness,
        patience: int = PATIENCE) -> Tuple[List[List[Individual]],
                                           List[Dict[str, int]]]:
    """INITIALISE, then evolve until the stopping condition says stop.

    Read the body and notice what is missing: no sine, no target, no mention of
    what is being optimised. That absence is the point of step 2.

    Args:
        fitness_function: the injected landscape.
        patience: generations without MIN_IMPROVEMENT before stopping.
            0 disables the rule (the control runs).

    Returns:
        (generations, censuses). The last generation is not necessarily
        the champion, because this flow has no elitism.

    Example:
        PATIENCE = 5 saves 49 to 51 of the 60 allowed generations and
        costs less than MIN_IMPROVEMENT on both smooth problems.
    """
    random.seed(SEED)
    population = [create_random(fitness_function) for _ in range(POPULATION_SIZE)]

    generations = [population]
    censuses: List[Dict[str, int]] = []
    for generation in range(1, MAX_GENERATIONS + 1):
        population, census = evolve_one_generation(population)
        generations.append(population)
        censuses.append(census)
        bests = [max(i.fitness for i in pop) for pop in generations]
        last_improvement = 0
        for g in range(1, len(bests)):
            if bests[g] > max(bests[:g]) + MIN_IMPROVEMENT:
                last_improvement = g
        if patience and generation - last_improvement >= patience:
            break
    return generations, censuses


# --- NEW (2) PROBLEMS ---------------------------------------------------------
PROBLEMS = [("sine landscape", sine_landscape),
            ("closeness to target", closeness_to_target),
            ("staircase", staircase)]
# ------------------------------------------------------------------------------

results = []
for name, fitness_function in PROBLEMS:
    generations, censuses = run(fitness_function)
    results.append((name, fitness_function, generations))

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
fig.savefig(FIGURES / "ga_flow_05_stepped_landscapes_history.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Figure saved to {FIGURES}/ga_flow_05_stepped_landscapes_history.png")

# A stopping rule is only worth having if it is cheap. Price it: run each
# problem again with the rule switched off, all the way to the cap.
#
# Both columns report the best individual EVER SEEN, not the best in the last
# generation. Without elitism those are different numbers, and the difference is
# reported below - a flow that throws its champion away has to remember it
# somewhere outside the population.
def best_ever(generations: List[List[Individual]]) -> Individual:
    """The best individual the run ever held.

    Args:
        generations: every generation the run held.

    Returns:
        The Individual with the highest fitness ever seen, which may
        not be in the last generation (this flow has no elitism).

    Example:
        On 1 of 2 problems in Lesson 02 step 4 the last generation no
        longer holds the champion.
    """
    return max((ind for pop in generations for ind in pop), key=lambda i: i.fitness)


print(f"Pricing the stopping rule: PATIENCE = {PATIENCE}, "
      f"MIN_IMPROVEMENT = {MIN_IMPROVEMENT}, cap = {MAX_GENERATIONS}.\n")
print("            problem | rule on          | rule off         | cost of stopping")
print("--------------------+------------------+------------------+-----------------")
priced = []
for name, fitness_function in PROBLEMS:
    patient, _ = run(fitness_function, patience=PATIENCE)
    full, _ = run(fitness_function, patience=0)
    on, off = best_ever(patient), best_ever(full)
    priced.append((name, patient, full, on, off))
    print(f"{name:>19s} | gen {len(patient) - 1:2d}  {on.fitness:+.4f} |"
          f" gen {len(full) - 1:2d}  {off.fitness:+.4f} |"
          f" {off.fitness - on.fitness:+.4f}")

savings = [MAX_GENERATIONS - (len(p) - 1) for _, p, _, _, _ in priced]
costs = [off.fitness - on.fitness for _, _, _, on, off in priced]
print(f"\nThe rule saved {min(savings)} to {max(savings)} generations of the "
      f"{MAX_GENERATIONS} allowed.")
if max(costs) < MIN_IMPROVEMENT:
    print(f"It cost less than MIN_IMPROVEMENT ({MIN_IMPROVEMENT}) on both problems:")
    print("the improvements really had dried up, and the rest of the cap was")
    print("waste. On these two landscapes the rule is free.")
else:
    print(f"It cost up to {max(costs):+.4f}: on at least one of these problems the")
    print("run was still going somewhere when the rule stopped it.")

lost = [(name, best_ever(p).fitness, max(p[-1], key=lambda i: i.fitness).fitness)
        for name, p, _, _, _ in priced]
dropped = [(n, e, f) for n, e, f in lost if e - f > MIN_IMPROVEMENT]
if dropped:
    print(f"\nAnd the wart: on {len(dropped)} of {len(lost)} problems the last")
    for n, e, f in dropped:
        print(f"generation does not contain the champion - {n} ended holding")
        print(f"{f:+.4f} after having seen {e:+.4f}. Nothing in this flow protects")
        print("the best individual; later lessons add elitism for exactly this.")
else:
    print("\nOn both problems the last generation still contains the champion,")
    print("which is luck rather than design: nothing in this flow protects it.")
# --- NEW (3) the verdict ------------------------------------------------------
# Step 4 asked what the rule COST. That question has a blind spot: a rule can be
# free because the search was finished, or free because the search was dead, and
# the cost column cannot tell those apart. So ask a second question the run
# cannot answer about itself - how far from the landscape's dense-grid reference did it
# end up? - by brute force on a grid, which is only possible because these
# landscapes are one-dimensional.
GRID = np.linspace(GENE_MIN, GENE_MAX, 20001)

print("\nWhat the cost column cannot see: distance from the dense-grid reference.")
print("            problem | best found | dense-grid reference | shortfall | final spread")
print("--------------------+------------+--------------+-----------+-------------")
shortfalls = []
for name, patient, full, on, off in priced:
    fitness_function = dict(PROBLEMS)[name]
    true_best = max(fitness_function(float(x)) for x in GRID)
    shortfall = true_best - off.fitness
    spread = population_metrics(full[-1])["spread"]
    shortfalls.append((name, shortfall, spread, len(patient) - 1, len(full) - 1))
    print(f"{name:>19s} | {off.fitness:+10.4f} | {true_best:+12.4f} |"
          f" {shortfall:+9.4f} | {spread:12.3f}")

reached = [row for row in shortfalls if row[1] <= MIN_IMPROVEMENT]
stalled = [row for row in shortfalls if row[1] > MIN_IMPROVEMENT]

print("\nA stopping rule is a bet that a plateau is the end of the search.")
if not stalled:
    print(f"On all {len(reached)} problems the bet was right: every run held its")
    print("landscape's optimum by the time the rule fired, staircase included.")
    print("\nSo the suspicion this step was built on is refuted. A stepped")
    print("landscape does not defeat a patience rule, because these steps RISE:")
    print("each boundary between them is a comparison selection can act on, and")
    print("between boundaries the population only has to drift far enough to")
    print("reach the next one. Flat in places is not the same as directionless.")
else:
    for name, shortfall, spread, stopped, capped in stalled:
        print(f"    {name}: ended {shortfall:+.4f} short of the optimum, and the")
        print(f"    control run to generation {capped} ended exactly as short as")
        print(f"    the one stopped at generation {stopped}. Final spread {spread:.3f}.")
    print("\nThe rule cost nothing and the run still failed, so the rule was not")
    print("the failure: the population ran out of spread to search with.")

print("\nKeep the instrument, not the result. 'The rule cost nothing' and 'the")
print("run succeeded' are different claims, and step 4's cost column can only")
print("make the first one. The second needs a reference outside the run. Here")
print("that reference is brute force on a grid, which works because x is one")
print("number; from Lesson 08 onwards it is not, and the honest answer to 'did")
print("this run succeed?' becomes genuinely hard to get.")

# The picture: the three landscapes, where each run stopped, and the dense-grid reference.
fig, axes = plt.subplots(1, len(PROBLEMS), figsize=(4.2 * len(PROBLEMS), 3.6))
for ax, (name, patient, full, on, off) in zip(axes, priced):
    fitness_function = dict(PROBLEMS)[name]
    values = [fitness_function(float(x)) for x in GRID]
    ax.plot(GRID, values, color="tab:blue", alpha=0.7, linewidth=1)
    top = float(GRID[int(np.argmax(values))])
    ax.plot([top], [max(values)], "*", color="tab:purple", markersize=14,
            label=f"dense-grid reference {max(values):+.3f}")
    ax.plot([on.gene], [on.fitness], "o", color="tab:red", markersize=9,
            label=f"rule on, gen {len(patient) - 1}")
    ax.plot([off.gene], [off.fitness], "s", color="tab:green", markersize=7,
            label=f"rule off, gen {len(full) - 1}")
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("x"); ax.set_ylabel("fitness")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=8, loc="lower center")
fig.suptitle("Where each run stopped, and where the optimum actually was")
fig.tight_layout(rect=(0, 0, 1, 0.90))
FIGURES.mkdir(exist_ok=True)
fig.savefig(FIGURES / "ga_flow_05_stepped_landscapes.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved to {FIGURES}/ga_flow_05_stepped_landscapes.png")
# ------------------------------------------------------------------------------

