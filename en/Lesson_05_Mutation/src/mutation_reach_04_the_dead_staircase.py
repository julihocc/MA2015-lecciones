"""
Lesson 05 - Step 4: The run Lesson 02 left for dead
====================================================
NEW IN THIS STEP: Lesson 02's generational loop and its staircase landscape, a
measurement of how wide the steps of that staircase actually are, and a sweep of
every mutation regime over 100 independent runs.

An earlier version of Lesson 02 step 5 ended with a run that stopped six
generations in, 0.2000 short of the staircase's optimum, with almost no gene
spread left, and pointed here: mutation "is the one operator that could put it
back". This step is where that promise came due - and it did not come due the way
it was written. What the investigation found instead sent a correction back to
Lesson 02, which no longer makes that claim.

What the sweep finds: no mutation regime rescues that run. Not a bigger sigma,
not a higher rate, not any of the eight combinations tried, over 100 seeds each.
Mutation multiplies the surviving gene spread several times over and buys
exactly zero fitness.

So the second question had to be asked - why is the top of that staircase
unreachable? - and the answer is not about mutation at all. With the tent height
written as 10.0, the top step of that staircase is a single point: it has ZERO
WIDTH. The brute-force grid that declared the optimum +2.0000 found it only
because x = 4.0 happens to be one of its 20001 sample points; no run could ever
land on it, with any operator. The shortfall was unreachable by construction, so
the "dead run" it described was not dead - it had arrived.

Raise the landscape by half a step - one constant - and the top step becomes
0.5 units wide. Then the same mutation regime, completely unchanged, reaches it
in 96 of 100 runs.

That is why Lesson 02 now uses a tent of 10.5 and reports a refuted hypothesis
rather than a shortfall. The broken geometry is kept here on purpose, as the
thing being diagnosed: this step is the diagnosis, not the bug.

The lesson is worth more than the one that was planned. Before asking which
operator is failing, check that the target exists.

CHANGES FROM mutation_reach_03_fitness_driven.py
Introduce them in this order:
    1. run()                  Lesson 02's whole generational loop, brought across unchanged
    2. staircase()            Lesson 02's third landscape, the one its run died on
    3. step_widths()          measure how wide each step of that landscape actually is
    4. PEAK_HEIGHT_REPAIRED   the repair the measurement demands: half a step of headroom
    5. sweep()                every mutation regime over 100 independent runs, on both landscapes

Run it:  python mutation_reach_04_the_dead_staircase.py

run() and staircase() reproduce Lesson 02's dead search (best +1.8000 when the
tent is 10.0). step_widths() says why: the top step is 0.000 wide. Eight
mutation regimes times 100 runs reach it 0 times in 800. Raising the tent to
10.5 gives the top step width 0.500, and Lesson 02's own regime then reaches
it 96 of 100 times.
"""
import random
import statistics
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

SEED = 52
GENE_MIN, GENE_MAX = -10.0, 10.0
MUTATION_MU = 0.0
# Lesson 02's settings, to the digit. Nothing here is tuned for this lesson.
POPULATION_SIZE = 10
CROSSOVER_PROBABILITY = 0.8
MAX_GENERATIONS = 60
TOURNAMENT_SIZE, BLEND_ALPHA = 3, 1.0
PATIENCE, MIN_IMPROVEMENT = 5, 1e-4
# The staircase's geometry, as Lesson 02 first had it. PEAK_HEIGHT is the value
# the tent reaches at its apex before floor() turns it into steps; it was written
# inline as the literal 10.0 there, and it is named here because the whole step
# turns on it. Lesson 02 has since been corrected to 10.5 - because of what this
# step found - so the 10.0 below is deliberately the OLD value: it is the
# specimen under the microscope, not a copy that drifted out of date.
PEAK = 4.0
SLOPE = 2.0
PEAK_HEIGHT = 10.0
STEP, RISE = 1.0, 0.2
# Eight mutation regimes: (probability that an individual mutates, sigma). With
# one gene per individual this probability IS step 2's per-gene p - the two
# coins coincide - so the regime grid is the same two dials as before.
REGIMES = ((0.1, 1.0), (0.1, 3.0), (0.3, 1.0), (0.3, 3.0),
           (0.6, 1.0), (0.6, 3.0), (1.0, 1.0), (1.0, 3.0))
LESSON_02_REGIME = (0.1, 1.0)
RUNS = 100
FIGURES = Path(__file__).resolve().parent.parent / "figures"

Fitness = Callable[[float], float]


def objective(x: float) -> float:
    """Lesson 01's landscape, unchanged. Works on a float and on an array.

    Args:
        x: a real gene, typically in [-10, 10].

    Returns:
        sin(x) - 0.2 * |x|. Larger is better.

    Example:
        The global peak of this landscape is near x = +1.372, f = +0.706.
    """
    return np.sin(x) - 0.2 * abs(x)


# --- NEW (2) staircase() ------------------------------------------------------
def staircase(x: float, height: float = PEAK_HEIGHT) -> float:
    """Lesson 02's third landscape: progress arrives in jumps, not on a slope.

    A tent of height `height` centred on PEAK, pushed through floor() so that
    every point on one step scores identically. Within a step, selection has
    nothing to compare and crossover nothing to combine; the only operator that
    can cross a step boundary is mutation. That is why Lesson 02 handed the
    problem to this lesson.

    `height` is a parameter rather than the literal 10.0 of Lesson 02 because
    step_widths() below is about to show that the literal is the whole problem.

    Args:
        x: a real gene.
        height: tent height. 10.0 makes the top step a single point;
            10.5 gives it width 0.500.

    Returns:
        A stepped fitness. The top step of height 10.0 is 0.000 wide.

    Example:
        Eight regimes times 100 runs reach the height-10.0 top step
        0 times in 800; with height 10.5, Lesson 02's own regime
        reaches it 96 of 100 times.
    """
    return float(np.floor((height - SLOPE * abs(x - PEAK)) / STEP) * RISE)
# ------------------------------------------------------------------------------


def clamp(gene: float, low: float = GENE_MIN, high: float = GENE_MAX) -> float:
    """Mutation can propose any real number; the search space has edges.

    Args:
        gene: raw gene (float).
        low, high: closed bounds of the search interval.

    Returns:
        gene projected onto [low, high].

    Example:
        At sigma = 6.0, 19.4% of proposals are clamped and mean travel
        drops to 0.690 of sigma; Lesson 01's sigma = 1.0 / 3.0 pair is
        0 of 2000 vs 110 of 2000 arrivals from x = -4.6.
    """
    return max(low, min(high, gene))


# --- NEW (1) run() ------------------------------------------------------------
# Everything in this band is Lesson 02's, transplanted without a change: the
# individual, tournament selection, blend crossover, gaussian mutation, the
# five-phase generation and the patience rule. It is new to this FILE, not to
# the students. The two mutation dials are arguments of run() instead of module
# constants, which is the only edit, and it is what makes a sweep possible.
#
# mutate() keeps Lesson 02's exact shape - one coin for the individual, then one
# gaussian draw - rather than calling step 3's mutate_random_deviation(). With a
# single-gene chromosome the two are the same operator, but they consume the
# random stream differently, and reproducing Lesson 02's run to the digit
# requires its stream.
class Individual:
    """One candidate solution, carrying the function that judges it.


    Example:
        From x = -4.6, sigma = 1.0 arrives 0 of 2000 times and sigma = 3.0
        arrives 110 of 2000.
    """

    def __init__(self, gene_list: list[float], fitness_function: Fitness) -> None:
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


def select_tournament(population: list[Individual], size: int) -> list[Individual]:
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


def crossover(p1: Individual, p2: Individual) -> tuple[Individual, Individual]:
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


def mutate(ind: Individual, sigma: float) -> Individual:
    """Add Gaussian noise and wrap the result as a new Individual.

    Args:
        ind: the parent Individual.
        sigma: mutation step. Swept in this file.

    Returns:
        A new Individual with a clamped mutant gene.
    """
    return Individual([clamp(ind.gene + random.gauss(MUTATION_MU, sigma))],
                      ind.fitness_function)


def evolve_one_generation(population: list[Individual], mutation_probability: float,
                          sigma: float) -> list[Individual]:
    """SELECT -> CROSSOVER -> MUTATE -> replace, once.

    Args:
        population: current generation.
        mutation_probability: per-individual coin.
        sigma: Gaussian step.

    Returns:
        The next generation.
    """
    selected = select_tournament(population, TOURNAMENT_SIZE)
    crossed: list[Individual] = []
    for p1, p2 in zip(selected[::2], selected[1::2]):
        if random.random() < CROSSOVER_PROBABILITY:
            crossed.extend(crossover(p1, p2))
        else:
            crossed.extend([p1, p2])
    return [mutate(ind, sigma) if random.random() < mutation_probability else ind
            for ind in crossed]


def run(fitness_function: Fitness, mutation_probability: float, sigma: float,
        seed: int = SEED, patience: int = PATIENCE) -> list[list[Individual]]:
    """Initialise, then evolve until the improvements dry up.

    Args:
        fitness_function: staircase or the sine landscape.
        mutation_probability, sigma: the mutation regime under test.
        seed: one run.
        patience: stagnation stop, as in Lesson 02.

    Returns:
        The list of generations.

    Example:
        Lesson 02's regime on the unrepaired staircase ends at +1.8000,
        0.2000 short, because the top step has width 0.000.
    """
    random.seed(seed)
    population = [Individual([random.uniform(GENE_MIN, GENE_MAX)], fitness_function)
                  for _ in range(POPULATION_SIZE)]
    generations = [population]
    for generation in range(1, MAX_GENERATIONS + 1):
        population = evolve_one_generation(population, mutation_probability, sigma)
        generations.append(population)
        bests = [max(i.fitness for i in pop) for pop in generations]
        last_improvement = 0
        for g in range(1, len(bests)):
            if bests[g] > max(bests[:g]) + MIN_IMPROVEMENT:
                last_improvement = g
        if patience and generation - last_improvement >= patience:
            break
    return generations


def best_ever(generations: list[list[Individual]]) -> Individual:
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


def gene_spread(population: list[Individual]) -> float:
    """gene_spread as used in this step.

    Args:
        population: one generation.

    Returns:
        Standard deviation of the genes. Near zero, the search is over.
    """
    return float(np.std([ind.gene for ind in population]))
# ------------------------------------------------------------------------------


# --- NEW (3) step_widths() ----------------------------------------------------
def step_widths(height: float, levels: int = 4) -> list[tuple[float, float]]:
    """How wide, in x, is each of the top `levels` steps of the staircase?

    Solved rather than sampled, which is the point: a grid search cannot see a
    step narrower than its own spacing, and that is precisely the mistake this
    function exists to catch.

    Step k covers the x with floor(height - SLOPE*|x - PEAK|) == k, that is
    (height - k - 1) / SLOPE < |x - PEAK| <= (height - k) / SLOPE, so its width
    is twice the length of that interval in |x - PEAK|, clipped at zero.

    Args:
        height: tent height, 10.0 or 10.5.
        levels: how many steps to report.

    Returns:
        (fitness, width) per step. Width 0.000 is an optimum the
        grid can see and the search cannot.

    Example:
        height 10.0: top step 0.000 wide. height 10.5: top step
        0.500 wide, reached 96 of 100 times by Lesson 02's regime.
    """
    top = int(np.floor(height))
    widths = []
    for k in range(top, top - levels, -1):
        outer = (height - k) / SLOPE
        inner = max(0.0, (height - k - 1) / SLOPE)
        widths.append((k * RISE, 2 * max(0.0, outer - inner)))
    return widths
# ------------------------------------------------------------------------------


# --- NEW (4) PEAK_HEIGHT_REPAIRED ---------------------------------------------
# Half a step of headroom. The tent now peaks at 10.5 instead of 10.0, so the
# topmost floor() band is no longer the single point where the tent touches an
# integer - it is an interval. Nothing else about the landscape changes: the
# number of steps, their height, the slope and the position of the peak are all
# as Lesson 02 wrote them, and the top step is still interior.
PEAK_HEIGHT_REPAIRED = 10.5
# ------------------------------------------------------------------------------


# --- NEW (5) sweep() ----------------------------------------------------------
def sweep(height: float, runs: int = RUNS) -> list[dict]:
    """Every regime in REGIMES, over `runs` independent seeds, on one landscape.

    One seed proves nothing about a stochastic search: Lesson 02's conclusion
    rested on a single run. `runs` separate seeds per regime is what lets a row
    of this table be read as a rate rather than an anecdote.

    Args:
        height: tent height under test.
        runs: 100 per regime.

    Returns:
        Per-regime arrival counts.

    Example:
        0 of 800 on the unrepaired staircase; 96 of 100 for the
        original regime on the repaired one.
    """
    def landscape(x: float) -> float:
        return staircase(x, height)

    optimum = max(level for level, _ in step_widths(height))
    rows = []
    for probability, sigma in REGIMES:
        bests, spreads, lengths = [], [], []
        for seed in range(runs):
            generations = run(landscape, probability, sigma, seed=seed)
            bests.append(best_ever(generations).fitness)
            spreads.append(gene_spread(generations[-1]))
            lengths.append(len(generations) - 1)
        rows.append({"probability": probability, "sigma": sigma,
                     "reached": sum(1 for b in bests if b >= optimum - 1e-9),
                     "runs": runs,
                     "mean_best": statistics.fmean(bests),
                     "mean_spread": statistics.fmean(spreads),
                     "mean_generations": statistics.fmean(lengths),
                     "optimum": optimum})
    return rows
# ------------------------------------------------------------------------------


def print_sweep(rows: list[dict]) -> None:
    """Print the per-regime arrival table.

    Args:
        rows: output of sweep().

    Returns:
        None. Prints the table this step is built around.
    """
    row_format = " {:>5} | {:>5} | {:>13} | {:>9} | {:>11} | {:>15}"
    header = row_format.format("p", "sigma", "reached top", "mean best",
                               "mean spread", "mean generations")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(row_format.format(f"{r['probability']:.1f}", f"{r['sigma']:.1f}",
                                f"{r['reached']} of {r['runs']}",
                                f"{r['mean_best']:.4f}",
                                f"{r['mean_spread']:.3f}",
                                f"{r['mean_generations']:.2f}"))


# ===== Part 1: reproduce the run Lesson 02 gave up on =========================
probability, sigma = LESSON_02_REGIME
generations = run(lambda x: staircase(x), probability, sigma)
champion = best_ever(generations)
optimum = max(level for level, _ in step_widths(PEAK_HEIGHT))
print("Lesson 02 step 5, reproduced here with its own seed and its own "
      "settings:")
print(f"    regime: mutation probability {probability}, sigma {sigma}, "
      f"SEED = {SEED}")
print(f"    stopped after {len(generations) - 1} generations of "
      f"{MAX_GENERATIONS} allowed")
print(f"    best ever found: {champion}")
print(f"    the staircase's optimum: {optimum:+.4f}")
print(f"    shortfall: {optimum - champion.fitness:+.4f}")
print(f"    gene spread in the final generation: "
      f"{gene_spread(generations[-1]):.3f}")
print("Lesson 02 reported the same shortfall and handed the problem here, on")
print("the grounds that a population with no spread left needs mutation to put")
print("some back. So: give it more mutation.")

# ===== Part 2: the sweep that was supposed to fix it ==========================
print(f"\nEvery regime, {RUNS} independent runs each, on Lesson 02's staircase "
      f"(peak height {PEAK_HEIGHT}):\n")
before = sweep(PEAK_HEIGHT)
print_sweep(before)

spreads = [r["mean_spread"] for r in before]
bests = [r["mean_best"] for r in before]
total_reached = sum(r["reached"] for r in before)
print(f"\nMutation does what it is supposed to do to the spread: from "
      f"{min(spreads):.3f} to {max(spreads):.3f},")
print(f"a factor of {max(spreads) / min(spreads):.1f} across the grid. And the "
      f"best-fitness column does not move:")
print(f"{min(bests):.4f} to {max(bests):.4f}, all of it short of "
      f"{before[0]['optimum']:+.4f}.")
print(f"The top step was reached {total_reached} times in "
      f"{len(before) * RUNS} runs.")
print("\nThat is a flat refusal, and it is too flat to be about tuning. When no")
print("setting of an operator changes an outcome at all, suspect the question.")

# ===== Part 3: the diagnosis ==================================================
print(f"\nHow wide are the steps of this staircase? Solved exactly, not sampled:")
print("    fitness of step | width in x")
print("    ----------------+-----------")
for level, width in step_widths(PEAK_HEIGHT):
    print(f"       {level:+12.4f} | {width:10.3f}")
top_level, top_width = step_widths(PEAK_HEIGHT)[0]
print(f"\nThe top step has width {top_width:.3f}. It is the single point "
      f"x = {PEAK:.1f},")
print("where the tent touches an integer exactly. A gene drawn from a continuous")
print("distribution lands on it with probability zero, so no mutation regime,")
print("no crossover and no amount of patience can ever produce it.")
print(f"\nLesson 02's brute-force check evaluated the landscape on "
      f"numpy.linspace({GENE_MIN:.0f}, {GENE_MAX:.0f}, 20001),")
print(f"whose spacing is {20.0 / 20000:.3f} - and {PEAK:.1f} is one of its "
      f"sample points. That is the only")
print(f"reason the optimum was reported as {top_level:+.4f}. The grid saw a peak "
      "that the search")
print("cannot reach, and the 0.2000 shortfall Lesson 02 published is the width")
print("of that illusion.")

# ===== Part 4: the repair, and the sweep that matters ========================
print(f"\nThe repair is one constant: peak height {PEAK_HEIGHT} -> "
      f"{PEAK_HEIGHT_REPAIRED}.")
print("    fitness of step | width in x")
print("    ----------------+-----------")
for level, width in step_widths(PEAK_HEIGHT_REPAIRED):
    print(f"       {level:+12.4f} | {width:10.3f}")
repaired_top, repaired_width = step_widths(PEAK_HEIGHT_REPAIRED)[0]
print(f"\nSame number of steps, same rise, same slope, same interior peak - and "
      f"a top")
print(f"step {repaired_width:.3f} units wide, worth {repaired_top:+.4f}, which "
      "is what the old landscape")
print("claimed to be worth all along.")

print(f"\nEvery regime again, {RUNS} runs each, on the repaired staircase:\n")
after = sweep(PEAK_HEIGHT_REPAIRED)
print_sweep(after)

lesson_02_row = next(r for r in after
                     if (r["probability"], r["sigma"]) == LESSON_02_REGIME)
best_row = max(after, key=lambda r: r["reached"])
worst_row = min(after, key=lambda r: r["reached"])
print(f"\nLesson 02's own regime - p = {lesson_02_row['probability']}, "
      f"sigma = {lesson_02_row['sigma']}, the one it called insufficient -")
print(f"reaches the top in {lesson_02_row['reached']} of "
      f"{lesson_02_row['runs']} runs. Nothing about mutation was changed to")
print("get that. The landscape was.")
print(f"\nThe best regime on the table is p = {best_row['probability']}, "
      f"sigma = {best_row['sigma']} at {best_row['reached']} of "
      f"{best_row['runs']};")
print(f"the worst is p = {worst_row['probability']}, sigma = "
      f"{worst_row['sigma']} at {worst_row['reached']}. Note which one that is:")
if worst_row["sigma"] > best_row["sigma"]:
    print("the LARGER sigma is the worse one. Step 1 found the same trade on a")
    print("single gene - a long step is good at arriving and bad at staying -")
    print("and here it costs whole runs: a mutation big enough to find the top")
    print("step is also big enough to knock its discoverer off again.")
    pairs = [(r["probability"],
              next(s["reached"] for s in after
                   if s["probability"] == r["probability"] and s["sigma"] == 1.0),
              next(s["reached"] for s in after
                   if s["probability"] == r["probability"] and s["sigma"] == 3.0))
             for r in after if r["sigma"] == 1.0]
    losses = sum(1 for _, at1, at3 in pairs if at3 < at1)
    print(f"It holds at every rate on the table: in {losses} of {len(pairs)} "
          "pairs sharing a p,")
    print("    " + ",  ".join(f"p={q}: {at1} -> {at3}" for q, at1, at3 in pairs))
    print("    raising sigma from 1.0 to 3.0 costs runs.")
else:
    print("the larger sigma is the better one here, which is not what step 1's")
    print("trade predicted; the staircase is wide enough to forgive long steps.")

print("\n--- what this step actually taught -----------------------------------")
print("Lesson 02 promised that mutation could revive its dead run. It could not,")
print("and measuring eight regimes over 100 runs each is what proved it. The")
print("failure was in the landscape, not the operator: the target had zero")
print("width. Before tuning an operator, check that what you are asking for")
print("exists - and then measure over many seeds, because one run cannot tell")
print("a dead search from an impossible one.")

# The picture: the two landscapes at the peak, and the success rates.
fig, (ax_land, ax_rate) = plt.subplots(1, 2, figsize=(11.5, 4.0))
fine = np.linspace(PEAK - 2.0, PEAK + 2.0, 4001)
ax_land.plot(fine, [staircase(float(x), PEAK_HEIGHT) for x in fine],
             color="tab:red", linewidth=1.6,
             label=f"peak height {PEAK_HEIGHT} (top step {top_width:.2f} wide)")
ax_land.plot(fine, [staircase(float(x), PEAK_HEIGHT_REPAIRED) for x in fine],
             color="tab:green", linewidth=1.6, linestyle="--",
             label=f"peak height {PEAK_HEIGHT_REPAIRED} "
                   f"(top step {repaired_width:.2f} wide)")
ax_land.plot([PEAK], [top_level], "*", color="tab:red", markersize=13)
ax_land.set_title("The top of the staircase, magnified", fontsize=10)
ax_land.set_xlabel("x")
ax_land.set_ylabel("fitness")
ax_land.grid(True, linestyle=":", alpha=0.5)
ax_land.legend(fontsize=8, loc="lower center")

labels = [f"{r['probability']}/{r['sigma']}" for r in before]
positions = np.arange(len(labels))
ax_rate.bar(positions - 0.2, [100 * r["reached"] / r["runs"] for r in before],
            width=0.4, color="tab:red", label=f"peak height {PEAK_HEIGHT}")
ax_rate.bar(positions + 0.2, [100 * r["reached"] / r["runs"] for r in after],
            width=0.4, color="tab:green",
            label=f"peak height {PEAK_HEIGHT_REPAIRED}")
ax_rate.set_xticks(positions)
ax_rate.set_xticklabels(labels, fontsize=8)
ax_rate.set_xlabel("mutation regime (probability / sigma)")
ax_rate.set_ylabel(f"% of {RUNS} runs reaching the top step")
ax_rate.set_title("No regime rescues the first landscape; "
                  "every regime solves the second", fontsize=10)
ax_rate.grid(True, axis="y", linestyle=":", alpha=0.5)
ax_rate.legend(fontsize=8)
fig.suptitle("Lesson 02's dead staircase was not a mutation problem")
FIGURES.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURES / "mutation_reach_04_the_dead_staircase.png",
            dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved to {FIGURES}/mutation_reach_04_the_dead_staircase.png")

