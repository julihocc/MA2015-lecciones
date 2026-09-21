"""
Lesson 06 - Step 2: What counts as success
===========================================
NEW IN THIS STEP: brute_force_optimum(), TOLERANCE and verdict(),
within_tolerance_share(), and a second problem - Lesson 01's one-gene landscape.

Step 1 showed eight answers to one question. Before they can be averaged into
anything, one word has to be defined: success. A run cannot judge itself - it
reports the best thing it found, never how good that was - so the judgement has
to come from outside, and Lesson 02 step 5 built exactly that instrument: brute
force the landscape on a grid, and measure every run against the result. It also
left a warning attached to it, in its own closing lines: that reference "works
because x is one number; from Lesson 08 onwards it is not, and the honest answer
to 'did this run succeed?' becomes genuinely hard to get".

This step is where that bill comes due, one lesson earlier than advertised,
because Chapter 6's own landscape has two genes instead of one. Two things break
at once, and only one of them is the obvious one.

The obvious one is cost: a grid fine enough to resolve one gene needs its
resolution raised to the power of the number of genes. Lesson 02's grid of
20,001 points becomes 400 million for two genes and is unwriteable for ten.

The other one is worse, and it is the same defect Lesson 05 found in Lesson 02's
staircase, wearing different clothes. The 2-D landscape's optimum exists, and it
has no volume: the penalty term vanishes only on two lines, and it approaches
zero as a tenth power, so to score within 0.01 of the optimum a point must sit
within 1e-20 of a line. The grid finds that optimum only because the line
happens to run through it. No search will ever land on it, and no verdict built
on it can distinguish a good run from a bad one - it reports failure for both.

So the lesson moves to a landscape where the word means something: Lesson 01's
f(x) = sin(x) - 0.2|x|, whose optimum is a smooth peak with a measurable width.
Everything from here on is measured there. The 2-D problem stays in the file as
the specimen - the reason for the move has to stay visible.

CHANGES FROM success_rate_01_one_run.py
Introduce them in this order:
    1. brute_force_optimum()      Lesson 02's instrument: the truth, from outside the run
    2. TOLERANCE                  how close counts as arrival, and verdict() applies it
    3. within_tolerance_share()   the check Lesson 05 taught: does the target have volume?
    4. SINE                       Lesson 01's one-gene landscape, where the verdict survives

Run it:  python success_rate_02_the_verdict.py

brute_force_optimum() is the truth from outside the run. verdict() asks
whether a champion is within TOLERANCE = 0.01 of it. within_tolerance_share()
measures the target's volume. The 2-D optimum is real (+0.500000) but occupies
0.003197% of the grid — all 128 winning points sit on y = 10 — so 0 of 8 runs
succeed. On sine 1-D the target is 1.4299% of the box and 5 of 8 runs find it:
62.5% ± 17.1%.
"""
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

# The algorithm, exactly as Lessons 01-05 left it.
POPULATION_SIZE = 10
CROSSOVER_PROBABILITY = 0.8
MUTATION_PROBABILITY = 0.1
MUTATION_MU, MUTATION_SIGMA = 0.0, 1.0
TOURNAMENT_SIZE, BLEND_ALPHA = 3, 1.0
# Run i uses seed i, for the whole lesson.
RUNS = 8
FIGURES = Path(__file__).resolve().parent.parent / "figures"


@dataclass(frozen=True)
class Problem:
    """A landscape, its box, and the generation budget it is run with.

    Args:
        name: label for the printed tables.
        fitness: function of a gene list.
        low, high: the box.
        genes: chromosome length.
        generations: the run's budget.

    Example:
        TWO_D is Gridin's landscape on [-100, 100]^2 with 25 generations;
        SINE is Lesson 01's landscape on [-10, 10] with 10.
    """
    name: str
    fitness: Callable[[list[float]], float]
    low: float
    high: float
    genes: int
    generations: int


def book_2d(genes: list[float]) -> float:
    """Gridin's Chapter 6 landscape. The penalty term is the interesting half.

    Args:
        genes: [x, y] in [-100, 100]^2.

    Returns:
        sin(x)cos(x) minus a tenth-power penalty that vanishes on
        x = -50 and y = 10.

    Example:
        The dense-grid reference is +0.500000, but it occupies
        0.003197% of the grid — all 128 winning points sit on y = 10.
    """
    x, y = genes
    return np.sin(x) * np.cos(x) - pow(abs((x + 50.0) * (y - 10.0)) / 10.0, 0.1)


TWO_D = Problem("book 2-D", book_2d, -100.0, 100.0, 2, 25)


# --- NEW (4) SINE -------------------------------------------------------------
def sine_landscape(genes: list[float]) -> float:
    """Lesson 01's landscape, unchanged: one gene, one smooth global peak.

    Chosen over the book's 2-D function for one measured reason, printed below:
    its optimum occupies a stretch of x wide enough to be found. Everything the
    rest of this lesson counts is counted here.

    Args:
        genes: a one-element list holding x.

    Returns:
        sin(x) - 0.2 * |x|. Larger is better.

    Example:
        Success on this landscape means within 0.01 of the brute-forced
        optimum +0.705908, a target 1.43% of the box wide.
    """
    return float(np.sin(genes[0]) - 0.2 * abs(genes[0]))


# Ten generations rather than twenty-five: enough to succeed most of the time
# and fail often enough for the failures to be worth counting. A problem that is
# always solved measures nothing.
SINE = Problem("sine 1-D", sine_landscape, -10.0, 10.0, 1, 10)
# ------------------------------------------------------------------------------


class Individual:
    """One candidate solution, carrying the problem that judges it.


    Example:
        Eight seeds on book_2d spread 0.8946 in best-ever fitness; on sine,
        5 of 8 runs land within 0.01 of the optimum.
    """

    def __init__(self, genes: list[float], problem: Problem) -> None:
        """Build an individual and score it immediately.

        Args:
            genes: the chromosome, clamped to the problem's box.
            problem: landscape, bounds, gene count.

        Example:
            Each construction increments Individual.evaluations in the
            budget lessons, because a new Individual is one fitness call.
        """
        self.problem = problem
        self.genes = [clamp(g, problem) for g in genes]
        self.fitness = float(problem.fitness(self.genes))

    def __repr__(self) -> str:
        """Compact snapshot used in every printed table of this lesson.

        Returns:
            A one-line snapshot such as 'x=+1.372 f=+0.706' or
            '(+1.372) f=+0.7060', used in every printed table.
        """
        coordinates = ", ".join(f"{g:+.3f}" for g in self.genes)
        return f"({coordinates}) f={self.fitness:+.4f}"


def clamp(gene: float, problem: Problem) -> float:
    """Crossover and mutation propose anything; the box has edges.

    Args:
        gene: raw gene (float).
        problem: carries the box [low, high].

    Returns:
        gene projected onto [problem.low, problem.high].

    Example:
        Needed because blend alpha = 1.0 and N(0, 1) mutation both
        leave the box of whichever landscape is injected.
    """
    return max(problem.low, min(problem.high, gene))


def select_tournament(population: list[Individual]) -> list[Individual]:
    """Lesson 03's winner: the pressure is one integer, set on purpose.

    Args:
        population: current Individuals. Tournament size is the module
            constant TOURNAMENT_SIZE = 3.

    Returns:
        A list of the same length, holding references into population.

    Example:
        k = 3 is Lesson 03's measured setting: 2.74 expected copies of
        the best, not a wheel and not a floor.
    """
    return [max([random.choice(population) for _ in range(TOURNAMENT_SIZE)],
                key=lambda i: i.fitness) for _ in range(len(population))]


def crossover(parent1: Individual, parent2: Individual
              ) -> tuple[Individual, Individual]:
    """Blend crossover, one blending factor per gene.

    Args:
        parent1, parent2: parent Individuals.

    Returns:
        Two children, one blending factor per gene, then clamped.

    Example:
        BLEND_ALPHA = 1.0, the course default since Lesson 01 step 4.
    """
    genes1, genes2 = [], []
    for a, b in zip(parent1.genes, parent2.genes):
        shift = (1 + 2 * BLEND_ALPHA) * random.random() - BLEND_ALPHA
        genes1.append((1 - shift) * a + shift * b)
        genes2.append(shift * a + (1 - shift) * b)
    return (Individual(genes1, parent1.problem),
            Individual(genes2, parent1.problem))


def mutate(individual: Individual, sigma: float = MUTATION_SIGMA) -> Individual:
    """One coin decides whether the individual mutates; then every gene moves.

    Args:
        individual: the parent Individual.
        sigma: Gaussian step, default MUTATION_SIGMA = 1.0.

    Returns:
        A new Individual; every gene moves if the individual-level coin
        in the caller came up heads.

    Example:
        Per-individual probability 0.1 is Lesson 05's convention, reused
        so later counts of evaluations stay comparable.
    """
    return Individual([g + random.gauss(MUTATION_MU, sigma)
                       for g in individual.genes], individual.problem)


def evolve_one_generation(population: list[Individual]) -> list[Individual]:
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
    selected = select_tournament(population)
    crossed: list[Individual] = []
    for parent1, parent2 in zip(selected[::2], selected[1::2]):
        if random.random() < CROSSOVER_PROBABILITY:
            crossed.extend(crossover(parent1, parent2))
        else:
            crossed.extend([parent1, parent2])
    return [mutate(ind) if random.random() < MUTATION_PROBABILITY else ind
            for ind in crossed]


def run(problem: Problem, seed: int) -> list[list[Individual]]:
    """One complete run: initialise, then evolve for the problem's budget.

    Args:
        problem: landscape, box, gene count and generation budget.
        seed: run i uses seed i, so subsets of runs are reproducible.

    Returns:
        The list of generations.

    Example:
        Eight seeds on book_2d spread 0.8946 in best-ever fitness.
    """
    random.seed(seed)
    population = [Individual([random.uniform(problem.low, problem.high)
                              for _ in range(problem.genes)], problem)
                  for _ in range(POPULATION_SIZE)]
    generations = [population]
    for _ in range(problem.generations):
        population = evolve_one_generation(population)
        generations.append(population)
    return generations


def best_ever(generations: list[list[Individual]]) -> Individual:
    """The best individual the run ever held, which is the run's answer.

    Args:
        generations: every generation the run held.

    Returns:
        The Individual with the highest fitness ever seen, which may
        not be in the last generation (this flow has no elitism).

    Example:
        On 1 of 2 problems in Lesson 02 step 4 the last generation no
        longer holds the champion.
    """
    return max((ind for pop in generations for ind in pop),
               key=lambda i: i.fitness)


# --- NEW (1) brute_force_optimum() --------------------------------------------
def brute_force_optimum(problem: Problem, points_per_axis: int) -> float:
    """The best value on a regular grid over the whole box.

    Lesson 02 step 5's instrument, generalised to more than one gene - which is
    where it starts to hurt. The grid has points_per_axis ** problem.genes
    points, so raising the resolution of a two-gene problem costs the square of
    what it costs for one gene. That exponent is the whole difficulty, and it is
    why this function takes the resolution as an argument instead of hiding it.

    Args:
        problem: the landscape.
        points_per_axis: grid resolution. 20001 on 1-D; much coarser
            on 2-D because the cost is exponential in gene count.

    Returns:
        The best value the grid can see.

    Example:
        Sine 1-D: +0.705908. Book 2-D: +0.500000, occupying
        0.003197% of that grid.
    """
    axis = np.linspace(problem.low, problem.high, points_per_axis)
    if problem.genes == 1:
        return float(max(problem.fitness([float(x)]) for x in axis))
    mesh = np.meshgrid(*[axis] * problem.genes, indexing="ij")
    x, y = mesh
    values = np.sin(x) * np.cos(x) - np.power(
        np.abs((x + 50.0) * (y - 10.0)) / 10.0, 0.1)
    return float(np.max(values))
# ------------------------------------------------------------------------------


# --- NEW (2) TOLERANCE --------------------------------------------------------
# How close to the optimum counts as having found it. There is no principled
# value; there is only a value stated out loud, so that a success rate quoted
# later can be read as "within 0.01 of the optimum" rather than as a fact about
# the algorithm. Change it and every rate in this lesson changes with it.
TOLERANCE = 0.01


def verdict(individual: Individual, optimum: float) -> bool:
    """Did this run arrive? A yes/no, decided from outside the run.

    Args:
        individual: a run's champion (best-ever).
        optimum: brute_force_optimum of the same landscape.

    Returns:
        True iff fitness >= optimum - TOLERANCE (0.01).

    Example:
        0 of 8 book-2-D runs succeed (target has no volume). 5 of 8
        sine runs succeed: 62.5% ± 17.1%. On 1000 sine runs: 79.4% ± 1.3%.
    """
    return individual.fitness >= optimum - TOLERANCE
# ------------------------------------------------------------------------------


# --- NEW (3) within_tolerance_share() -----------------------------------------
def within_tolerance_share(problem: Problem, optimum: float,
                           points_per_axis: int) -> float:
    """What fraction of the box scores within TOLERANCE of the optimum?

    Lesson 05 found Lesson 02's staircase declaring an optimum that occupied a
    single point, so that no run could reach it and no operator could be blamed.
    This is that check, made routine: before believing a success rate, measure
    how much of the search space counts as success. A share of zero means the
    verdict is not measuring the search.

    Args:
        problem, optimum, points_per_axis: as brute_force_optimum.

    Returns:
        Fraction of the grid that scores within TOLERANCE of the
        optimum — the hit probability of one blind draw.

    Example:
        Book 2-D: 0.003197%. Sine 1-D: 1.4299% of the box (0.286 wide).
    """
    axis = np.linspace(problem.low, problem.high, points_per_axis)
    if problem.genes == 1:
        values = np.array([problem.fitness([float(x)]) for x in axis])
    else:
        x, y = np.meshgrid(*[axis] * problem.genes, indexing="ij")
        values = np.sin(x) * np.cos(x) - np.power(
            np.abs((x + 50.0) * (y - 10.0)) / 10.0, 0.1)
    return float(np.mean(values >= optimum - TOLERANCE))


def on_the_singular_line(points_per_axis: int, optimum: float
                         ) -> tuple[int, int]:
    """Of the 2-D grid points that count as success, how many sit on y = 10?

    The answer decides whether the share above measures a region or an artefact
    of where the grid's lines happen to fall.

    Args:
        points_per_axis: the 2-D grid resolution.
        optimum: the brute-forced 2-D peak.

    Returns:
        (winning points, how many of those sit on y = 10).

    Example:
        128 winning points, all 128 on y = 10: the target is a line,
        not a region.
    """
    axis = np.linspace(TWO_D.low, TWO_D.high, points_per_axis)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    values = np.sin(x) * np.cos(x) - np.power(
        np.abs((x + 50.0) * (y - 10.0)) / 10.0, 0.1)
    winners = values >= optimum - TOLERANCE
    return int(winners.sum()), int((winners & (y == 10.0)).sum())
# ------------------------------------------------------------------------------


# ===== Part 1: the instrument, and what two genes cost it =====================
print("Lesson 02's instrument, applied to a two-gene landscape.\n")
print("  points per axis |     grid points | optimum found")
print("  ----------------+-----------------+--------------")
for resolution in (201, 801, 2001):
    print(f"  {resolution:>15} | {resolution ** TWO_D.genes:>15,} | "
          f"{brute_force_optimum(TWO_D, resolution):+.6f}")
optimum_2d = brute_force_optimum(TWO_D, 2001)
print(f"\nThe grid says {optimum_2d:+.6f}, and it is right. On the line y = 10")
print("the penalty term is |0| ** 0.1 = 0 exactly, so the landscape there is")
print("0.5 * sin(2x), whose maximum is +0.500000. The optimum is real.")
lesson_02_grid = 20001
print(f"\nThe cost of finding it that way: Lesson 02 used a grid of "
      f"{lesson_02_grid:,} points")
print(f"for one gene. The same resolution costs "
      f"{lesson_02_grid ** 2:,} points for two genes,")
print(f"and about 10 ** {10 * np.log10(lesson_02_grid):.0f} for ten. Brute "
      "force is not a method; it is a luxury")
print("that one-dimensional teaching problems happen to afford.")

# ===== Part 2: the verdict that cannot be given ===============================
share_2d = within_tolerance_share(TWO_D, optimum_2d, 2001)
print(f"\nCost is not the real problem. Of the {2001 ** 2:,} grid points, "
      f"{share_2d:.6%} score")
print(f"within {TOLERANCE} of the optimum. Solve it by hand and see why: "
      "f >= 0.49 needs")
threshold = TOLERANCE ** 10
print(f"the penalty below {TOLERANCE}, and a tenth power below {TOLERANCE} "
      f"needs")
print(f"|(x + 50)(y - 10)| / 10 < {threshold:.0e}. At x = +22.5, that is a "
      f"band in y of")
print(f"half-width {10 * threshold / abs(22.5 + 50.0):.1e}, which no grid and "
      "no search resolves.")
winners, on_line = on_the_singular_line(2001, optimum_2d)
print(f"\nSo where are those {winners} winning grid points? "
      f"{on_line} of {winners} of them have y = 10.0")
print("exactly. The grid sees the optimum only because one of its own lines")
print("runs along it.")
print("\nThat is Lesson 05's finding again in two dimensions: an optimum with no")
print("volume. Lesson 05 could repair its staircase with one constant. Here the")
print("shape of the penalty IS the problem, so the verdict is what has to go.")
print(f"\nThe {RUNS} runs of step 1, judged against {optimum_2d:+.4f}:")
answers_2d = [best_ever(run(TWO_D, seed)) for seed in range(RUNS)]
successes_2d = sum(verdict(a, optimum_2d) for a in answers_2d)
print(f"    {successes_2d} of {RUNS} succeeded. Best run "
      f"{max(a.fitness for a in answers_2d):+.4f}, worst "
      f"{min(a.fitness for a in answers_2d):+.4f}.")
print("A verdict that returns the same answer for the best run and the worst")
print("one is not measuring the runs.")

# ===== Part 3: a landscape where the word means something =====================
optimum_1d = brute_force_optimum(SINE, lesson_02_grid)
share_1d = within_tolerance_share(SINE, optimum_1d, lesson_02_grid)
width_1d = share_1d * (SINE.high - SINE.low)
print(f"\nLesson 01's landscape, {SINE.name}, on the same grid Lesson 02 used "
      f"({lesson_02_grid:,} points):")
print(f"    optimum {optimum_1d:+.6f}")
print(f"    share of the box within {TOLERANCE}: {share_1d:.4%}, "
      f"a stretch of x {width_1d:.3f} wide")
print(f"    a gene drawn uniformly lands in it once in "
      f"{1 / share_1d:.0f} draws")
print("\nThat is a target. It has width, a blind draw hits it at a rate you can")
print("write down, and a run that reaches it did something the draw did not.")
print(f"\nThe same {RUNS} seeds, on this landscape, judged the same way:\n")
print("  seed |  best ever |  shortfall | verdict")
print("  -----+------------+------------+--------")
answers_1d = [best_ever(run(SINE, seed)) for seed in range(RUNS)]
for seed, champion in enumerate(answers_1d):
    hit = verdict(champion, optimum_1d)
    print(f"  {seed:>4} | {champion.fitness:+10.4f} | "
          f"{optimum_1d - champion.fitness:+10.4f} | "
          f"{'FOUND IT' if hit else 'missed'}")
successes_1d = sum(verdict(a, optimum_1d) for a in answers_1d)
rate = successes_1d / RUNS
wobble = (RUNS * rate * (1 - rate)) ** 0.5
print(f"\n{successes_1d} of {RUNS} runs found the optimum, a rate of "
      f"{rate:.1%}. That is a number worth")
print(f"having, and it is worth very little: if {rate:.1%} were the true rate, "
      f"the count in")
print(f"{RUNS} runs would still move by {wobble:.1f} runs from one experiment "
      "to the next, one")
print("standard deviation. Step 3 runs it properly.")

# The picture: the 1-D target has width; the 2-D one is a cliff edge.
fig, (ax_sine, ax_cusp) = plt.subplots(1, 2, figsize=(11.5, 4.0))
fine = np.linspace(SINE.low, SINE.high, 4001)
values = np.array([sine_landscape([float(x)]) for x in fine])
ax_sine.plot(fine, values, color="tab:blue", linewidth=1.4)
ax_sine.axhline(optimum_1d - TOLERANCE, color="tab:green", linestyle="--",
                linewidth=1.0, label=f"optimum - {TOLERANCE}")
ax_sine.fill_between(fine, optimum_1d - TOLERANCE, values,
                     where=values >= optimum_1d - TOLERANCE,
                     color="tab:green", alpha=0.6,
                     label=f"target, {width_1d:.3f} wide")
for champion in answers_1d:
    ax_sine.plot([champion.genes[0]], [champion.fitness], "o",
                 color="tab:red", markersize=5)
ax_sine.set_title(f"sine 1-D: the target has width ({share_1d:.2%} of the box)",
                  fontsize=10)
ax_sine.set_xlabel("x")
ax_sine.set_ylabel("fitness")
ax_sine.grid(True, linestyle=":", alpha=0.5)
ax_sine.legend(fontsize=8, loc="lower center")

ys = np.linspace(9.0, 11.0, 4001)
x_slice = 22.5
cusp = np.sin(x_slice) * np.cos(x_slice) - np.power(
    np.abs((x_slice + 50.0) * (ys - 10.0)) / 10.0, 0.1)
ax_cusp.plot(ys, cusp, color="tab:red", linewidth=1.4)
ax_cusp.axhline(optimum_2d - TOLERANCE, color="tab:green", linestyle="--",
                linewidth=1.0, label=f"optimum - {TOLERANCE}")
ax_cusp.set_title(f"book 2-D, slice at x = {x_slice}: the target is a line",
                  fontsize=10)
ax_cusp.set_xlabel("y")
ax_cusp.set_ylabel("fitness")
ax_cusp.grid(True, linestyle=":", alpha=0.5)
ax_cusp.legend(fontsize=8, loc="lower center")

fig.suptitle("Before counting successes, check that success is reachable")
FIGURES.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURES / "success_rate_02_the_verdict.png", dpi=150,
            bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved to {FIGURES}/success_rate_02_the_verdict.png")

