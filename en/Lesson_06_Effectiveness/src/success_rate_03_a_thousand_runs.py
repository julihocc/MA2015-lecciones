"""
Lesson 06 - Step 3: A thousand runs
====================================
NEW IN THIS STEP: success_rate(), standard_error(), wilson_interval(), and
RUNS = 1000.

Step 2 ended with a success rate measured on eight runs - 62.5% - and a
warning about what such a rate is worth: if 62.5% were the true rate, the
count in 8 runs would still wander by 1.4 runs from one experiment to the
next. This step pays that warning off. It runs the experiment a thousand
times, the way Gridin's Chapter 6 does, and replaces the anecdote with a
measurement: a rate, its standard error, a 95% Wilson interval, and the full
distribution of answers behind it.

Two things come out of the thousand runs that no small sample could show.
The 8-run estimate turns out to differ from the 1000-run estimate by almost exactly
the one estimated standard deviation step 2 told us to expect - that one-SE estimate was the
correct instrument all along. And the distribution of best-ever fitness is
not a bell around the optimum: it spikes at the global peak, and it spikes
again at the local optimum Lesson 01 step 7 stumbled into. That failure was
never one unlucky seed. It is a fixed percentage of everything this
algorithm does, and this step measures it.

CHANGES FROM success_rate_02_the_verdict.py
Introduce them in this order:
    1. success_rate()      the Monte Carlo instrument: N runs, count the verdicts
    2. standard_error()    what N buys: sqrt(p(1-p)/N), the wobble of a measured rate
    3. wilson_interval()   a 95% binomial interval that works at 0% and 100%
    4. RUNS = 1000         changed from 8: the book's thousand, at a few seconds

REMOVED FROM success_rate_02_the_verdict.py: the 2-D specimen (book_2d, TWO_D,
on_the_singular_line, and the two-gene branches of the grid instruments). Its
lesson is taught; everything from here on is measured on SINE.

Run it:  python success_rate_03_a_thousand_runs.py

success_rate() repeats verdict() over RUNS seeds; standard_error() is
sqrt(p(1-p)/n). On 1000 runs the observed rate is 79.4% with one-SE 1.3%; the 8-run estimate
missed by 16.9 points. 71 runs (7.1%) die on Lesson 01's local peak at x =
-4.51 — it was never one unlucky seed.
"""
import random
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
# Run i uses seed i, for the whole lesson - so the first eight of these
# thousand runs ARE step 2's experiment, seed for seed.
RUNS = 1000   # --- CHANGED --- (4) from 8: the book's thousand, at a few seconds
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


def sine_landscape(genes: list[float]) -> float:
    """Lesson 01's landscape, unchanged: one gene, one smooth global peak.

    Args:
        genes: a one-element list holding x.

    Returns:
        sin(x) - 0.2 * |x|. Larger is better.

    Example:
        Success on this landscape means within 0.01 of the brute-forced
        optimum +0.705908, a target 1.43% of the box wide.
    """
    return float(np.sin(genes[0]) - 0.2 * abs(genes[0]))


# Ten generations: enough to succeed most of the time and fail often enough
# for the failures to be worth counting.
SINE = Problem("sine 1-D", sine_landscape, -10.0, 10.0, 1, 10)


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


def brute_force_optimum(problem: Problem, points_per_axis: int) -> float:
    """The best value on a regular grid over the whole box.

    Lesson 02 step 5's instrument, back in the one dimension where it is
    affordable: the truth about the landscape, from outside any run.

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
    return float(max(problem.fitness([float(x)]) for x in axis))


# How close to the optimum counts as having found it. Stated out loud, so a
# success rate can be read as "within 0.01 of the optimum" rather than as a
# fact about the algorithm.
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


def within_tolerance_share(problem: Problem, optimum: float,
                           points_per_axis: int) -> float:
    """What fraction of the box scores within TOLERANCE of the optimum?

    Also the hit probability of one blind draw - the benchmark step 4 spends
    the same budget on.

    Args:
        problem, optimum, points_per_axis: as brute_force_optimum.

    Returns:
        Fraction of the grid that scores within TOLERANCE of the
        optimum — the hit probability of one blind draw.

    Example:
        Book 2-D: 0.003197%. Sine 1-D: 1.4299% of the box (0.286 wide).
    """
    axis = np.linspace(problem.low, problem.high, points_per_axis)
    values = np.array([problem.fitness([float(x)]) for x in axis])
    return float(np.mean(values >= optimum - TOLERANCE))


# --- NEW (1) success_rate() ---------------------------------------------------
def success_rate(problem: Problem, runs: int, optimum: float
                 ) -> tuple[list[Individual], int]:
    """The Monte Carlo instrument: run the algorithm `runs` times, judge each
    run's answer from outside, and count the successes.

    It returns the answers themselves, not just the count, because the count
    is never the whole story - the distribution of the failures is where the
    algorithm's character shows.

    Args:
        problem: sine 1-D here.
        runs: 1000.
        optimum: brute-forced.

    Returns:
        successes / runs, using seed 0 .. runs-1.

    Example:
        79.4% ± 1.3%. 71 of 1000 runs (7.1%) die on the local peak
        at x = -4.51.
    """
    answers = [best_ever(run(problem, seed)) for seed in range(runs)]
    successes = sum(verdict(a, optimum) for a in answers)
    return answers, successes
# ------------------------------------------------------------------------------


# --- NEW (2) standard_error() -------------------------------------------------
def standard_error(rate: float, n: int) -> float:
    """The wobble of a measured rate: sqrt(p(1-p)/n).

    A success rate is a coin measured by flipping it. This is how far the
    measured proportion wanders from the true one, one standard deviation -
    and it shrinks with the square root of n, which is why certainty is
    expensive: four times the runs buys only half the wobble.

    Args:
        rate: an observed success fraction.
        n: the number of runs.

    Returns:
        sqrt(p(1-p)/n).

    Example:
        8 runs at 62.5% carry ±17.1 points; 1000 runs at 79.4% carry
        ±1.3. The 8-run estimate missed by 16.9 points, 1.0 of its
        own standard deviations.
    """
    return (rate * (1 - rate) / n) ** 0.5
# ------------------------------------------------------------------------------


# --- NEW (3) wilson_interval() ------------------------------------------------
def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Return the two-sided Wilson interval for a binomial success rate.

    Unlike ``rate +/- standard_error``, this interval remains inside [0, 1]
    and does not collapse to zero width when every observed run succeeds or
    fails. ``z=1.96`` gives the usual approximate 95% interval.
    """
    rate = successes / n
    denominator = 1 + z ** 2 / n
    centre = (rate + z ** 2 / (2 * n)) / denominator
    radius = z / denominator * (
        rate * (1 - rate) / n + z ** 2 / (4 * n ** 2)
    ) ** 0.5
    return centre - radius, centre + radius
# ------------------------------------------------------------------------------


# ===== Part 1: the measurement, and the fate of step 2's estimate ============
optimum = brute_force_optimum(SINE, 20001)
print(f"The target (grid of {20001:,} points): optimum {optimum:+.6f}, "
      f"success means within {TOLERANCE} of it.\n")
print(f"{RUNS:,} runs, population {POPULATION_SIZE}, "
      f"{SINE.generations} generations each...\n")
answers, successes = success_rate(SINE, RUNS, optimum)
rate = successes / RUNS
low, high = wilson_interval(successes, RUNS)
print(f"    {successes} of {RUNS:,} runs met the grid-reference tolerance.")
print(f"    success rate {rate:.1%} +/- {standard_error(rate, RUNS):.1%} "
      f"(one standard error)")
print(f"    95% Wilson interval [{low:.1%}, {high:.1%}]")

# The first eight of these runs are step 2's experiment, seed for seed.
step2_runs = 8
step2_successes = sum(verdict(a, optimum) for a in answers[:step2_runs])
step2_rate = step2_successes / step2_runs
step2_wobble = standard_error(step2_rate, step2_runs)
miss = rate - step2_rate
print(f"\nStep 2 ran the first {step2_runs} of these same seeds and reported "
      f"{step2_rate:.1%} +/- {step2_wobble:.1%}.")
print(f"The 1000-run estimate is {rate:.1%}. The 8-run estimate differs by "
      f"{miss:.1%} -")
print(f"that is {miss / step2_wobble:.1f} of its own standard deviations. The "
      "wobble step 2 computed")
print("was informative about sampling spread; it was not a confidence interval.")

# ===== Part 2: the distribution is not a bell =================================
# Find the landscape's peaks on a fine grid, so the claims below are measured,
# not remembered from Lesson 01.
fine = np.linspace(SINE.low, SINE.high, 200001)
fine_values = np.array([sine_landscape([float(x)]) for x in fine])
is_peak = (fine_values[1:-1] > fine_values[:-2]) \
    & (fine_values[1:-1] > fine_values[2:]) & (fine_values[1:-1] > -0.5)
peak_x, peak_f = fine[1:-1][is_peak], fine_values[1:-1][is_peak]
not_global = np.abs(peak_f - optimum) > TOLERANCE
tallest_local = int(np.argmax(peak_f[not_global]))
local_x, local_f = float(peak_x[not_global][tallest_local]), \
    float(peak_f[not_global][tallest_local])

fits = [a.fitness for a in answers]
at_global = successes
at_local = sum(1 for f in fits if abs(f - local_f) < TOLERANCE)
in_between = RUNS - at_global - at_local
print(f"\nWhere the {RUNS:,} answers landed:")
print(f"    {at_global:>4} runs ({at_global / RUNS:5.1%})  within {TOLERANCE} "
      f"of the dense-grid reference (f = {optimum:+.4f})")
print(f"    {at_local:>4} runs ({at_local / RUNS:5.1%})  died on the local "
      f"peak at x = {local_x:+.2f} (f = {local_f:+.4f})")
print(f"    {in_between:>4} runs ({in_between / RUNS:5.1%})  in between - "
      f"near-misses on the final hill")
print(f"\nLesson 01 step 7 changed one seed and watched the run stall near "
      f"x = {local_x:+.2f}.")
print(f"That was never one unlucky seed: it is {at_local / RUNS:.1%} of "
      f"everything this algorithm does.")
print(f"The worst run of the thousand ended at f = {min(fits):+.4f}.")

# ===== Part 3: what the rate cannot say =======================================
share = within_tolerance_share(SINE, optimum, 20001)
print(f"\nA gene drawn uniformly from the box lands in the target "
      f"{share:.2%} of the time.")
print(f"Each run evaluated about a hundred individuals (step 4 counts them "
      f"exactly)")
print(f"to turn {share:.2%} into {rate:.1%}. Whether those evaluations were "
      f"well spent is not")
print("a question the rate can answer. Step 4 counts the cost.")

# The picture: the book's histogram, annotated with what the spikes are.
fig, ax = plt.subplots(figsize=(8.5, 4.2))
ax.hist(fits, bins=16, facecolor="tab:blue", alpha=0.75, edgecolor="black")
ax.axvline(optimum - TOLERANCE, color="tab:green", linestyle="--",
           linewidth=1.2, label=f"optimum - {TOLERANCE}")
ax.annotate(f"{at_global} of {RUNS:,} runs land here",
            xy=(optimum, at_global), xytext=(0.32, 0.72),
            textcoords="axes fraction", fontsize=9,
            arrowprops=dict(arrowstyle="->", color="black"))
ax.annotate(f"{at_local} runs died on\nthe local peak",
            xy=(local_f, at_local), xytext=(0.13, 0.45),
            textcoords="axes fraction", fontsize=9,
            arrowprops=dict(arrowstyle="->", color="black"))
ax.set_title(f"Best-ever fitness over {RUNS:,} independent runs "
             f"(population {POPULATION_SIZE}, {SINE.generations} generations)")
ax.set_xlabel("best fitness reached")
ax.set_ylabel("number of runs")
ax.grid(True, axis="y", linestyle=":", alpha=0.5)
ax.legend(fontsize=9)
FIGURES.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURES / "success_rate_03_a_thousand_runs.png", dpi=150,
            bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved to {FIGURES}/success_rate_03_a_thousand_runs.png")

