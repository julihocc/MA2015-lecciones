"""
Lesson 07 - Step 1: The book's method, faithfully
==================================================
NEW IN THIS STEP: nothing is new to the students. This is Lessons 01-06's
algorithm, unchanged, driven the way Gridin's Chapter 7 drives it: pick a
knob, run one setting once, look at the answer, pick a winner.

Chapter 7 is the book's laboratory chapter - 42 figures, three knobs
(crossover probability, mutation probability, population size), each turned
on its own landscape, each setting shown as one run, generation by
generation. It is the natural way to tune, and it is exactly the method
Lesson 06 demolished. This step keeps the book's protocol and its numbers -
population 16, mutation probability 0.2, crossover probabilities 0.0, 0.4,
0.7, seed 63 - on the course's landscape, and reads it the way the book
reads it. Then it reads it seven more times.

The landscape is Lesson 01's sine, one gene, optimum +0.705908 known by
brute force, "success" meaning within 0.01 of it. One landscape for the
whole lesson, on purpose: the book turns each knob on a different function,
so its chapter never compares anything to anything. Here every setting
faces the same problem, so the knobs can be compared - first badly (this
step), then properly (the rest of the lesson).

Run it:  python tuning_01_single_runs.py

run() is the course GA driven the book's way: one setting, one seed, one
answer. history() is the convergence curve. One run per setting crowns pc =
0.7 by 0.0021; over 8 seeds the win counts go 0/3/5, the tightest split is
0.000004 (seed 3) and the widest 0.6284 (seed 1).
"""
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

# The algorithm, exactly as Lessons 01-06 left it.
MUTATION_MU, MUTATION_SIGMA = 0.0, 1.0
TOURNAMENT_SIZE, BLEND_ALPHA = 3, 1.0
# The book's protocol: its population, its mutation probability, its three
# crossover settings, its seed.
POPULATION_SIZE = 16
MUTATION_PROBABILITY = 0.2
CROSSOVER_SETTINGS = (0.0, 0.4, 0.7)
SEED = 63
SEEDS = 8
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


SINE = Problem("sine 1-D", sine_landscape, -10.0, 10.0, 1, 10)


class Individual:
    """One candidate solution, carrying the problem that judges it.


    Example:
        Each construction is one fitness call; the budget lessons count
        them because a knob that raises success also raises spend.
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


def run(problem: Problem, seed: int, crossover_probability: float
        ) -> list[list[Individual]]:
    """One complete run, the book's way: one setting, one seed, one answer.

    Args:
        problem: the sine landscape.
        seed: the book's seed 63 in part 1; seeds 0..7 in part 2.
        crossover_probability: the knob the book turns.

    Returns:
        The list of generations.

    Example:
        One run crowns pc = 0.7 by 0.0021; over 8 seeds the ranking
        is a coin flip (win counts 0/3/5).
    """
    random.seed(seed)
    population = [Individual([random.uniform(problem.low, problem.high)
                              for _ in range(problem.genes)], problem)
                  for _ in range(POPULATION_SIZE)]
    generations = [population]
    for _ in range(problem.generations):
        selected = select_tournament(population)
        crossed: list[Individual] = []
        for parent1, parent2 in zip(selected[::2], selected[1::2]):
            if random.random() < crossover_probability:
                crossed.extend(crossover(parent1, parent2))
            else:
                crossed.extend([parent1, parent2])
        population = [mutate(ind) if random.random() < MUTATION_PROBABILITY
                      else ind for ind in crossed]
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


def history(generations: list[list[Individual]]) -> list[float]:
    """Best-ever fitness after each generation - the convergence curve.

    Args:
        generations: every generation of one run.

    Returns:
        Per-generation curves used to draw the figure (average / best
        / best-ever, or best-ever only, depending on the lesson).

    Example:
        The tracked Lesson 06 run gains +0.5822; eight seeds of the
        same configuration spread 0.8946.
    """
    ever, rows = -float("inf"), []
    for population in generations:
        ever = max(ever, max(ind.fitness for ind in population))
        rows.append(ever)
    return rows


# ===== Part 1: the book's method, once ========================================
print(f"The book's protocol: population {POPULATION_SIZE}, mutation "
      f"{MUTATION_PROBABILITY}, seed {SEED},")
print(f"one run per crossover setting, {SINE.generations} generations each:\n")
print("  crossover | best ever found")
print("  ----------+-----------------")
single = {}
for pc in CROSSOVER_SETTINGS:
    champion = best_ever(run(SINE, SEED, pc))
    single[pc] = champion.fitness
    print(f"  {pc:>9} | {champion.fitness:+15.4f}")
winner = max(single, key=single.get)
print(f"\nVerdict, book style: crossover probability {winner} wins. Write it "
      "down and move on.")
spread = max(single.values()) - min(single.values())
print(f"(The margin over the worst setting: {spread:.4f}.)")

# ===== Part 2: the same protocol, seven more times ============================
print(f"\nThe same protocol, {SEEDS} seeds:\n")
header = "  seed |" + "|".join(f"  pc = {pc:>3}  " for pc in CROSSOVER_SETTINGS)
print(header + "  winner")
print("  -----+" + "+".join("-" * 12 for _ in CROSSOVER_SETTINGS) + "--------")
wins = {pc: 0 for pc in CROSSOVER_SETTINGS}
curves = {pc: [] for pc in CROSSOVER_SETTINGS}
gaps, shortfalls = [], []
for seed in range(SEEDS):
    answers = {}
    for pc in CROSSOVER_SETTINGS:
        generations = run(SINE, seed, pc)
        answers[pc] = best_ever(generations).fitness
        curves[pc].append(history(generations))
    round_winner = max(answers, key=answers.get)
    wins[round_winner] += 1
    placed = sorted(answers.values(), reverse=True)
    gaps.append(placed[0] - placed[1])
    shortfalls.append(placed[0] - placed[-1])
    row = "  ".join(f"{answers[pc]:+9.4f} " for pc in CROSSOVER_SETTINGS)
    print(f"  {seed:>4} | {row}  pc = {round_winner}")
print("\nWin counts: " + ", ".join(f"pc = {pc}: {wins[pc]} of {SEEDS}"
                                   for pc in CROSSOVER_SETTINGS))
tightest = min(range(SEEDS), key=lambda s: gaps[s])
roughest = max(range(SEEDS), key=lambda s: shortfalls[s])
print(f"\nRead the table. On seed {tightest} the top two settings are split by "
      f"{gaps[tightest]:.6f}; on seed {roughest} the same")
print(f"three settings are spread over {shortfalls[roughest]:.4f}. The knob "
      f"did not change - the seed did.")
print("A method that cannot reproduce its own ranking on the same problem is "
      "not measuring")
print("the knob. It is reading noise.")
print("\nLesson 06 built the instrument for exactly this. Step 2 picks it up.")

# The picture: one panel per seed, three convergence curves each. The ranking
# that looked like a fact on the book's single run flips from panel to panel.
fig, axes = plt.subplots(2, 4, figsize=(11.5, 4.6), sharex=True, sharey=True)
for seed, ax in enumerate(axes.flat):
    for pc in CROSSOVER_SETTINGS:
        ax.plot(curves[pc][seed], linewidth=1.4, label=f"pc = {pc}")
    ax.set_title(f"seed {seed}", fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.5)
    if seed % 4 == 0:
        ax.set_ylabel("best so far", fontsize=8)
axes[1, 1].legend(fontsize=7, loc="lower right")
fig.suptitle("One run per setting, eight times: the ranking is unstable")
FIGURES.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURES / "tuning_01_single_runs.png", dpi=150,
            bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved to {FIGURES}/tuning_01_single_runs.png")

