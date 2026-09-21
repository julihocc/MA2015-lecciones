"""
Lesson 08 - Step 6: The verdict, twelve times over
==================================================
NEW IN THIS STEP: the Monte Carlo and the two readings.

Step 5's run looked like a success, and one run is an anecdote. This step
repeats the whole search twelve times, from twelve seeds, and lays the
twelve champions side by side. Nothing about the algorithm changes: same
chromosome, same operators, same knobs, same 100 generations. Only the seed
moves.

The verdict splits in two. WHERE: every one of the twelve champions sits at
x = -10.95 - twelve independent searches, all started inside [-10, 10], all
walk to the same spot outside it and park within 0.008 of each other. WHAT:
the fitnesses they report run from 5.8e-09 to 1.2e-04, a factor of 21,000
between the weakest and the strongest answer to the same question. An
algorithm that agrees with itself about where to go, and disagrees by four
orders of magnitude about what it found there, is not converging on a value.
It is climbing something with no visible ceiling. Step 7 opens the box.

GONE FROM THIS STEP: the single-run diagnostics (one run is no longer the
unit of evidence) and the success-curve figure.

CHANGES FROM black_box_05_the_search.py
Introduce them in this order:
    1. run_search()        step 5's search as a function: a seed in, a champion out
    2. the Monte Carlo     twelve runs, twelve seeds, twelve champions
    3. the two readings    where the champions sit vs what they report

Run it:  python black_box_06_the_verdict.py

Champions occupy a narrow x interval while their fitness differs by
21,504×, which is inconsistent with convergence to a finite maximum.
"""
import math
import random
from pathlib import Path
from typing import Any, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

POPULATION_SIZE = 200
GENERATIONS = 100
CROSSOVER_PROB = 0.8
MUTATION_PROB = 0.2
ELITE_SIZE = 3
RUNS = 12
A_MIN, A_MAX = 0.0, 1.0
B_MIN, B_MAX = 0.0, 1.0
X_MIN, X_MAX = -100.0, 100.0
N_SET = range(0, 21)
FUN_SET = ("sin", "cos")
INIT_X_MIN, INIT_X_MAX = -10.0, 10.0
FIGURES = Path(__file__).resolve().parent.parent / "figures"


def complicated_one(a: float, b: float, x: float, n: int, fun_name: str) -> float:
    """The black box, unchanged from step 1. One step away from being read.

    Args:
        a: real input on [0, 1].
        b: real input on [0, 1].
        x: real input on [-100, 100].
        n: integer input in 0..20.
        fun_name: ``"sin"`` or ``"cos"``.

    Returns:
        One float stored as ``Individual.fitness``.

    Example:
        Twelve champions report fitnesses that differ by ``21,504×`` while
        sitting in a narrow ``x`` interval.
    """
    total = 0.0
    for _ in range(10, 10 + n + 1):
        if fun_name == "cos":
            trig = math.cos(math.log2(b ** n + 1) * math.pi * x) / (n + 1) + math.sin(x) ** n
        elif fun_name == "sin":
            trig = math.sin(math.log2(b ** n + 1) * math.pi * x) / (n + 1) + math.cos(x) ** n
        else:
            raise ValueError(f"Unknown function: {fun_name}")
        resid = trig - math.log2(n + 1)
        div = ((n + 1) ** 2) * (1 + a + b) * (120 - x ** 2) * resid + 1 / 2
        total += ((x * n + math.log(n + 1)) / div) / (10 ** 15)
    return total


def clamp(gene: float, low: float, high: float) -> float:
    """Push a real gene back inside its interval.

    Args:
        gene: proposed real value, legal or not.
        low, high: inclusive edges of that gene's declared interval.

    Returns:
        ``gene`` if it is already inside; otherwise the nearer edge.
    """
    return max(low, min(high, gene))


def closest(value: float, allowed: Sequence[int]) -> int:
    """Snap a real number onto the nearest legal point of a lattice.

    Args:
        value: the real an operator proposed.
        allowed: the legal lattice, here ``range(0, 21)``.

    Returns:
        The nearest legal integer.
    """
    return min(allowed, key=lambda candidate: abs(candidate - value))


class Individual:
    """A five-gene chromosome that cannot exist in an illegal state.

    The class counts its own instantiations: that count IS the
    fitness-evaluation budget of whichever run is underway.

    Args:
        gene_list: five raw values ``[a, b, x, n, fun_name]``.

    Example:
        Twelve independent constructors still park their champions in a
        narrow ``x`` interval whose fitnesses differ by ``21,504×``.
    """

    counter = 0

    def __init__(self, gene_list: Sequence[Any]) -> None:
        """Repair every gene, then evaluate once.

        Args:
            gene_list: five raw values ``[a, b, x, n, fun_name]``.

        Returns:
            None. Fitness is stored on ``self.fitness``.
        """
        self.__class__.counter += 1
        a_raw, b_raw, x_raw, n_raw, fun_name = gene_list
        self.gene_list: List[Any] = [
            clamp(float(a_raw), A_MIN, A_MAX),
            clamp(float(b_raw), B_MIN, B_MAX),
            clamp(float(x_raw), X_MIN, X_MAX),
            closest(float(n_raw), N_SET),
            fun_name,
        ]
        self.fitness: float = complicated_one(*self.gene_list)

    def __str__(self) -> str:
        a, b, x, n, fun_name = self.gene_list
        return f"[a={a:.4f} b={b:.4f} x={x:+.4f} n={n:2d} fun={fun_name}]"


def create_random() -> Individual:
    """One random individual, each gene drawn the way its own type allows.

    Returns:
        A legal ``Individual``. ``x`` is drawn on [-10, 10], not [-100, 100].
    """
    return Individual([
        random.uniform(A_MIN, A_MAX),
        random.uniform(B_MIN, B_MAX),
        random.uniform(INIT_X_MIN, INIT_X_MAX),
        random.choice(N_SET),
        random.choice(FUN_SET),
    ])


def blend(g1: float, g2: float, alpha: float) -> tuple[float, float]:
    """Lesson 04's blend crossover for one pair of raw gene values.

    Args:
        g1, g2: parental values of one gene.
        alpha: how far past the parents the interval extends.

    Returns:
        Two independent samples from the extended interval, not yet repaired.
    """
    low = min(g1, g2) - alpha * abs(g2 - g1)
    high = max(g1, g2) + alpha * abs(g2 - g1)
    return (low + random.random() * (high - low),
            low + random.random() * (high - low))


def crossover(parent1: Individual, parent2: Individual
              ) -> tuple[Individual, Individual]:
    """The book's per-gene crossover: one coin per gene, one alpha per gene.

    Args:
        parent1, parent2: legal individuals. Their genes are copied.

    Returns:
        Two new ``Individual``s. Each constructor call is one evaluation.
    """
    a1, b1, x1, n1, fun1 = parent1.gene_list
    a2, b2, x2, n2, fun2 = parent2.gene_list
    if random.random() < 0.5:
        a1, a2 = blend(a1, a2, 0.3)
    if random.random() < 0.5:
        b1, b2 = blend(b1, b2, 0.5)
    if random.random() < 0.5:
        x1, x2 = blend(x1, x2, 1.0)
    if random.random() < 0.5:
        n1, n2 = blend(n1, n2, 0.5)
    if random.random() < 0.5:
        fun1, fun2 = fun2, fun1
    return (Individual([a1, b1, x1, n1, fun1]),
            Individual([a2, b2, x2, n2, fun2]))


def mutate(individual: Individual) -> Individual:
    """The book's per-gene mutation: one coin per gene, one sigma per gene.

    Args:
        individual: the parent. Its genes are not written in place.

    Returns:
        A new ``Individual``. The constructor may silently undo the step.
    """
    a, b, x, n, fun_name = individual.gene_list
    if random.random() < 0.5:
        a += random.gauss(0.0, 0.2)
    if random.random() < 0.5:
        b += random.gauss(0.0, 0.2)
    if random.random() < 0.5:
        x += random.gauss(0.0, 1.0)
    if random.random() < 0.5:
        n += random.gauss(0.0, 1.0)
    if random.random() < 0.5:
        fun_name = random.choice(FUN_SET)
    return Individual([a, b, x, n, fun_name])


def select_rank_with_elite(population: List[Individual],
                           elite_size: int) -> List[Individual]:
    """Lesson 03's rank selection with the top `elite_size` copied through.

    Args:
        population: the current generation, any order.
        elite_size: how many of the current best are copied through. 3 here.

    Returns:
        A new list of length ``len(population)``.
    """
    ordered = sorted(population, key=lambda ind: -ind.fitness)
    size = len(population)
    weights = [1 - i / size for i in range(size)]
    total = sum(weights)
    selected: List[Individual] = list(ordered[:elite_size])
    for _ in range(size - elite_size):
        point = random.random() * total
        cumulative = 0.0
        for ind, weight in zip(ordered, weights):
            cumulative += weight
            if cumulative > point:
                selected.append(ind)
                break
    return selected


# --- CHANGED --- step 5's generational loop becomes a function: a seed goes
# in, a champion (and its evaluation bill) comes out. Nothing else moves.
def run_search(seed: int) -> tuple[Individual, int]:
    """One full search: 100 generations from `seed`, best-ever returned.

    Args:
        seed: the only input that differs across the twelve runs.

    Returns:
        ``(champion, evaluations)``. Evaluations equal ``Individual.counter``.

    Example:
        Twelve seeds park in a narrow ``x`` interval while fitness differs
        by ``21,504×``.
    """
    random.seed(seed)
    Individual.counter = 0
    population = [create_random() for _ in range(POPULATION_SIZE)]
    best_ever = max(population, key=lambda ind: ind.fitness)
    for _ in range(GENERATIONS):
        selected = select_rank_with_elite(population, ELITE_SIZE)
        crossed: List[Individual] = []
        for parent1, parent2 in zip(selected[::2], selected[1::2]):
            if random.random() < CROSSOVER_PROB:
                crossed.extend(crossover(parent1, parent2))
            else:
                crossed.extend([parent1, parent2])
        population = [mutate(ind) if random.random() < MUTATION_PROB else ind
                      for ind in crossed]
        generation_best = max(population, key=lambda ind: ind.fitness)
        if generation_best.fitness > best_ever.fitness:
            best_ever = generation_best
    return best_ever, Individual.counter


print("=" * 78)
print("Lesson 08 - Step 6: the verdict, twelve times over")
print("=" * 78)

# --- NEW (2) the Monte Carlo --------------------------------------------------
# Twelve independent searches, seeds 0..11. Same algorithm, same knobs; only
# the seed moves. Each line below is a whole step-5 run compressed to its
# champion.
champions: List[Individual] = []
total_evaluations = 0
print(f"\n{RUNS} runs x {GENERATIONS} generations (population "
      f"{POPULATION_SIZE}, pc = {CROSSOVER_PROB}, pm = {MUTATION_PROB})")
print("-" * 78)
print(f"  {'seed':>4}  {'champion':<44}  {'fitness':>10}  {'evals':>7}")
for seed in range(RUNS):
    champion, evaluations = run_search(seed)
    champions.append(champion)
    total_evaluations += evaluations
    print(f"  {seed:>4}  {champion!s:<44}  {champion.fitness:>10.3e}  "
          f"{evaluations:>7,}")
print(f"  total fitness evaluations: {total_evaluations:,}")
# ------------------------------------------------------------------------------

# --- NEW (3) the two readings -------------------------------------------------
# Reading one, WHERE: the x genes of the twelve champions.
# Reading two, WHAT: the fitnesses they report.
champion_xs = [ind.gene_list[2] for ind in champions]
champion_fits = [ind.fitness for ind in champions]
x_spread = max(champion_xs) - min(champion_xs)
fit_ratio = max(champion_fits) / min(champion_fits)
print("\nThe two readings")
print("-" * 78)
print(f"  WHERE the champions sit:  x in [{min(champion_xs):+.4f}, "
      f"{max(champion_xs):+.4f}]  (spread {x_spread:.4f})")
print(f"  WHAT they report:         fitness in [{min(champion_fits):.3e}, "
      f"{max(champion_fits):.3e}]  (ratio {fit_ratio:,.0f}x)")
print("\n  Unanimous about where, four orders of magnitude apart about what.")
print("  A search converging on a maximum does not do that. A search climbing")
print("  something unbounded does exactly that - and step 7 opens the box.")
# ------------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.scatter(champion_xs, champion_fits, s=60, color="tab:blue", zorder=3)
for seed, ind in enumerate(champions):
    ax.annotate(f"{seed}", (ind.gene_list[2], ind.fitness),
                textcoords="offset points", xytext=(5, 5), fontsize=8,
                color="tab:gray")
ax.set_yscale("log")
ax.set_xlabel("champion's x gene")
ax.set_ylabel("champion's fitness (log scale)")
ax.set_title(f"Twelve champions: unanimous on where "
             f"(spread {x_spread:.3f}), {fit_ratio:,.0f}x apart on what",
             fontsize=10)
ax.grid(True, linestyle=":", alpha=0.5)
FIGURES.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURES / "black_box_06_the_verdict.png", dpi=150,
            bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved to {FIGURES}/black_box_06_the_verdict.png")

