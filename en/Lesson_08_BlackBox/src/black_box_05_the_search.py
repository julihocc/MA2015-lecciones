"""
Lesson 08 - Step 5: The search, and an answer that will not sit still
=====================================================================
NEW IN THIS STEP: select_rank_with_elite(), the generational loop, and the
diagnostics.

Every piece is finally on the table: a chromosome that cannot be illegal
(step 3), operators tuned gene by gene (step 4). This step assembles the
book's genetic algorithm and lets it run. The selection is Lesson 03's rank
selection with an elite of 3 - the book's choice - and the loop is Lesson
02's: select, cross pairs with probability 0.8, mutate with probability 0.2.
The book runs this loop for 1,000 generations; 100 is enough to make the
point, so 100 it is.

And the point is uncomfortable. The run reports a best fitness of 2.278e-09
at x = -10.9536 - a chromosome parked far outside the [-10, 10] interval the
initial population was drawn from, in a direction the grid of step 2 never
reached. It spent 20,204 evaluations to get there: 0.02% of the book-grid
budget from step 2. By every measure the course has used so far, this is a
success. And yet the best-ever curve is still rising when the budget runs
out, and 97.5% of the final population lives outside the interval we started
them in. The algorithm is unanimous about WHERE to go. Step 6 asks whether
it agrees with itself about WHAT it found.

GONE FROM THIS STEP: the repair census and the drift demo (step 4's point is
made), and the one-shot operator showcase.

CHANGES FROM black_box_04_the_operators.py
Introduce them in this order:
    1. select_rank_with_elite()   Lesson 03's method, elite of 3: the book's choice
    2. the generational loop      100 generations (the book ran 1,000)
    3. the diagnostics            best vs average, the x census, the evaluation census

Run it:  python black_box_05_the_search.py

Best fitness reaches 2.278e-09, the curve is still rising, and 97.5% of the
final population has left the initial x interval.
"""
import math
import random
from pathlib import Path
from typing import Any, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

SEED = 19
POPULATION_SIZE = 200
GENERATIONS = 100                 # the book used 1,000; 100 makes the point
CROSSOVER_PROB = 0.8
MUTATION_PROB = 0.2
ELITE_SIZE = 3
A_MIN, A_MAX = 0.0, 1.0
B_MIN, B_MAX = 0.0, 1.0
X_MIN, X_MAX = -100.0, 100.0
N_SET = range(0, 21)
FUN_SET = ("sin", "cos")
INIT_X_MIN, INIT_X_MAX = -10.0, 10.0
FIGURES = Path(__file__).resolve().parent.parent / "figures"


def complicated_one(a: float, b: float, x: float, n: int, fun_name: str) -> float:
    """The black box, unchanged from step 1. Still nobody has read it.

    Args:
        a: real input on [0, 1].
        b: real input on [0, 1].
        x: real input on [-100, 100].
        n: integer input in 0..20.
        fun_name: ``"sin"`` or ``"cos"``.

    Returns:
        One float stored as ``Individual.fitness``.

    Example:
        This run's champion reports ``2.278e-09`` at ``x = -10.9536``.
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
        The nearest legal integer. Ties go to the first minimum ``min`` finds.

    Example:
        Step 4 measured 41.5% of n-mutations snapping back; the same snap
        still runs inside every constructor call here.
    """
    return min(allowed, key=lambda candidate: abs(candidate - value))


class Individual:
    """A five-gene chromosome that cannot exist in an illegal state.

    All repair lives in the constructor, so the operators below can treat
    every gene as a real number. The class counts its own instantiations:
    from this step onwards that count IS the fitness-evaluation budget.

    Args:
        gene_list: five raw values ``[a, b, x, n, fun_name]``.

    Example:
        This run spends 20,204 constructor calls to reach fitness
        ``2.278e-09``.
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

    Example:
        97.5% of the final population has left that initial interval.
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

    Example:
        Step 4 found 41.5% of n-mutations snapping back; that silence is
        still inside every mutated child here.
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


# --- NEW (1) select_rank_with_elite() -----------------------------------------
def select_rank_with_elite(population: List[Individual],
                           elite_size: int) -> List[Individual]:
    """Lesson 03's rank selection with the top `elite_size` copied through.

    The book's selection, unchanged: sort best first, give position i the
    weight 1 - i/N, and spin the wheel N - elite_size times. The elite slots
    are filled before any draw, so the best individual cannot be lost - the
    cheapest guarantee in the course, and on this landscape the load-bearing
    one.

    Args:
        population: the current generation, any order.
        elite_size: how many of the current best are copied through. 3 here.

    Returns:
        A new list of length ``len(population)``. The first ``elite_size``
        entries are the current best, not clones.

    Example:
        With this elite the run still reports ``2.278e-09`` and a curve
        that is rising when the 100-generation budget ends.
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
# ------------------------------------------------------------------------------


# --- NEW (2) the generational loop --------------------------------------------
# Lesson 02's loop with the book's knobs: select a new population, cross the
# pairs with probability 0.8, mutate the survivors with probability 0.2.
random.seed(SEED)
Individual.counter = 0
population = [create_random() for _ in range(POPULATION_SIZE)]
best_ever = max(population, key=lambda ind: ind.fitness)
ever_history: List[float] = []
average_history: List[float] = []
for generation in range(GENERATIONS):
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
    ever_history.append(best_ever.fitness)
    average_history.append(sum(ind.fitness for ind in population)
                           / len(population))
# ------------------------------------------------------------------------------

print("=" * 78)
print("Lesson 08 - Step 5: the search, and an answer that will not sit still")
print("=" * 78)
print(f"\n{GENERATIONS} generations, population {POPULATION_SIZE}, "
      f"pc = {CROSSOVER_PROB}, pm = {MUTATION_PROB}, elite {ELITE_SIZE}")
print("-" * 78)
print(f"  best ever: {best_ever}  fitness {best_ever.fitness:.3e}")

# --- NEW (3) the diagnostics --------------------------------------------------
# Three readings of the same run. None of them is the verdict by itself;
# together they are the question step 6 has to answer.
print("\nThe diagnostics")
print("-" * 78)

# (a) best vs average: the champion stands how far above its own generation?
final_average = average_history[-1]
print(f"  best ever:              {best_ever.fitness:+.3e}")
print(f"  final population mean:  {final_average:+.3e}")
print(f"  best-ever curve at generation 1:   {ever_history[0]:+.3e}")
print(f"  best-ever curve at generation {GENERATIONS}: {ever_history[-1]:+.3e}"
      "  (still rising)")

# (b) the x census: where does the final population actually live?
final_xs = [ind.gene_list[2] for ind in population]
outside = sum(1 for value in final_xs if not INIT_X_MIN <= value <= INIT_X_MAX)
print(f"\n  final population x range: [{min(final_xs):+.2f}, {max(final_xs):+.2f}]")
print(f"  outside the initial [{INIT_X_MIN:+.0f}, {INIT_X_MAX:+.0f}] interval: "
      f"{outside}/{len(final_xs)} ({outside / len(final_xs):.1%})")

# (c) the evaluation census: what did the answer cost?
print(f"\n  fitness evaluations used: {Individual.counter:,}")
print(f"  book-grid budget (step 2): 105,000,000 "
      f"({Individual.counter / 105_000_000:.2%} of it spent)")
# ------------------------------------------------------------------------------

fig, (ax_curve, ax_census) = plt.subplots(1, 2, figsize=(11.5, 4.0))
ax_curve.plot(range(1, GENERATIONS + 1), ever_history, color="tab:blue",
              linewidth=1.6, label="best ever")
ax_curve.set_yscale("log")
ax_curve.set_title("The success curve: best fitness so far", fontsize=10)
ax_curve.set_xlabel("generation")
ax_curve.set_ylabel("fitness (log scale)")
ax_curve.grid(True, linestyle=":", alpha=0.5)
ax_curve.legend(fontsize=8)

ax_census.hist(final_xs, bins=40, color="tab:blue", edgecolor="white")
ax_census.axvspan(INIT_X_MIN, INIT_X_MAX, color="tab:green", alpha=0.15,
                  label="initial x range [-10, 10]")
ax_census.set_title("Where the final population lives", fontsize=10)
ax_census.set_xlabel("x gene")
ax_census.set_ylabel("count")
ax_census.legend(fontsize=8)
ax_census.grid(True, axis="y", linestyle=":", alpha=0.5)

fig.suptitle("A textbook success - that has left the map we gave it")
FIGURES.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURES / "black_box_05_the_search.png", dpi=150,
            bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved to {FIGURES}/black_box_05_the_search.png")

