"""
Lesson 08 - Step 4: The operators, and what silent repair costs
================================================================
NEW IN THIS STEP: crossover(), mutate(), the repair census, and the drift
demo.

Step 3 put all legality in the constructor, so the operators can be written
as if every gene were a real number. This step writes them - the book's
design, and a deliberate one: every gene gets its own coin and its own step
size, because the genes are different kinds of object. Blend alpha 0.3 for
a, 0.5 for b, 1.0 for x; gaussian sigma 0.2 for a and b, 1.0 for x and n;
the label gene mutates by re-choosing, the only mutation a set admits.

Then the bill step 3 promised. Repair is silent, and silence has a price:
measure how often an n-mutation actually moves n. The answer is 58.5% -
the other 41.5% of n-mutations are handed back the same n by the snap, and
no one is told. And at the limit (the book's impossible-mutation example):
make the lattice coarser than the operator's step and mutation goes
ABSOLUTELY silent - a thousand mutations, and the discrete gene is exactly
where it started, while its real-valued twin drifts away. The operator did
not fail loudly. It just never happened.

GONE FROM THIS STEP: repair_report() and the illegal-gene showcase (step 3's
point is made), and the initial-population census (step 3's figure).

CHANGES FROM black_box_03_the_chromosome.py
Introduce them in this order:
    1. crossover()         the book's per-gene coins: each gene, its own alpha
    2. mutate()            per-gene sigmas; n mutated as a real, then snapped
    3. the repair census   count what the constructor silently undid
    4. the drift demo      a lattice coarser than the step: absolute silence

Run it:  python black_box_04_the_operators.py

41.5% of forced integer mutations snap back unchanged; a lattice coarser
than the step can silence mutation completely.
"""
import math
import random
from pathlib import Path
from typing import Any, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

SEED = 19
POPULATION_SIZE = 200
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
        One float stored as ``Individual.fitness`` after repair.
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

    Example:
        Step 3 stored ``a = 1.7`` as ``1.0``; operators below rely on the
        same silent push-back.
    """
    return max(low, min(high, gene))


def closest(value: float, allowed: Sequence[int]) -> int:
    """Snap a real number onto the nearest legal point of a lattice.

    Args:
        value: the real an operator proposed.
        allowed: the legal lattice. ``N_SET`` here; spacing 2 in the drift demo.

    Returns:
        The nearest legal integer.

    Example:
        41.5% of forced n-mutations with sigma 1.0 snap back unchanged;
        a spacing-2 lattice silences a unit-step operator completely.
    """
    return min(allowed, key=lambda candidate: abs(candidate - value))


class Individual:
    """A five-gene chromosome that cannot exist in an illegal state.

    All repair lives in the constructor, so the operators below can treat
    every gene as a real number. The class counts its own instantiations:
    from step 5 onwards that count IS the fitness-evaluation budget.

    Args:
        gene_list: five raw values ``[a, b, x, n, fun_name]``.

    Example:
        Silent snap-back is why 41.5% of forced n-mutations change nothing.
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
        Step 3's initial population covered only 10% of declared ``x``;
        the operators here start from that same draw.
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
        alpha: how far past the parents the interval extends. 0.3 for a,
            0.5 for b and n, 1.0 for x.

    Returns:
        Two independent samples from the extended interval, not yet repaired.

    Example:
        n is blended as a real; the constructor then snaps it, which is
        where 41.5% of later n-mutations disappear.
    """
    low = min(g1, g2) - alpha * abs(g2 - g1)
    high = max(g1, g2) + alpha * abs(g2 - g1)
    return (low + random.random() * (high - low),
            low + random.random() * (high - low))


# --- NEW (1) crossover() ------------------------------------------------------
def crossover(parent1: Individual, parent2: Individual
              ) -> tuple[Individual, Individual]:
    """The book's per-gene crossover: one coin per gene, one alpha per gene.

    Lesson 04 crossed every gene with the same operator. Here x gets the full
    blend(alpha = 1.0) the course has used all along, a and b get gentler
    blends, n is blended as a real and left for the constructor to snap, and
    the label gene can only be swapped - there is no midpoint between 'sin'
    and 'cos', so crossover for a set is an exchange or nothing.

    Args:
        parent1, parent2: legal individuals. Their genes are copied, not
            mutated in place.

    Returns:
        Two new ``Individual``s. Each constructor call is one evaluation.

    Example:
        The label gene is swapped or left; there is no midpoint between
        ``"sin"`` and ``"cos"``.
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
# ------------------------------------------------------------------------------

# --- NEW (2) mutate() ---------------------------------------------------------
def mutate(individual: Individual) -> Individual:
    """The book's per-gene mutation: one coin per gene, one sigma per gene.

    The real genes get gaussian steps sized to their intervals. The lattice
    gene gets the same treatment as a real - n + gauss(0, 1) - and the
    constructor snaps the result back onto the lattice. That last line is
    where the repair census below finds its numbers.

    Args:
        individual: the parent. Its genes are not written in place.

    Returns:
        A new ``Individual``. Identity with the parent means every coin
        missed or was repaired away.

    Example:
        41.5% of forced n-mutations snap back unchanged; a spacing-2
        lattice can silence the operator completely.
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
# ------------------------------------------------------------------------------

print("=" * 78)
print("Lesson 08 - Step 4: the operators, and what silent repair costs")
print("=" * 78)

# One crossover, one mutation, so the per-gene coins are visible once.
random.seed(SEED)
parent1, parent2 = create_random(), create_random()
child1, child2 = crossover(parent1, parent2)
print("\nOne crossover (coins and alphas differ per gene)")
print("-" * 78)
print(f"  parent 1: {parent1}")
print(f"  parent 2: {parent2}")
print(f"  child 1:  {child1}")
print(f"  child 2:  {child2}")
print(f"\nOne mutation of parent 1: {mutate(parent1)}")

# --- NEW (3) the repair census ------------------------------------------------
# Force the n-mutation to fire, and count how often n actually changes.
# A gaussian with sigma 1.0 lands within +-0.5 of its centre 38.3% of the
# time, and every one of those lands back on the n it started from - the
# snap is invisible from outside.
TRIALS = 5000
random.seed(2)
moved = 0
for _ in range(TRIALS):
    n_before = random.choice(N_SET)
    n_after = closest(n_before + random.gauss(0.0, 1.0), N_SET)
    if n_after != n_before:
        moved += 1
snapped_back = TRIALS - moved
print(f"\nThe repair census: {TRIALS} forced n-mutations, sigma = 1.0")
print("-" * 78)
print(f"  n actually changed:     {moved:5d}  ({moved / TRIALS:.1%})")
print(f"  silently snapped back:  {snapped_back:5d}  ({snapped_back / TRIALS:.1%})")

# And the whole operator: of 1000 full mutations, how many individuals come
# out gene-identical - every coin either missed or was repaired away?
random.seed(1)
population = [create_random() for _ in range(1000)]
identical = sum(1 for ind in population
                if mutate(ind).gene_list == ind.gene_list)
print(f"\n  Of 1000 full mutate() calls, {identical} returned a gene-identical "
      f"individual ({identical / 1000:.1%}).")
print("  Every one of those cost a fitness evaluation and changed nothing. "
      "Repair does not")
print("  announce itself; it just makes the operator quieter than its "
      "parameters say.")
# ------------------------------------------------------------------------------

# --- NEW (4) the drift demo ---------------------------------------------------
# The book's impossible-mutation example, and the limit case of the census:
# a lattice of spacing 2, an operator that steps at most 1. The snap can
# only ever hand back the starting point.
LATTICE = range(-1000, 1001, 2)      # spacing 2
DRIFT_STEPS = 1000
random.seed(5)
real_walk, discrete_walk = [0.0], [0]
for _ in range(DRIFT_STEPS):
    real_walk.append(real_walk[-1] + random.uniform(-1, 1))
    discrete_walk.append(closest(discrete_walk[-1] + random.uniform(-1, 1),
                                 LATTICE))
print(f"\nThe drift demo: {DRIFT_STEPS} uniform(-1, +1) mutations")
print("-" * 78)
print(f"  real gene:                 drifted to {real_walk[-1]:+.2f}")
print(f"  lattice gene (spacing 2):  ended at    {discrete_walk[-1]}")
print("\n  Same operator, same draws. The real gene random-walks; the lattice "
      "gene never")
print("  moves, because every step is smaller than half the lattice spacing, "
      "so the snap")
print("  undoes every one. Nothing raised, nothing warned. When the lattice "
      "is coarser")
print("  than the step, mutation is not weak - it is absent.")
# ------------------------------------------------------------------------------

fig, (ax_snap, ax_drift) = plt.subplots(1, 2, figsize=(11.5, 4.0))
# Replay the census draw pattern (seed 2) to get the before/after deltas.
random.seed(2)
deltas = []
for _ in range(TRIALS):
    n0 = random.choice(N_SET)
    deltas.append(closest(n0 + random.gauss(0.0, 1.0), N_SET) - n0)
ax_snap.hist(deltas, bins=np.arange(-6.5, 6.5, 1.0), color="tab:blue",
             edgecolor="white")
ax_snap.bar([0], [deltas.count(0)], width=1.0, color="tab:red", alpha=0.75,
            label="snapped back to the same n")
ax_snap.set_title("Where 5000 n-mutations actually landed", fontsize=10)
ax_snap.set_xlabel("change in n after the snap")
ax_snap.set_ylabel("count")
ax_snap.legend(fontsize=8)
ax_snap.grid(True, axis="y", linestyle=":", alpha=0.5)

ax_drift.plot(real_walk, linewidth=1.2, color="tab:blue",
              label=f"real gene (ends {real_walk[-1]:+.2f})")
ax_drift.plot(discrete_walk, linewidth=1.2, color="tab:red",
              label=f"lattice gene, spacing 2 (ends {discrete_walk[-1]})")
ax_drift.set_title("Same operator, same draws, 1000 mutations", fontsize=10)
ax_drift.set_xlabel("mutation number")
ax_drift.set_ylabel("gene value")
ax_drift.legend(fontsize=8)
ax_drift.grid(True, linestyle=":", alpha=0.5)

fig.suptitle("Silent repair: the operator is quieter than its parameters say")
FIGURES.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURES / "black_box_04_the_operators.png", dpi=150,
            bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved to {FIGURES}/black_box_04_the_operators.png")

