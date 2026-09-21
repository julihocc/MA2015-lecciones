"""
Lesson 08 - Step 3: One chromosome, three kinds of gene
========================================================
NEW IN THIS STEP: Individual, and the two repair functions that make it legal.

Until now a chromosome was a real number, or a vector of real numbers, and
"legal" meant "inside [-10, 10]". Here one individual carries three different
kinds of gene at once:

    a, b, x   real values inside an interval        -> can be pushed back to the edge
    n         a point on an integer lattice         -> can be snapped to the nearest
    fun_name  a member of a set with no order       -> can only be replaced

Nothing in the course so far applies to the last two. Lesson 04's blend crossover
and Lesson 05's gaussian mutation both produce arbitrary reals; handed to this
problem they produce n = 3.47 and no value of fun_name at all. The design
decision this step makes - and it IS a decision, not a given - is to let the
operators stay sloppy and put ALL the legality in the constructor. An Individual
cannot exist in an illegal state, so no operator ever has to check.

The price of that decision is measured here and paid in step 4: repair is
silent. A gene the operator moved and the constructor moved back looks exactly
like a gene the operator never touched.

GONE FROM THIS STEP: grid_maximum() and the refinement ladder. Step 2 settled
that question; the grid does not come back.

CHANGES FROM black_box_02_the_grid.py
Introduce them in this order:
    1. clamp()                a bounded real gene has edges to be pushed back to
    2. closest()              a lattice gene has no in-between; snap to the nearest
    3. Individual             the constructor repairs, so the operators need not
    4. create_random()        the initial population, and where it actually lives
    5. repair_report()        measure how often repair fires, since it is silent

Run it:  python black_box_03_the_chromosome.py

The constructor repairs all five illegal examples; the initial population
samples only 10% of the declared x range.
"""
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

SEED = 19
POPULATION_SIZE = 200
A_MIN, A_MAX = 0.0, 1.0
B_MIN, B_MAX = 0.0, 1.0
X_MIN, X_MAX = -100.0, 100.0
N_SET = range(0, 21)
FUN_SET = ("sin", "cos")
# The initialiser deliberately does NOT cover the declared x range. This is the
# book's choice and it is kept, because step 6 has to notice it.
INIT_X_MIN, INIT_X_MAX = -10.0, 10.0
FIGURES = Path(__file__).resolve().parent.parent / "figures"


def complicated_one(a: float, b: float, x: float, n: int, fun_name: str) -> float:
    """The black box, unchanged from step 1.

    Args:
        a: real input on [0, 1].
        b: real input on [0, 1].
        x: real input on [-100, 100].
        n: integer input in 0..20.
        fun_name: ``"sin"`` or ``"cos"``.

    Returns:
        One float stored as ``Individual.fitness`` after repair.

    Example:
        Step 1's live slices still peak at ``x = -10.950``; this step does
        not re-plot them.
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


print("=" * 78)
print("Lesson 08 - Step 3: one chromosome, three kinds of gene")
print("=" * 78)

# --- NEW (1) clamp() ----------------------------------------------------------
def clamp(gene: float, low: float, high: float) -> float:
    """Push a real gene back inside its interval.

    Lesson 04 already needed this for one gene on one interval. Nothing new
    except that there are now three of them with three different intervals, so
    the bounds are arguments rather than module constants.

    Args:
        gene: proposed real value, legal or not.
        low, high: inclusive edges of that gene's declared interval.

    Returns:
        ``gene`` if it is already inside; otherwise the nearer edge.

    Example:
        Step 3 hands in ``a = 1.7`` and stores ``1.0``; ``x = 250`` stores
        ``100``. All five illegal examples come back legal.
    """
    return max(low, min(high, gene))
# ------------------------------------------------------------------------------

# --- NEW (2) closest() --------------------------------------------------------
def closest(value: float, allowed: Sequence[int]) -> int:
    """Snap a real number onto the nearest legal point of a lattice.

    An integer gene is not a real gene with a rounding step bolted on; it is a
    different object. There is no n = 3.47 to evaluate, so any operator that
    produces one has produced nothing, and something has to decide what it meant.
    Nearest-neighbour is the cheap, obvious answer - and step 4 shows what it
    costs when the lattice is coarser than the operator's step size.

    Args:
        value: the real an operator proposed.
        allowed: the legal lattice, here ``range(0, 21)``.

    Returns:
        The nearest legal integer. Ties go to the first minimum ``min`` finds.

    Example:
        Step 3 snaps ``n = 3.47`` to ``3`` and ``n = 99`` to ``20``; both
        individuals exist, and neither raises.
    """
    return min(allowed, key=lambda candidate: abs(candidate - value))
# ------------------------------------------------------------------------------

# --- NEW (3) Individual -------------------------------------------------------
class Individual:
    """A five-gene chromosome that cannot exist in an illegal state.

    All the repair lives here, in one place, so that crossover and mutation can
    be written as if every gene were a real number and still never produce an
    invalid individual. The class counts its own instantiations because from
    step 5 onwards the number of Individuals created IS the number of fitness
    evaluations spent, and that census is the only budget line this lesson has.

    Args:
        gene_list: five raw values ``[a, b, x, n, fun_name]``. Reals are
            clamped, ``n`` is snapped, the label is stored as given.

    Example:
        The constructor repairs all five illegal examples; the initial
        population still samples only 10% of the declared ``x`` range.
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
        self.raw_gene_list: List[Any] = list(gene_list)
        self.gene_list: List[Any] = [
            clamp(float(a_raw), A_MIN, A_MAX),
            clamp(float(b_raw), B_MIN, B_MAX),
            clamp(float(x_raw), X_MIN, X_MAX),
            closest(float(n_raw), N_SET),
            fun_name,
        ]
        a, b, x, n, fun_name = self.gene_list
        self.fitness: float = complicated_one(a, b, x, n, fun_name)

    def __str__(self) -> str:
        a, b, x, n, fun_name = self.gene_list
        return f"[a={a:.4f} b={b:.4f} x={x:+.4f} n={n:2d} fun={fun_name}]"
# ------------------------------------------------------------------------------

# --- NEW (4) create_random() --------------------------------------------------
def create_random() -> Individual:
    """One random individual, each gene drawn the way its own type allows.

    Note the asymmetry: a and b are drawn over their whole declared range, but x
    is drawn over [-10, 10] out of a declared [-100, 100]. The lattice gene is
    drawn by choosing a legal point, never by drawing a real and snapping, and
    the label gene is drawn by choosing a member. There is no other way to draw
    them - which is the point.

    Returns:
        A legal ``Individual``. ``x`` is drawn on [-10, 10], not [-100, 100].

    Example:
        The initial population of 200 covers only 10% of the declared ``x``
        range; whatever lies outside can only be reached by mutation.
    """
    return Individual([
        random.uniform(A_MIN, A_MAX),
        random.uniform(B_MIN, B_MAX),
        random.uniform(INIT_X_MIN, INIT_X_MAX),
        random.choice(N_SET),
        random.choice(FUN_SET),
    ])
# ------------------------------------------------------------------------------

# --- NEW (5) repair_report() --------------------------------------------------
def repair_report(candidates: Sequence[Sequence[Any]]) -> List[Tuple[str, Any, Any]]:
    """Build every candidate and record which genes the constructor changed.

    Repair is invisible from outside: the caller hands in a gene list and gets
    back a legal Individual, with no signal that anything was altered. That is
    convenient and it is dangerous, so this function exists to make the silent
    part audible at least once.

    Args:
        candidates: raw five-gene lists, legal or not.

    Returns:
        One ``(name, raw, kept)`` row per gene the constructor changed.

    Example:
        All five illegal examples in this step produce a row; the
        constructor never raises.
    """
    rows: List[Tuple[str, Any, Any]] = []
    for gene_list in candidates:
        individual = Individual(gene_list)
        for name, raw, kept in zip("a b x n fun".split(), gene_list, individual.gene_list):
            if raw != kept:
                rows.append((name, raw, kept))
    return rows
# ------------------------------------------------------------------------------

print("\nWhat the constructor does to genes an operator might hand it")
print("-" * 78)
ILLEGAL = [
    [1.7, 0.5, 3.0, 3, "cos"],        # a above its ceiling
    [0.5, -0.4, 3.0, 3, "cos"],       # b below its floor
    [0.5, 0.5, 250.0, 3, "cos"],      # x far outside the declared range
    [0.5, 0.5, 3.0, 3.47, "cos"],     # n between two lattice points
    [0.5, 0.5, 3.0, 99.0, "cos"],     # n past the end of the lattice
]
for name, raw, kept in repair_report(ILLEGAL):
    print(f"  gene {name:<3} handed in {str(raw):>8}  ->  stored as {str(kept):>8}")
print("\n  Five illegal gene lists went in; five legal Individuals came out, and not")
print("  one of them raised. That is the design decision of this step, stated as")
print("  bluntly as it can be stated.")

random.seed(SEED)
Individual.counter = 0
population = [create_random() for _ in range(POPULATION_SIZE)]

print(f"\nThe initial population ({POPULATION_SIZE} individuals, SEED = {SEED})")
print("-" * 78)
xs = [ind.gene_list[2] for ind in population]
ns = [ind.gene_list[3] for ind in population]
funs = Counter(ind.gene_list[4] for ind in population)
fits = [ind.fitness for ind in population]
declared_width = X_MAX - X_MIN
covered = max(xs) - min(xs)
print(f"  x  spans [{min(xs):+.3f}, {max(xs):+.3f}] - {covered / declared_width * 100:.1f}% of the "
      f"declared [{X_MIN:+.0f}, {X_MAX:+.0f}]")
print(f"  n  takes {len(set(ns))} of the {len(N_SET)} legal values; "
      f"most common is n={Counter(ns).most_common(1)[0][0]} "
      f"({Counter(ns).most_common(1)[0][1]} individuals)")
print(f"  fun_name: " + ", ".join(f"{k}={v}" for k, v in sorted(funs.items())))
dead = [ind for ind in population if ind.fitness == 0.0]
on_dead_sheet = [ind for ind in population if ind.gene_list[3] == 0]
print(f"\n  {len(dead)} of {POPULATION_SIZE} individuals have fitness exactly 0.0, and "
      f"{len(on_dead_sheet)} of them")
print(f"  sit on the n = 0 sheet that step 1 found to be identically zero.")
print(f"  Best fitness in the initial population: {max(fits):.6e}")
print(f"  Fitness evaluations spent so far: {Individual.counter}")

print("\nThe part that is not free")
print("-" * 78)
print(f"  The initialiser samples x over [{INIT_X_MIN:+.0f}, {INIT_X_MAX:+.0f}] while the box accepts")
print(f"  [{X_MIN:+.0f}, {X_MAX:+.0f}]. That is {(INIT_X_MAX - INIT_X_MIN) / declared_width * 100:.0f}% of the declared axis, and it is the book's")
print("  choice, kept deliberately. Whatever lies outside can only be reached by")
print("  mutation walking there one step at a time. Step 6 checks whether it ever did.")

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))
axes[0].hist(xs, bins=25, color="tab:blue", edgecolor="white")
axes[0].axvspan(X_MIN, INIT_X_MIN, color="tab:red", alpha=0.12)
axes[0].axvspan(INIT_X_MAX, X_MAX, color="tab:red", alpha=0.12, label="declared but never sampled")
axes[0].set_xlim(X_MIN, X_MAX)
axes[0].set_title("x: real, and only 10% of it is used")
axes[0].set_xlabel("x")
axes[0].legend(fontsize=8)
axes[1].hist(ns, bins=np.arange(min(N_SET) - 0.5, max(N_SET) + 1.5, 1.0),
             color="tab:green", edgecolor="white")
axes[1].set_title("n: a lattice, 21 legal points")
axes[1].set_xlabel("n")
axes[2].bar(list(sorted(funs)), [funs[k] for k in sorted(funs)], color="tab:purple")
axes[2].set_title("fun_name: a set, no order, no midpoint")
axes[2].set_xlabel("fun_name")
for ax in axes:
    ax.set_ylabel("individuals")
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)
fig.suptitle(f"One chromosome, three kinds of gene (initial population, SEED = {SEED})")
FIGURES.mkdir(exist_ok=True)
fig.savefig(FIGURES / "black_box_03_the_chromosome.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved to {FIGURES}/black_box_03_the_chromosome.png")

