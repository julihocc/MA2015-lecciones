"""
Lesson 08 - Step 2: The grid, priced and then refined
======================================================
NEW IN THIS STEP: the brute-force check from Lessons 01-06, costed honestly on
this problem and then interrogated.

Every earlier lesson checked its runs against a brute-force optimum on a grid.
That check is what made "did this run succeed?" answerable. This step tries to
buy it here, and fails twice.

The first failure is arithmetic: the book's own grid over these five genes is
105 million evaluations, and we measure - not guess - what that costs.

The second failure is worse, and it is the one to teach. Even a grid we CAN
afford gives an answer that will not sit still. Refine the x axis by ten and the
reported maximum does not converge to a limit; it multiplies. A sequence that
grows by 2x, then 21x, then 1387x is not an estimate approaching a value. It is
the grid telling us, in the only language a grid has, that whatever it is
sampling is not a maximum. We are not allowed to open the box yet, so we take
that as data and carry it forward.

CHANGES FROM black_box_01_no_picture.py
Introduce them in this order:
    1. evaluation_cost()      measure the price of one call before quoting any total
    2. BOOK_GRID              the book's brute force, priced rather than run
    3. grid_maximum()         the exhaustive search we can actually afford
    4. REFINEMENTS            run it at four resolutions and read the sequence

Run it:  python black_box_02_the_grid.py

The book grid costs 105,525,000 calls. Refining the affordable x-axis makes
the reported maximum grow 61,099-fold instead of converge.
"""
import math
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

A_MIN, A_MAX = 0.0, 1.0
B_MIN, B_MAX = 0.0, 1.0
X_MIN, X_MAX = -100.0, 100.0
N_SET = range(0, 21)
FUN_SET = ("sin", "cos")
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
        One float. A grid that reports a growing maximum is sampling this.

    Example:
        The book grid over these five genes is 105,525,000 calls; refinement
        then grows the reported maximum 61,099-fold instead of converging.
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
print("Lesson 08 - Step 2: the grid, priced and then refined")
print("=" * 78)

# --- NEW (1) evaluation_cost() ------------------------------------------------
def evaluation_cost(samples: int = 20000, seed: int = 7) -> float:
    """Seconds per call to the box, measured on this machine, right now.

    Quoting "105 million evaluations" means nothing without a price per
    evaluation, and the price is not a constant of the problem: the box's inner
    loop runs n+1 times, so a call with n = 20 costs twenty times a call with
    n = 0. We therefore time a sample drawn the way the grid will draw it -
    n uniform over its whole set - and report the mean.

    Args:
        samples: timed calls. 20,000 is enough for a stable mean.
        seed: draws n uniformly over 0..20, matching the grid's mix.

    Returns:
        Mean seconds per call on this machine. Timing is not a constant;
        quote the number this run prints.

    Example:
        Multiplied by the book grid of 105,525,000 cells this is the class
        budget the step refuses to spend.
    """
    rng = np.random.default_rng(seed)
    a_s = rng.uniform(A_MIN, A_MAX, samples)
    b_s = rng.uniform(B_MIN, B_MAX, samples)
    x_s = rng.uniform(X_MIN, X_MAX, samples)
    n_s = rng.integers(min(N_SET), max(N_SET) + 1, samples)
    f_s = rng.integers(0, len(FUN_SET), samples)
    start = time.perf_counter()
    for a, b, x, n, fi in zip(a_s, b_s, x_s, n_s, f_s):
        complicated_one(float(a), float(b), float(x), int(n), FUN_SET[int(fi)])
    return (time.perf_counter() - start) / samples
# ------------------------------------------------------------------------------

# --- NEW (2) BOOK_GRID --------------------------------------------------------
# Gridin's Chapter 8 brute_force.py, transcribed as sizes rather than as loops:
#   a in arange(0, 1, .02)      b in arange(0, 1, .02)
#   x in arange(-100, 101, .2)  n in range(0, 21)      fun in {cos, sin}
# The book runs it. We price it, because pricing it is the lesson.
BOOK_GRID: List[Tuple[str, int, str]] = [
    ("a", len(np.arange(0.0, 1.0, 0.02)), "step 0.02"),
    ("b", len(np.arange(0.0, 1.0, 0.02)), "step 0.02"),
    ("x", len(np.arange(-100.0, 101.0, 0.2)), "step 0.2"),
    ("n", len(N_SET), "every legal value"),
    ("fun_name", len(FUN_SET), "every legal value"),
]
# ------------------------------------------------------------------------------

seconds_per_eval = evaluation_cost()
print(f"\nMeasured cost of one call to the box: {seconds_per_eval * 1e6:.2f} microseconds")
print(f"  (mean over 20000 calls with n drawn uniformly from {min(N_SET)}..{max(N_SET)})")

print("\nThe book's brute-force grid, priced")
print("-" * 78)
book_total = 1
for name, count, note in BOOK_GRID:
    book_total *= count
    print(f"  {name:<9} {count:>6d} points   ({note})")
book_seconds = book_total * seconds_per_eval
print(f"  {'':<9} {'':>6}   product = {book_total:,} evaluations")
print(f"  At the measured price that is {book_seconds:,.0f} s = {book_seconds / 60:.1f} minutes,")
print("  for one problem, in one class, on five genes. Lesson 02's grid was 20,001")
print(f"  points and finished instantly; this one is {book_total / 20001:,.0f} times bigger, because")
print("  a grid costs the PRODUCT of its axes and this problem has five of them.")

# --- NEW (3) grid_maximum() ---------------------------------------------------
def grid_maximum(a_values: Sequence[float], b_values: Sequence[float],
                 x_values: Sequence[float], n_values: Sequence[int],
                 fun_values: Sequence[str]) -> Tuple[float, Tuple, int]:
    """Exhaustive search over an explicit grid. Returns (best, where, evaluations).

    Deliberately written the naive way. There is no cleverness to add: the whole
    claim of brute force is that it looks everywhere, and the whole cost of it is
    that "everywhere" is a product.

    Args:
        a_values, b_values, x_values: the three continuous axes to walk.
        n_values: integer sheets included in the product.
        fun_values: categorical labels included in the product.

    Returns:
        ``(best, where, evaluations)``. ``where`` is the five-tuple that
        produced ``best``; ``evaluations`` is the product of the axis lengths.

    Example:
        Four refinements of x make the reported maximum grow 61,099-fold
        instead of settling, which is not an estimate of a finite peak.
    """
    best = -math.inf
    where: Tuple = ()
    evaluations = 0
    for a in a_values:
        for b in b_values:
            for n in n_values:
                for fun_name in fun_values:
                    for x in x_values:
                        evaluations += 1
                        value = complicated_one(float(a), float(b), float(x), int(n), fun_name)
                        if value > best:
                            best, where = value, (float(a), float(b), float(x), int(n), fun_name)
    return best, where, evaluations
# ------------------------------------------------------------------------------

# --- NEW (4) REFINEMENTS ------------------------------------------------------
# A grid we can afford: the two continuous genes a and b get two values each, the
# discrete genes are pinned, and only x is refined. This is generous to the grid -
# we are letting it spend its whole budget on the single axis most likely to
# matter, which is the best case for the method, not the worst.
GRID_A = (0.0, 0.5)
GRID_B = (0.0, 0.5)
GRID_N = (3,)
REFINEMENTS = (201, 2001, 20001, 200001)
# ------------------------------------------------------------------------------

print("\nRefining one axis: a in {0.0, 0.5}, b in {0.0, 0.5}, n = 3, both labels")
print("-" * 78)
print(f"  {'x points':>9} {'spacing':>9} {'evaluations':>12} {'reported maximum':>18} "
      f"{'at x':>10} {'vs previous':>12}")
ladder: List[Tuple[float, float, float]] = []
previous = None
for points in REFINEMENTS:
    xs = np.linspace(X_MIN, X_MAX, points)
    best, where, evaluations = grid_maximum(GRID_A, GRID_B, xs, GRID_N, FUN_SET)
    spacing = (X_MAX - X_MIN) / (points - 1)
    ratio = best / previous if previous else float("nan")
    ladder.append((spacing, best, evaluations))
    shown = f"{ratio:11.1f}x" if previous else f"{'-':>12}"
    print(f"  {points:9d} {spacing:9.4f} {evaluations:12d} {best:18.6e} "
          f"{where[2]:10.3f} {shown}")
    previous = best

# Every sentence below is computed from the ladder just printed.
growth = [ladder[i][1] / ladder[i - 1][1] for i in range(1, len(ladder))]
total_growth = ladder[-1][1] / ladder[0][1]
print(f"\n  Each tenfold refinement multiplied the reported maximum by "
      f"{', '.join(f'{g:.1f}' for g in growth)}.")
print(f"  Over the whole ladder the answer grew {total_growth:,.0f}-fold and the growth")
print(f"  {'accelerated' if growth[-1] > growth[0] else 'slowed'} rather than settling.")
print("\n  A converging estimate has ratios approaching 1. These are going the other")
print("  way. Whatever the grid is climbing, it has no top that the grid can reach,")
print("  and the number it prints is a statement about the spacing, not about the")
print("  function. Any earlier lesson would have quoted the finest row as 'the")
print("  optimum'. Here that would be an invented number.")

print("\nWhat this costs us")
print("-" * 78)
print("  Lessons 01-06 answered 'did the run succeed?' by comparing the run against")
print("  the grid. Both halves of that are now gone: the honest grid is unaffordable,")
print("  and the affordable grid has no answer to give. Steps 3 to 5 build the search")
print("  anyway, and step 6 has to find some other way to judge it.")

fig, ax = plt.subplots(figsize=(7.5, 5))
spacings = [s for s, _, _ in ladder]
values = [v for _, v, _ in ladder]
ax.loglog(spacings, values, "o-", color="tab:red", linewidth=1.6, markersize=7)
for spacing, value, evaluations in ladder:
    ax.annotate(f"{evaluations:,} evals", (spacing, value),
                textcoords="offset points", xytext=(8, -12), fontsize=8)
ax.invert_xaxis()
ax.set_xlabel("x-grid spacing (finer to the right)")
ax.set_ylabel("maximum the grid reports")
ax.set_title("A brute-force answer that refuses to converge")
ax.grid(True, which="both", linestyle=":", alpha=0.5)
FIGURES.mkdir(exist_ok=True)
fig.savefig(FIGURES / "black_box_02_the_grid.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved to {FIGURES}/black_box_02_the_grid.png")

