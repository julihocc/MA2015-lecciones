"""
Lesson 08 - Step 7: Opening the box
===================================
NEW IN THIS STEP: denominator(), the pole, and the ladder.

Six steps have treated complicated_one() as a black box, the way the book
presents it. The searches are done; now we read the one line that was doing
all the damage. Every term of the sum is a numerator over a denominator, and
the denominator is

    div = ((n + 1) ** 2) * (1 + a + b) * (120 - x ** 2) * resid + 1 / 2

Everything in it is tame except the product (120 - x ** 2) * resid, which
can be made as close to -1/(2 * (n+1)^2 * (1+a+b)) as x allows - and there
the denominator crosses zero. The box does not contain a maximum. It
contains a pole.

For step 5's champion genes the crossing is found by bisection at
x* = -10.9535836, a hair away from -sqrt(120) = -10.9544512 (the +1/2
shifts it). Then the ladder: evaluate f at x* + 10^-k for growing k. The
reported "fitness" multiplies by 100 every time the distance shrinks by 100
- 2.2e-14, 2.2e-12, 2.2e-10, 2.2e-08 - with no ceiling in sight. And the
champion the GA was so proud of? It is parked 9.6e-08 from the pole, and its
2.278e-09 is exactly the rung of the ladder at that distance. The algorithm
did not find a maximum. It measured its own parking precision against a
singularity.

The moral is the chapter's: with a black box, the algorithm's report and the
truth are different documents. Judging the answer takes knowledge the run
cannot supply - so before trusting a search, open the box.

GONE FROM THIS STEP: the entire genetic algorithm. The search ended in
step 6; the champion is quoted below as a literal, with its provenance.

CHANGES FROM black_box_06_the_verdict.py
Introduce them in this order:
    1. denominator()   the dangerous subexpression, isolated and readable
    2. the pole        the denominator changes sign: bisection finds where
    3. the ladder      f at x* + 10^-k: what the GA was actually climbing

Run it:  python black_box_07_opening_the_box.py

The apparent optimum is a pole near x = -10.9535836; the reported fitness
measures distance to the singularity.
"""
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

FIGURES = Path(__file__).resolve().parent.parent / "figures"

# Step 5's champion, verbatim: SEED = 19, 100 generations, 20,204
# evaluations. Twelve independent runs in step 6 parked within 0.008 of
# this same x.
CHAMPION = [0.3259749523591883, 0.47589136714114383, -10.953583467984679,
            3, "cos"]
CHAMPION_FITNESS = 2.278347670913726e-09


def complicated_one(a: float, b: float, x: float, n: int, fun_name: str) -> float:
    """The box, open at last: same code as always, now we are reading it.

    Args:
        a: real input, held at the champion's 0.326.
        b: real input, held at the champion's 0.476.
        x: the axis we now walk; the pole sits near ``-10.9535836``.
        n: integer input, held at 3.
        fun_name: held at ``"cos"``.

    Returns:
        One float. Near the pole it tracks distance, not a maximum.

    Example:
        The champion's ``2.278e-09`` is the ladder rung at ``9.6e-08`` from
        the pole ``x = -10.9535836``.
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


# --- NEW (1) denominator() ----------------------------------------------------
# The one subexpression that mattered, isolated. The numerator is boring -
# at the champion's genes it is about -126 and varies slowly. Everything the
# searches "discovered" lives in this denominator.
def denominator(x: float, a: float, b: float, n: int, fun_name: str) -> float:
    """The divisor of every term, for fixed (a, b, n, fun_name).

    Args:
        x: the only gene this step varies.
        a, b, n, fun_name: held at step 5's champion.

    Returns:
        The ``div`` subexpression. A sign change is a pole, not a peak.

    Example:
        Bisection finds the zero at ``x* = -10.9535836``, a hair off
        ``-sqrt(120)`` because of the ``+ 1/2``.
    """
    if fun_name == "cos":
        trig = math.cos(math.log2(b ** n + 1) * math.pi * x) / (n + 1) + math.sin(x) ** n
    else:
        trig = math.sin(math.log2(b ** n + 1) * math.pi * x) / (n + 1) + math.cos(x) ** n
    resid = trig - math.log2(n + 1)
    return ((n + 1) ** 2) * (1 + a + b) * (120 - x ** 2) * resid + 1 / 2
# ------------------------------------------------------------------------------

A, B, X_CHAMP, N, FUN = CHAMPION

print("=" * 78)
print("Lesson 08 - Step 7: opening the box")
print("=" * 78)
print(f"\nThe genes stay fixed at step 5's champion: a={A:.4f}, b={B:.4f}, "
      f"n={N}, fun={FUN}")
print("Only x moves now.")

# --- NEW (2) the pole ---------------------------------------------------------
# div(-11.0) = +26.59 and div(-10.9) = -31.38: the denominator changes sign
# between them. A continuous function that changes sign has a zero in
# between, and bisection finds it to machine precision in 60 halvings.
low, high = -11.0, -10.9
print("\nThe pole")
print("-" * 78)
print(f"  denominator(-11.0) = {denominator(low, A, B, N, FUN):+.4f}")
print(f"  denominator(-10.9) = {denominator(high, A, B, N, FUN):+.4f}")
for _ in range(60):
    mid = (low + high) / 2
    if denominator(low, A, B, N, FUN) * denominator(mid, A, B, N, FUN) <= 0:
        high = mid
    else:
        low = mid
X_STAR = (low + high) / 2
print(f"  sign change bracketed: pole at x* = {X_STAR:.10f}")
print(f"  compare -sqrt(120) = {-math.sqrt(120):.10f}  "
      f"(the +1/2 shifts the zero off the obvious spot)")
print(f"  the champion sits at x* {X_CHAMP - X_STAR:+.2e}  "
      f"- parked {abs(X_CHAMP - X_STAR):.1e} from the pole")
# ------------------------------------------------------------------------------

# --- NEW (3) the ladder -------------------------------------------------------
# Walk toward the pole in powers of ten and read f at each rung. If f were
# bounded near x*, the rungs would settle. They do not: every 100x closer
# multiplies the reading by 100. That is the signature of 1/(x - x*), and
# it is exactly the law the grid of step 2 met (2x, 21x, 1387x) and the GA
# of steps 5-6 climbed (12 champions, 21,504x apart, all parked here).
print("\nThe ladder: f at x* + 10^-k")
print("-" * 78)
print(f"  {'k':>3}  {'distance':>10}  {'f(x* + 10^-k)':>14}")
for k in (1, 2, 4, 6, 8):
    distance = 10.0 ** (-k)
    value = complicated_one(A, B, X_STAR + distance, N, FUN)
    print(f"  {k:>3}  {distance:>10.0e}  {value:>14.3e}")
print(f"\n  the champion's own rung: distance {abs(X_CHAMP - X_STAR):.1e}, "
      f"f = {CHAMPION_FITNESS:.3e}  (it sits exactly on this ladder)")
print("\n  There is no maximum. The ladder has no top rung; the grid of step 2")
print("  could not see one because there is none to see, and the GA of steps")
print("  5-6 'improved' by parking closer to a singularity. The algorithm's")
print("  report and the truth were different documents - and only opening")
print("  the box reconciles them.")
# ------------------------------------------------------------------------------

fig, (ax_div, ax_ladder) = plt.subplots(1, 2, figsize=(11.5, 4.0))
xs = np.linspace(-11.05, -10.85, 500)
ax_div.plot(xs, [denominator(v, A, B, N, FUN) for v in xs],
            color="tab:blue", linewidth=1.4)
ax_div.axhline(0.0, color="tab:gray", linewidth=0.8)
ax_div.axvline(X_STAR, color="tab:red", linewidth=1.0, linestyle="--",
               label=f"pole at x* = {X_STAR:.6f}")
ax_div.set_title("The denominator crosses zero", fontsize=10)
ax_div.set_xlabel("x")
ax_div.set_ylabel("denominator(x)")
ax_div.legend(fontsize=8)
ax_div.grid(True, linestyle=":", alpha=0.5)

distances = np.logspace(-9, -1, 200)
ax_ladder.loglog(distances,
                 [abs(complicated_one(A, B, X_STAR + d, N, FUN))
                  for d in distances],
                 color="tab:blue", linewidth=1.4,
                 label="|f(x* + d)|: the ladder")
ax_ladder.scatter([abs(X_CHAMP - X_STAR)], [CHAMPION_FITNESS], s=70,
                  color="tab:red", zorder=3,
                  label="step 5's champion, parked on the ladder")
ax_ladder.set_title("What the search was climbing: no top rung", fontsize=10)
ax_ladder.set_xlabel("distance from the pole")
ax_ladder.set_ylabel("|f|  (log scale)")
ax_ladder.legend(fontsize=8)
ax_ladder.grid(True, which="both", linestyle=":", alpha=0.5)

fig.suptitle("Opening the box: a pole, not a maximum")
FIGURES.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURES / "black_box_07_opening_the_box.png", dpi=150,
            bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved to {FIGURES}/black_box_07_opening_the_box.png")

