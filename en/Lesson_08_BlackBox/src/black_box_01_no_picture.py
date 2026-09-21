"""
Lesson 08 - Step 1: The function you cannot look at
====================================================
NEW IN THIS STEP: complicated_one(), slice_along_x(), and the discovery that
"the landscape" is not a thing you can draw.

Every lesson so far has optimised f(x) = sin(x) - 0.2*|x| or a close relative:
one real gene, one curve, one picture on the slide, and - crucially - a
brute-force optimum on a grid to check the run against. Lesson 02 step 5 said so
in as many words and warned that the honest answer to "did this run succeed?"
gets hard once x stops being one number.

This is where it stops being one number. The function below arrives from
somewhere else. Nobody derived it, nobody will differentiate it, and its five
arguments are of three different kinds:

    a, b      real, bounded to [0, 1]
    x         real, bounded to [-100, 100]
    n         an INTEGER, 0 to 20 - there is no f(a, b, x, 3.5, ...)
    fun_name  a LABEL, 'sin' or 'cos' - there is no midpoint between them

So the domain is not a region of the plane. It is 21 x 2 = 42 separate
three-dimensional sheets, and a plot is a line drawn through one of them.

This step draws six such lines. The plan was to show that they disagree about
everything; they do not, and the truth turned out to be more useful. All five
live slices put their best x at the SAME place, to the resolution they were
drawn at; their best values differ by nearly a factor of ten; and one whole
sheet is identically zero. A picture that agrees with itself about the location
and not about the value has not resolved what is happening there - and nothing
inside a picture can tell you that. From here on the plot is decoration, and
every later step has to replace what the plot used to do for us.

Run it:  python black_box_01_no_picture.py

Five live slices agree on x = -10.950 while their maxima differ by 9.23×;
one whole sheet is identically zero. The picture has not resolved the value.
"""
import math
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# The declared domain. Read these as a contract from whoever handed us the box:
# they say which inputs are legal, and they say nothing at all about the output.
A_MIN, A_MAX = 0.0, 1.0
B_MIN, B_MAX = 0.0, 1.0
X_MIN, X_MAX = -100.0, 100.0
N_SET = range(0, 21)             # integer gene: a lattice, not an interval
FUN_SET = ("sin", "cos")         # categorical gene: a set, not a range
FIGURES = Path(__file__).resolve().parent.parent / "figures"


def complicated_one(a: float, b: float, x: float, n: int, fun_name: str) -> float:
    """The black box.

    Reproduced from Gridin's Chapter 8 exactly as the book gives it. Do not read
    it for meaning - the point of a black-box problem is that you are not
    supposed to. It is here so that the lesson runs; the only fact about it we
    are allowed to use is that it takes five arguments and returns a float.

    (Step 7 breaks that rule on purpose, once every legitimate method has been
    tried and has failed to answer the question.)

    Args:
        a: real input on [0, 1].
        b: real input on [0, 1].
        x: real input on [-100, 100].
        n: integer input in 0..20; the inner loop runs n+1 times.
        fun_name: ``"sin"`` or ``"cos"``. Any other label raises.

    Returns:
        One float. The signature does not bound its magnitude.

    Example:
        Step 1's five live slices all peak at ``x = -10.950`` while their
        maxima still differ by ``9.23×``.
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


def slice_along_x(a: float, b: float, n: int, fun_name: str,
                  points: int = 4001) -> Tuple[np.ndarray, np.ndarray]:
    """One line through the box: vary x, hold the other four genes still.

    This is the only kind of picture a five-argument function admits, and it is
    why the pictures in Lessons 01-06 were doing more work than they looked
    like. There, the slice WAS the landscape. Here it is one line out of an
    uncountable family.

    Args:
        a, b: the two real genes held fixed for this slice.
        n: the integer sheet to walk.
        fun_name: ``"sin"`` or ``"cos"``.
        points: samples along x, inclusive of both ends. 4001 gives spacing
            0.05 on [-100, 100]; nothing in a plot says whether that is enough.

    Returns:
        ``(xs, ys)``: x-coordinates and the corresponding box values.

    Example:
        The six slices of this step put five live peaks at ``x = -10.950``
        and leave one sheet identically zero.
    """
    xs = np.linspace(X_MIN, X_MAX, points)
    ys = np.array([complicated_one(a, b, float(x), n, fun_name) for x in xs])
    return xs, ys


print("=" * 78)
print("Lesson 08 - Step 1: the function you cannot look at")
print("=" * 78)

print("\nThe declared domain")
print("-" * 78)
print(f"  a         real, [{A_MIN}, {A_MAX}]")
print(f"  b         real, [{B_MIN}, {B_MAX}]")
print(f"  x         real, [{X_MIN}, {X_MAX}]")
print(f"  n         integer, {min(N_SET)}..{max(N_SET)}   ({len(N_SET)} legal values)")
print(f"  fun_name  label, one of {list(FUN_SET)}   ({len(FUN_SET)} legal values)")
sheets = len(N_SET) * len(FUN_SET)
print(f"\n  Continuous axes: 3.  Discrete combinations: {len(N_SET)} x {len(FUN_SET)} = {sheets}.")
print(f"  So the domain is {sheets} separate 3-D volumes, not one surface. There is no")
print("  single object to plot, and no continuous path from n=3 to n=4.")

# Four slices along x, all at the same (a, b), differing only in the two
# discrete genes. If the discrete genes were a detail, these would look alike.
CASES: List[Tuple[float, float, int, str]] = [
    (0.5, 0.5, 0, "cos"),     # the discrete genes move, a and b stay put ...
    (0.5, 0.5, 3, "cos"),
    (0.5, 0.5, 3, "sin"),
    (0.5, 0.5, 11, "cos"),
    (0.0, 0.0, 3, "cos"),     # ... and now a and b move, the discrete genes stay
    (0.0, 0.0, 3, "sin"),
]

print("\nSix slices along x through the same box")
print("-" * 78)
print(f"  {'a':>4} {'b':>4} {'n':>3} {'fun':>4} | {'max over slice':>15} {'at x':>10} "
      f"{'min over slice':>15} {'sign changes':>13}")
slices = []
for a, b, n, fun_name in CASES:
    xs, ys = slice_along_x(a, b, n, fun_name)
    slices.append((a, b, n, fun_name, xs, ys))
    top = int(np.argmax(ys))
    crossings = int(np.count_nonzero(np.sign(ys[:-1]) != np.sign(ys[1:])))
    print(f"  {a:4.1f} {b:4.1f} {n:3d} {fun_name:>4} | {ys[top]:15.6e} {xs[top]:10.3f} "
          f"{ys.min():15.6e} {crossings:13d}")

# Every claim below is computed from the four rows just printed.
live = [(a, b, n, fn, xs, ys) for a, b, n, fn, xs, ys in slices if ys.max() > 0.0]
argmaxes = [float(xs[int(np.argmax(ys))]) for _, _, _, _, xs, ys in live]
maxima = [float(ys.max()) for _, _, _, _, _, ys in live]
dead = len(slices) - len(live)
argmax_span = max(argmaxes) - min(argmaxes)
value_ratio = max(maxima) / min(maxima)

print(f"\n  {dead} of the {len(slices)} slices never rise above zero at all.")
print(f"  The other {len(live)} put their best x within {argmax_span:.3f} of each other")
print(f"  (all of them at x = {min(argmaxes):.3f}), so the slices agree about WHERE.")
print(f"  Their best VALUES differ by a factor of {value_ratio:.3g}.")
print("\n  That combination is the trap. Agreement on the location reads as evidence")
print(f"  that the picture is telling the truth, while the factor of {value_ratio:.3g} in the")
print("  value says it has not resolved whatever is actually happening there.")
spacing = (X_MAX - X_MIN) / (len(slices[0][4]) - 1)
print(f"  These curves were sampled at {len(slices[0][4])} points, spacing {spacing:.4f}. Nothing")
print("  inside a plot tells you whether that was fine enough. Step 2 asks the grid")
print("  that question directly, and gets an answer nobody wants.")

# One of the 21 integer values is special, and it is worth naming: with n = 0 the
# numerator (x*n + log(n+1)) is 0 + log(1) = 0 for every x, a, b.
zero_sheet = [n for n in N_SET
              if max(abs(complicated_one(0.3, 0.7, x, n, "cos")) for x in (-7.0, 0.0, 7.0, 40.0)) == 0.0]
print(f"\n  Integer values of n for which the box returns exactly 0.0 everywhere: {zero_sheet}")
print(f"  That is {len(zero_sheet)} of the {len(N_SET)} sheets, dead by construction. A search that")
print("  spends its population there is not searching; it is idling. Nothing in the")
print("  function signature warned us, and no plot along x would have shown it.")

print("\nWhat is gone, and what has to replace it")
print("-" * 78)
print("  Gone:  the picture. There is no curve to point at on a slide, and no way")
print("         to see the optimum before running anything.")
print("  Gone:  the brute-force check. Step 2 prices it and finds it unaffordable.")
print("  Left:  the diagnostics this course has already built - best-vs-average,")
print("         gene spread, the census of evaluations. From here they are not")
print("         extra credit; they are the only instruments there are.")

fig, axes = plt.subplots(2, 3, figsize=(15, 7))
for ax, (a, b, n, fun_name, xs, ys) in zip(axes.ravel(), slices):
    ax.plot(xs, ys, color="tab:blue", linewidth=0.9)
    ax.set_title(f"a={a}, b={b}, n={n}, fun='{fun_name}'", fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("f")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
fig.suptitle("Six slices through the same black box: same shape, wildly different scale")
FIGURES.mkdir(exist_ok=True)
fig.savefig(FIGURES / "black_box_01_no_picture.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved to {FIGURES}/black_box_01_no_picture.png")

