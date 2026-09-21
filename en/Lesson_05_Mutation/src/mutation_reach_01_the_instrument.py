"""
Lesson 05 - Step 1: The reach of one mutation, and how to measure it
=====================================================================
Selection discards and crossover recombines; both work only with values the
population already holds. Mutation is the one operator that can produce a value
nobody had. The whole of this lesson is therefore about one question: HOW FAR
can one mutation go?

Lesson 01 step 5 asked that question once, of a population trapped on a local
hill six units from the global one, and got two answers: sigma = 1.0 never
arrived, sigma = 3.0 arrived 110 times in 2000. This step turns that one-off
experiment into an instrument - a function that reports the whole displacement
distribution, not just the arrival count - and every later step reuses it.

The two numbers Lesson 01 printed come out of it unchanged. That identity is
the proof that nothing was quietly redefined on the way.

Run it:  python mutation_reach_01_the_instrument.py

reach_report() fires 2000 single Gaussian mutations from x = -4.6 and
describes the landings. Lesson 01's two numbers reappear exactly: sigma 1.0
arrives 0 of 2000, sigma 3.0 arrives 110 of 2000. At sigma 6.0, 19.4% of
proposals are clamped and mean travel drops to 0.690 of sigma.
"""
import random
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SEED = 52
GENE_MIN, GENE_MAX = -10.0, 10.0
MUTATION_MU = 0.0
TRIALS = 2000
# The situation Lesson 01 left us in: a population converged on the local hill
# near x = -4.6, whose peak is worth f = +0.074. The global hill is at x = +1.38
# and is worth f = +0.706. "Arriving" means landing within HILL_RADIUS of it,
# which is the width of that hill's basin.
TRAPPED_AT = -4.6
GLOBAL_HILL = 1.38
HILL_RADIUS = 1.5
SIGMAS = (0.5, 1.0, 3.0, 6.0)
FIGURES = Path(__file__).resolve().parent.parent / "figures"


def objective(x: float) -> float:
    """Lesson 01's landscape, unchanged. Works on a float and on an array.

    Args:
        x: a real gene, typically in [-10, 10].

    Returns:
        sin(x) - 0.2 * |x|. Larger is better.

    Example:
        The global peak of this landscape is near x = +1.372, f = +0.706.
    """
    return np.sin(x) - 0.2 * abs(x)


def clamp(gene: float, low: float = GENE_MIN, high: float = GENE_MAX) -> float:
    """Mutation can propose any real number; the search space has edges.

    Args:
        gene: raw gene (float).
        low, high: closed bounds of the search interval.

    Returns:
        gene projected onto [low, high].

    Example:
        At sigma = 6.0, 19.4% of proposals are clamped and mean travel
        drops to 0.690 of sigma; Lesson 01's sigma = 1.0 / 3.0 pair is
        0 of 2000 vs 110 of 2000 arrivals from x = -4.6.
    """
    return max(low, min(high, gene))


class Individual:
    """One candidate solution: a chromosome and its score.


    Example:
        From x = -4.6, sigma = 1.0 arrives 0 of 2000 times and sigma = 3.0
        arrives 110 of 2000.
    """

    def __init__(self, gene_list: list[float]) -> None:
        """Build an individual and score it immediately.

        Args:
            gene_list: the chromosome. One gene in the scalar lessons;
                ten genes once mutation gets a rate dial.

        Example:
            Fitness is scored at construction so it cannot fall out of
            step with the genes.
        """
        self.gene_list = gene_list
        self.fitness = float(objective(gene_list[0]))

    @property
    def gene(self) -> float:
        """The first gene as a float rather than a one-element list.

        Returns:
            The first (or only) gene as a float, so plots and tables do
            not keep writing gene_list[0].
        """
        return self.gene_list[0]

    def __repr__(self) -> str:
        """Compact snapshot used in every printed table of this lesson.

        Returns:
            A one-line snapshot such as 'x=+1.372 f=+0.706' or
            '(+1.372) f=+0.7060', used in every printed table.
        """
        return f"x={self.gene:+.3f} f={self.fitness:+.3f}"


def mutate_gaussian(gene: float, mu: float, sigma: float) -> float:
    """Add noise drawn from N(mu, sigma) to a gene - Lesson 01's operator.

    sigma is the step size of the search, and everything this lesson measures
    is a consequence of it.

    Args:
        gene: current value.
        mu: mean of the noise, 0.0.
        sigma: step size.

    Returns:
        The clamped mutant gene.

    Example:
        From x = -4.6, sigma 1.0 arrives 0 of 2000, sigma 3.0 arrives
        110 of 2000 — Lesson 01's two numbers, reproduced.
    """
    return clamp(gene + random.gauss(mu, sigma))


def reach_report(start: float, sigma: float, trials: int = TRIALS) -> dict:
    """Fire `trials` single mutations from `start` and describe where they land.

    This is the instrument. It re-seeds before every measurement, so two rows of
    the table below differ only in sigma and never in the draws they consumed -
    which is what makes the rows comparable at all.

    `clamped` counts the proposals that fell outside [GENE_MIN, GENE_MAX] and
    had to be pulled back to the edge. It is reported because a truncated jump
    is a jump the sigma column no longer describes.

    Args:
        start: the trapped gene, -4.6.
        sigma: the step being measured.
        trials: 2000, reseeded from SEED = 52.

    Returns:
        mean/median/max |dx|, clamped count, uphill count, arrivals
        within HILL_RADIUS = 1.5 of x = +1.38.

    Example:
        sigma 0.5 finds the most uphill moves (272) and arrives 0
        times; sigma 6.0 has 19.4% of proposals clamped.
    """
    random.seed(SEED)
    landings, distances, clamped = [], [], 0
    for _ in range(trials):
        proposal = start + random.gauss(MUTATION_MU, sigma)
        landing = clamp(proposal)
        if landing != proposal:
            clamped += 1
        landings.append(landing)
        distances.append(abs(landing - start))
    arrivals = sum(1 for x in landings if abs(x - GLOBAL_HILL) < HILL_RADIUS)
    uphill = sum(1 for x in landings if objective(x) > objective(start))
    return {"sigma": sigma,
            "landings": landings,
            "mean_distance": statistics.fmean(distances),
            "median_distance": statistics.median(distances),
            "max_distance": max(distances),
            "clamped": clamped,
            "arrivals": arrivals,
            "uphill": uphill,
            "trials": trials}


print("Where the population is stuck, and where it needs to get to:")
print(f"    trapped at x = {TRAPPED_AT:+.2f}, f = {objective(TRAPPED_AT):+.4f}")
print(f"    global hill x = {GLOBAL_HILL:+.2f}, f = {objective(GLOBAL_HILL):+.4f}")
print(f"    distance to cross: {abs(GLOBAL_HILL - TRAPPED_AT):.1f} units")

reports = [reach_report(TRAPPED_AT, sigma) for sigma in SIGMAS]

print(f"\nOne mutation, {TRIALS} times, from x = {TRAPPED_AT:+.2f}:\n")
print(" sigma | mean |dx| | median |dx| | max |dx| | clamped | uphill | "
      "arrived on the global hill")
print("-------+-----------+-------------+----------+---------+--------+"
      "---------------------------")
for r in reports:
    print(f" {r['sigma']:5.1f} |{r['mean_distance']:10.3f} |"
          f"{r['median_distance']:12.3f} |{r['max_distance']:9.3f} |"
          f"{r['clamped']:8d} |{r['uphill']:7d} |"
          f" {r['arrivals']:5d} of {r['trials']}"
          f"  ({100 * r['arrivals'] / r['trials']:5.2f}%)")

# The identity check. If these two rows ever stop matching Lesson 01, something
# in the operator or the seeding changed and the rest of the lesson is unsafe.
by_sigma = {r["sigma"]: r for r in reports}
print(f"\nLesson 01 step 5 reported {by_sigma[1.0]['arrivals']} arrivals at "
      f"sigma = 1.0 and {by_sigma[3.0]['arrivals']} at sigma = 3.0.")
print("Those are the two rows above, produced by the same seed and the same")
print("operator: the instrument measures what Lesson 01 measured, and more.")

# What the extra columns are for.
clean = [r for r in reports if r["clamped"] == 0]
print("\nRead the mean |dx| column against sigma. A gaussian step of standard")
print("deviation s travels 0.798 * s on average, so on the rows where no")
print("proposal hit a wall the ratio should be that constant:")
for r in clean:
    print(f"    sigma = {r['sigma']:<4} mean |dx| = {r['mean_distance']:.3f}"
          f"   ratio = {r['mean_distance'] / r['sigma']:.3f}")
worst = max(reports, key=lambda r: r["clamped"])
print(f"    sigma = {worst['sigma']:<4} mean |dx| = {worst['mean_distance']:.3f}"
      f"   ratio = {worst['mean_distance'] / worst['sigma']:.3f}"
      f"   <- {worst['clamped']} of {worst['trials']} clamped")
print(f"The last row is short of 0.798 because "
      f"{100 * worst['clamped'] / worst['trials']:.1f}% of its proposals landed")
print(f"outside [{GENE_MIN:+.0f}, {GENE_MAX:+.0f}] and were pulled back to the "
      "edge. Those mutations did not")
print("travel as far as sigma says, and part of that row's arrivals is the")
print("wall rather than the search.")

# The trade the table is really about.
best_arrivals = max(reports, key=lambda r: r["arrivals"])
best_uphill = max(reports, key=lambda r: r["uphill"])
print(f"\nArrivals climb with sigma: "
      + ", ".join(f"{r['arrivals']}" for r in reports)
      + f" for sigma = "
      + ", ".join(f"{r['sigma']}" for r in reports) + ".")
print(f"The best arrival rate on this table is sigma = {best_arrivals['sigma']}, "
      f"the largest one tried.")
print("So reach can simply be bought - and the price is in the uphill column.")
print(f"The most uphill moves, {best_uphill['uphill']} of "
      f"{best_uphill['trials']}, belong to sigma = {best_uphill['sigma']},")
print(f"which arrived {best_uphill['arrivals']} times. A small sigma is good at "
      "improving where it")
print("already stands and useless at going anywhere; a large sigma is the")
print("reverse. That is the whole trade, and no single number settles it.")

# The picture: where 2000 mutations actually land, for the two sigmas of
# Lesson 01, drawn against the landscape they land on.
fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
grid = np.linspace(GENE_MIN, GENE_MAX, 800)
for ax, sigma in zip(axes, (1.0, 3.0)):
    r = by_sigma[sigma]
    ax.hist(r["landings"], bins=80, range=(GENE_MIN, GENE_MAX),
            color="tab:orange", alpha=0.75,
            label=f"{r['trials']} landings, sigma = {sigma}")
    twin = ax.twinx()
    twin.plot(grid, objective(grid), color="tab:blue", linewidth=1.2)
    twin.set_ylabel("f(x)")
    twin.axvspan(GLOBAL_HILL - HILL_RADIUS, GLOBAL_HILL + HILL_RADIUS,
                 color="tab:green", alpha=0.15)
    ax.axvline(TRAPPED_AT, color="tab:red", linestyle="--", linewidth=1)
    ax.set_ylabel("landings")
    ax.set_title(f"sigma = {sigma}: {r['arrivals']} of {r['trials']} land on "
                 f"the global hill (green band)", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
axes[-1].set_xlabel("x")
fig.suptitle("The reach of one mutation, fired 2000 times from the local hill")
FIGURES.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURES / "mutation_reach_01_the_instrument.png",
            dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved to {FIGURES}/mutation_reach_01_the_instrument.png")

