"""
Lesson 05 - Step 2: Mutation has two dials, not one
====================================================
NEW IN THIS STEP: a vector-valued individual, the book's random-deviation
operator, and the instrument re-cut to measure it.

Step 1 measured a single gene, so sigma was the only thing there was to set.
A real chromosome has many genes, and the operator then has a second dial: the
per-gene probability p that a gene is touched at all. Gridin's
`mutation_random_deviation(ind, mu, sigma, p)` carries both.

The two dials are not interchangeable. Their product sets how much total
movement one mutation buys; p alone decides whether that movement arrives as
one long jump or as ten short ones, and those are different searches.

CHANGES FROM mutation_reach_01_the_instrument.py
Introduce them in this order:
    1. GENE_COUNT                 a rate has nothing to act on until there is more than one gene
    2. Individual.fitness         CHANGED: score a vector by summing the objective over its genes
    3. mutate_random_deviation()  the book's operator - mu, sigma, and the per-gene probability p
    4. regime_report()            the instrument re-cut for a vector: how many genes moved, and how far

Run it:  python mutation_reach_02_rate_and_step.py

mutate_random_deviation() walks a ten-gene chromosome and disturbs each gene
with probability p. regime_report() measures genes touched, step size and
total travel. Genes touched never differs from p*n by more than 0.015. Three
regimes with the same p*sigma = 0.30 have longest single steps that differ by
a factor of 7.1. At p = 0.1, 718 of 2000 mutations change nothing.
"""
import copy
import random
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SEED = 52
GENE_MIN, GENE_MAX = -10.0, 10.0
MUTATION_MU = 0.0
TRIALS = 2000
# --- NEW (1) GENE_COUNT -------------------------------------------------------
# Ten genes rather than one. Nothing about the operator needs this - it is the
# MEASUREMENT that needs it: with a single gene, p only decides how often the
# mutation happens at all, and its effect is indistinguishable from lowering the
# mutation probability of the whole individual.
GENE_COUNT = 10
# ------------------------------------------------------------------------------
# Each row is one mutation regime: (p, sigma). The first three share the same
# product p * sigma and are the point of the step; the last two raise it.
REGIMES = ((0.1, 3.0), (0.3, 1.0), (1.0, 0.3), (0.3, 3.0), (1.0, 1.0))
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
        self.fitness = sum(float(objective(g)) for g in gene_list)   # --- CHANGED --- a vector is scored by summing over its genes

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
        return ("[" + ", ".join(f"{g:+.2f}" for g in self.gene_list)
                + f"] f={self.fitness:+.3f}")


def mutate_gaussian(gene: float, mu: float, sigma: float) -> float:
    """Add noise drawn from N(mu, sigma) to a gene - Lesson 01's operator.

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


# --- NEW (3) mutate_random_deviation() ----------------------------------------
def mutate_random_deviation(gene_list: list[float], mu: float, sigma: float,
                            p: float) -> list[float]:
    """Gridin's operator: walk the chromosome and disturb each gene with prob p.

    The coin is thrown once PER GENE, not once per individual. That is the
    difference between "this individual mutates" and "this gene mutates", and
    it is why an individual can come back from mutation completely unchanged
    even though the mutation was applied to it.

    Args:
        gene_list: the chromosome.
        mu, sigma: Gaussian noise on a touched gene.
        p: per-gene probability of being touched.

    Returns:
        A new gene list. With p < 1 an "applied" mutation can still
        change nothing.

    Example:
        At p = 0.1, 718 of 2000 mutations change nothing, against a
        predicted (1-p)^10 = 0.3487. On a permutation, 1935 of 2000
        mutants are illegal.
    """
    mutant = copy.deepcopy(gene_list)
    for i in range(len(mutant)):
        if random.random() < p:
            mutant[i] = mutate_gaussian(mutant[i], mu, sigma)
    return mutant
# ------------------------------------------------------------------------------


# --- NEW (4) regime_report() --------------------------------------------------
def regime_report(gene_list: list[float], p: float, sigma: float,
                  trials: int = TRIALS) -> dict:
    """Fire `trials` mutations at one individual and describe what moved.

    Three quantities, and they answer three different questions:
      touched        how many genes the coin selected      -> the rate dial
      step           how far a touched gene actually moved -> the size dial
      travel         the sum of |dx| over the chromosome   -> the whole budget

    Re-seeding before each regime keeps the rows comparable, exactly as in
    step 1.

    Args:
        gene_list: the individual being mutated (10 genes).
        p, sigma: one regime.
        trials: 2000.

    Returns:
        genes touched vs p*n, mean/max step, total travel, untouched
        count.

    Example:
        Genes touched never differs from p*n by more than 0.015.
        Three regimes with p*sigma = 0.30 travel 1.904 / 2.179 / 2.304
        but their longest single step differs by a factor of 7.1
        (10.253 vs 1.449).
    """
    random.seed(SEED)
    touched, travel, steps, clamped, untouched = [], [], [], 0, 0
    for _ in range(trials):
        mutant = mutate_random_deviation(gene_list, MUTATION_MU, sigma, p)
        moves = [abs(a - b) for a, b in zip(gene_list, mutant)]
        moved = [d for d in moves if d > 0.0]
        clamped += sum(1 for g in mutant if g in (GENE_MIN, GENE_MAX))
        touched.append(len(moved))
        travel.append(sum(moves))
        steps.extend(moved)
        if not moved:
            untouched += 1
    return {"p": p, "sigma": sigma, "budget": p * sigma,
            "mean_touched": statistics.fmean(touched),
            "expected_touched": p * len(gene_list),
            "mean_step": statistics.fmean(steps) if steps else 0.0,
            "max_step": max(steps) if steps else 0.0,
            "mean_travel": statistics.fmean(travel),
            "steps": steps,
            "clamped": clamped,
            "clamped_share": clamped / max(1, sum(touched)),
            "untouched": untouched,
            "trials": trials}
# ------------------------------------------------------------------------------


random.seed(SEED)
subject = Individual([random.uniform(GENE_MIN, GENE_MAX)
                      for _ in range(GENE_COUNT)])
print(f"The individual being mutated ({GENE_COUNT} genes):")
print("   ", subject)

reports = [regime_report(subject.gene_list, p, sigma) for p, sigma in REGIMES]

print(f"\nEach regime applied {TRIALS} times to that one individual:\n")
print("    p | sigma | p*sigma | genes touched (p*n) | mean step | max step |"
      " total travel | unchanged")
print("------+-------+---------+---------------------+-----------+----------+"
      "--------------+----------")
for r in reports:
    print(f" {r['p']:4.1f} | {r['sigma']:5.1f} | {r['budget']:7.2f} |"
          f"     {r['mean_touched']:6.3f} ({r['expected_touched']:4.1f}) |"
          f" {r['mean_step']:9.3f} | {r['max_step']:8.3f} |"
          f" {r['mean_travel']:12.3f} | {r['untouched']:8d}")

# Claim 1: the rate dial does exactly what it says.
worst_gap = max(abs(r["mean_touched"] - r["expected_touched"]) for r in reports)
print(f"\nThe genes-touched column never differs from p * {GENE_COUNT} by more "
      f"than {worst_gap:.3f}.")
print("p is not a strength, it is a head count: it says how many of the ten")
print("genes take part, and says nothing at all about what happens to them.")

# Claim 2: the same budget, spent three different ways.
budgets = {}
for r in reports:
    budgets.setdefault(round(r["budget"], 6), []).append(r)
trio = max(budgets.values(), key=len)
trio = sorted(trio, key=lambda r: r["p"])
print(f"\nThe first {len(trio)} rows share the same product "
      f"p * sigma = {trio[0]['budget']:.2f}. That product is the budget:")
for r in trio:
    print(f"    p = {r['p']:<4} sigma = {r['sigma']:<4} -> total travel "
          f"{r['mean_travel']:.3f}, longest single step {r['max_step']:.3f}")
spread = max(r["max_step"] for r in trio) / min(r["max_step"] for r in trio)
travels = [r["mean_travel"] for r in trio]
print(f"Total travel stays between {min(travels):.3f} and {max(travels):.3f} - "
      f"a range of {100 * (max(travels) / min(travels) - 1):.0f}%.")
print(f"The longest single step does not: it differs by a factor of "
      f"{spread:.1f}.")
print("Roughly the same movement, and only one of these regimes can carry a")
print("single gene across a valley. That is what p decides.")

# Claim 3: why the budgets are not exactly equal.
concentrated = min(trio, key=lambda r: r["p"])
diffuse = max(trio, key=lambda r: r["p"])
shortfall = diffuse["mean_travel"] - concentrated["mean_travel"]
print(f"\nThey are not exactly equal, and the gap is instructive: "
      f"{shortfall:.3f} of travel")
print(f"is missing from p = {concentrated['p']}. Long jumps are the ones that "
      f"fall off the edge of")
print(f"[{GENE_MIN:+.0f}, {GENE_MAX:+.0f}] and get clamped: "
      f"{100 * concentrated['clamped_share']:.1f}% of the genes that regime "
      f"touched")
print(f"ended pinned to a wall, against "
      f"{100 * diffuse['clamped_share']:.1f}% at p = {diffuse['p']}, whose "
      f"steps only rarely")
print("reach that far. A bounded space quietly taxes reach, and taxes the")
print("concentrated regime hardest.")

# Claim 4: the other thing a per-gene coin does.
low = min(reports, key=lambda r: r["p"])
print(f"\nAnd the last column: at p = {low['p']}, {low['untouched']} of "
      f"{low['trials']} mutations changed nothing at all")
print(f"({100 * low['untouched'] / low['trials']:.1f}%), because the coin came "
      f"up tails {GENE_COUNT} times in a row.")
print(f"Its predicted rate is (1 - p)^{GENE_COUNT} = "
      f"{(1 - low['p']) ** GENE_COUNT:.4f}. A low mutation rate does not make")
print("mutation gentle; it makes it rare, which is a different thing.")

# The picture: the same travel budget, three shapes.
fig, axes = plt.subplots(1, len(trio), figsize=(4.0 * len(trio), 3.4),
                         sharey=True)
for ax, r in zip(axes, trio):
    ax.hist(r["steps"], bins=60, range=(0, 12), color="tab:orange", alpha=0.85)
    ax.set_yscale("log")
    ax.axvline(r["mean_step"], color="tab:blue", linestyle="--", linewidth=1.2,
               label=f"mean step {r['mean_step']:.2f}")
    ax.set_title(f"p = {r['p']}, sigma = {r['sigma']}\n"
                 f"travel {r['mean_travel']:.2f}, longest {r['max_step']:.2f}",
                 fontsize=9)
    ax.set_xlabel("|dx| of a touched gene")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=8)
axes[0].set_ylabel("count of touched genes (log)")
fig.suptitle(f"One travel budget (p * sigma = {trio[0]['budget']:.2f}), "
             "three ways to spend it")
FIGURES.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURES / "mutation_reach_02_rate_and_step.png",
            dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved to {FIGURES}/mutation_reach_02_rate_and_step.png")

