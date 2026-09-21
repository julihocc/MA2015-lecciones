"""
Lesson 05 - Step 3: Mutation that refuses to make things worse
===============================================================
NEW IN THIS STEP: the fitness-driven variant of random deviation, and the
counter that prices it.

Every mutation so far has been blind: it moves the genes and accepts whatever
the objective says about the result. Most of the time the result is worse -
step 1 already showed that at sigma = 3.0 only 144 of 2000 mutations were
uphill. Gridin's last operator refuses those: it retries, up to MAX_TRIES
times, and returns the first mutant that beats its parent.

It works. It also costs, and the cost is the interesting part, because it is
paid in the only currency a genetic algorithm really spends: calls to the
fitness function.

CHANGES FROM mutation_reach_02_rate_and_step.py
Introduce them in this order:
    1. MAX_TRIES              how many attempts one mutation is allowed
    2. mutate_fitness_driven()  retry until the mutant beats its parent, or give up
    3. compare()              measure both operators per mutation AND per fitness call

Run it:  python mutation_reach_03_fitness_driven.py

mutate_fitness_driven() retries random deviation until the mutant beats its
parent, or until MAX_TRIES. compare() prices that filter: mean fitness change
flips from -1.0337 to +0.0548 and arrivals double (102 to 273), but it spends
5603 fitness calls against 2000, so per thousand calls it arrives 48.7 times
against blind mutation's 51.0.
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
GENE_COUNT = 10
REGIMES = ((0.1, 3.0), (0.3, 1.0), (1.0, 0.3), (0.3, 3.0), (1.0, 1.0))
# --- NEW (1) MAX_TRIES --------------------------------------------------------
# Gridin's default. It is a budget, not a guarantee: if none of the three
# attempts improves on the parent, the parent is returned unchanged and the
# three fitness evaluations are gone anyway.
MAX_TRIES = 3
# ------------------------------------------------------------------------------
# Back to the one-gene individual of step 1 and its trapped population, because
# the question here is not how movement is distributed across a chromosome but
# whether filtering movement by fitness helps a search that needs to escape.
TRAPPED_AT = -4.6
GLOBAL_HILL = 1.38
HILL_RADIUS = 1.5
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
        self.fitness = sum(float(objective(g)) for g in gene_list)

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


def mutate_random_deviation(gene_list: list[float], mu: float, sigma: float,
                            p: float) -> list[float]:
    """Gridin's operator: walk the chromosome and disturb each gene with prob p.

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


# --- NEW (2) mutate_fitness_driven() ------------------------------------------
def mutate_fitness_driven(individual: Individual, mu: float, sigma: float,
                          p: float, max_tries: int = MAX_TRIES
                          ) -> tuple[Individual, int]:
    """Retry random deviation until the mutant beats its parent, or give up.

    The second return value is the number of fitness evaluations this single
    mutation consumed. Returning it is the whole reason this step exists: the
    operator's advantage is obvious and its price is not, and a figure of merit
    that ignores the price will always prefer it.

    Note what the acceptance test can and cannot see. It compares the mutant
    with its own parent only, so it can never accept a step that goes down in
    order to come up somewhere better - and on a flat region of the landscape
    it cannot accept anything at all, because nothing there is strictly better.

    Args:
        individual: the parent.
        mu, sigma, p: the underlying random-deviation operator.
        max_tries: 3 here.

    Returns:
        (mutant_or_parent, evaluations_spent).

    Example:
        Turns mean fitness change from -1.0337 into +0.0548 and
        doubles arrivals (102 to 273) at 5603 fitness calls against
        2000. Acceptance 20.2% is 1-(1-q)^3 for the blind uphill rate q.
    """
    evaluations = 0
    for _ in range(max_tries):
        mutant = Individual(mutate_random_deviation(individual.gene_list,
                                                    mu, sigma, p))
        evaluations += 1
        if mutant.fitness > individual.fitness:
            return mutant, evaluations
    return individual, evaluations
# ------------------------------------------------------------------------------


# --- NEW (3) compare() --------------------------------------------------------
def compare(start: float, sigma: float, trials: int = TRIALS) -> dict:
    """Run both operators from the same gene, under the same seed, and price them.

    Every per-mutation figure is reported twice: once per mutation, which is how
    an operator usually gets judged, and once per thousand fitness evaluations,
    which is what a run actually pays. p = 1.0 here so that "one mutation" means
    the same thing for both operators on a single-gene chromosome.

    Args:
        start: the trapped gene, -4.6.
        sigma: 3.0 in the printed comparison.
        trials: 2000.

    Returns:
        Per-mutation and per-evaluation figures for blind vs driven.

    Example:
        Per thousand calls the driven operator arrives 48.7 times
        against blind mutation's 51.0.
    """
    parent = Individual([start])

    random.seed(SEED)
    blind_deltas, blind_arrivals, blind_evaluations = [], 0, 0
    for _ in range(trials):
        mutant = Individual(mutate_random_deviation([start], MUTATION_MU,
                                                    sigma, 1.0))
        blind_evaluations += 1
        blind_deltas.append(mutant.fitness - parent.fitness)
        if abs(mutant.gene - GLOBAL_HILL) < HILL_RADIUS:
            blind_arrivals += 1

    random.seed(SEED)
    driven_deltas, driven_arrivals, driven_evaluations, accepted = [], 0, 0, 0
    for _ in range(trials):
        mutant, evaluations = mutate_fitness_driven(parent, MUTATION_MU,
                                                    sigma, 1.0)
        driven_evaluations += evaluations
        driven_deltas.append(mutant.fitness - parent.fitness)
        accepted += mutant is not parent
        if abs(mutant.gene - GLOBAL_HILL) < HILL_RADIUS:
            driven_arrivals += 1

    return {"sigma": sigma, "trials": trials,
            "blind_mean_delta": statistics.fmean(blind_deltas),
            "blind_uphill": sum(1 for d in blind_deltas if d > 0),
            "blind_arrivals": blind_arrivals,
            "blind_evaluations": blind_evaluations,
            "driven_mean_delta": statistics.fmean(driven_deltas),
            "driven_accepted": accepted,
            "driven_arrivals": driven_arrivals,
            "driven_evaluations": driven_evaluations,
            "blind_deltas": blind_deltas,
            "driven_deltas": driven_deltas}
# ------------------------------------------------------------------------------


print(f"Mutating a single gene sitting at x = {TRAPPED_AT:+.2f} "
      f"(f = {objective(TRAPPED_AT):+.4f}),")
print(f"{TRIALS} times with each operator, at MAX_TRIES = {MAX_TRIES}.\n")

results = [compare(TRAPPED_AT, sigma) for sigma in (1.0, 3.0)]

ROW = " {:>5} | {:<14} | {:>16} | {:>18} | {:>8} | {:>13}"
header = ROW.format("sigma", "operator", "mean change in f",
                    "accepted or uphill", "arrivals", "fitness calls")
print(header)
print("-" * len(header))
for r in results:
    print(ROW.format(f"{r['sigma']:.1f}", "blind",
                     f"{r['blind_mean_delta']:+.4f}",
                     f"{r['blind_uphill']} of {r['trials']}",
                     r["blind_arrivals"], r["blind_evaluations"]))
    print(ROW.format("", "fitness-driven",
                     f"{r['driven_mean_delta']:+.4f}",
                     f"{r['driven_accepted']} of {r['trials']}",
                     r["driven_arrivals"], r["driven_evaluations"]))

print("\nThe uphill counts differ a little from step 1's table (140 and 144)")
print("because mutate_random_deviation() throws a coin per gene before drawing")
print("the noise, so the two operators consume the random stream differently.")
print("Within this script both columns were drawn from the same seed, which is")
print("what the comparison needs.")

for r in results:
    sigma = r["sigma"]
    print(f"\n--- sigma = {sigma} " + "-" * (58 - len(str(sigma))))
    print(f"Blind mutation changes fitness by {r['blind_mean_delta']:+.4f} on "
          f"average; the filter turns that")
    print(f"into {r['driven_mean_delta']:+.4f}. It is doing exactly what it "
          "claims to do.")
    expected = 1 - (1 - r["blind_uphill"] / r["trials"]) ** MAX_TRIES
    print(f"It accepted {r['driven_accepted']} of {r['trials']} mutations "
          f"({100 * r['driven_accepted'] / r['trials']:.1f}%), against "
          f"{r['blind_uphill']} uphill draws")
    print(f"({100 * r['blind_uphill'] / r['trials']:.1f}%) for blind mutation. "
          f"Three independent attempts at a {100 * r['blind_uphill'] / r['trials']:.1f}%")
    observed = r["driven_accepted"] / r["trials"]
    print(f"event succeed {100 * expected:.1f}% of the time; the acceptance "
          f"column says {100 * observed:.1f}%,")
    print(f"a gap of {100 * abs(observed - expected):.1f} points. The filter "
          "adds no new ability - it just")
    print("buys more tickets in the same lottery.")
    blind_rate = 1000 * r["blind_arrivals"] / r["blind_evaluations"]
    driven_rate = 1000 * r["driven_arrivals"] / r["driven_evaluations"]
    print(f"\nArrivals on the global hill: {r['blind_arrivals']} blind, "
          f"{r['driven_arrivals']} fitness-driven - the filter")
    if r["driven_arrivals"] > r["blind_arrivals"]:
        verdict = "reaches the far hill more often"
    elif r["driven_arrivals"] < r["blind_arrivals"]:
        verdict = "reaches the far hill less often"
    else:
        verdict = "does not change how often the far hill is reached"
    print(f"{verdict}. But it spent "
          f"{r['driven_evaluations']} fitness calls against")
    print(f"{r['blind_evaluations']}, so per thousand calls the rates are "
          f"{blind_rate:.1f} blind and {driven_rate:.1f} driven.")
    if blind_rate == driven_rate == 0.0:
        print("Neither operator ever gets there, so there is nothing to")
        print("compare: at this sigma the gap is simply too wide for one jump,")
        print("and no amount of filtering creates reach that the step size")
        print("does not have.")
    elif driven_rate < blind_rate:
        print("Per mutation the filter wins; per fitness evaluation it LOSES.")
    else:
        print("The filter wins on both counts here.")

print("\nThat is the verdict on this operator, and it is not the one its name")
print("suggests. Retrying until you improve is an excellent way to make one")
print("mutation look good and a poor way to spend an evaluation budget - and in")
print("Lessons 06 and 07, the evaluation budget is the thing being measured.")
print("\nThere is a second limit, structural rather than statistical. The test is")
print(f"`mutant.fitness > parent.fitness`, with no tolerance, so on a region")
print("where the landscape is flat nothing is ever strictly better and the")
print("operator returns the parent every single time, having paid MAX_TRIES")
print("evaluations for it. Step 4 runs on exactly such a landscape.")

# The picture: the distribution of fitness change, blind against filtered.
fig, axes = plt.subplots(1, len(results), figsize=(5.2 * len(results), 3.6))
for ax, r in zip(axes, results):
    bins = np.linspace(min(min(r["blind_deltas"]), min(r["driven_deltas"])),
                       max(max(r["blind_deltas"]), max(r["driven_deltas"])), 60)
    ax.hist(r["blind_deltas"], bins=bins, color="tab:red", alpha=0.6,
            label=f"blind (mean {r['blind_mean_delta']:+.3f})")
    ax.hist(r["driven_deltas"], bins=bins, color="tab:green", alpha=0.6,
            label=f"fitness-driven (mean {r['driven_mean_delta']:+.3f})")
    ax.set_yscale("log")
    ax.axvline(0.0, color="grey", linewidth=0.9, linestyle="--")
    ax.set_title(f"sigma = {r['sigma']}", fontsize=10)
    ax.set_xlabel("change in fitness after one mutation")
    ax.set_ylabel("count (log)")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(fontsize=8, loc="upper left")
fig.suptitle("The filter removes the left tail - and pays for it in fitness calls")
FIGURES.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURES / "mutation_reach_03_fitness_driven.png",
            dpi=150, bbox_inches="tight")
plt.close(fig)
print("Comparing the two operators across whole runs, at equal evaluation")
print("budget, is Lesson 06. This step prices one mutation, not a search.")
print(f"\nFigure saved to {FIGURES}/mutation_reach_03_fitness_driven.png")

