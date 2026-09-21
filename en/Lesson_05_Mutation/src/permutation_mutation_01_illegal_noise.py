"""
Lesson 05 - Step 1 of the second example: when noise is not a legal move
=========================================================================
Every operator so far has added a number to a gene. That works because a gene
was a real value and any real value in [-10, +10] was a legal gene.

Change the encoding and the whole idea collapses. If a chromosome is an ORDER -
the sequence in which ten places are visited, ten jobs are run, ten items are
packed - then its genes are not free values. They are a permutation: each of
the ten symbols appears exactly once. Add a gaussian to one of them and the
result is not a worse solution, it is not a solution at all.

This is not a special case. It is half of the mutation operators in Gridin's
Chapter 5, and it is the encoding of every problem in Lessons 09 to 11.

The measuring idea from the first example survives the change of encoding, and
that is why this half of the lesson can be read against the other. Reach there
was distance in gene units. Reach here has two components, and they turn out not
to agree: how many positions the mutation disturbs, and how far the symbols
actually travel.

Run it:  python permutation_mutation_01_illegal_noise.py

is_permutation() is the legality test an order encoding actually has. Random
deviation on a permutation produces 1935 invalid mutants of 2000, and 0 of the
65 survivors had any gene touched. Bit flip changes exactly 1 position every
time; exchange exactly 2.
"""
import copy
import random
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

SEED = 52
CHROMOSOME_LENGTH = 10
TRIALS = 2000
MUTATION_MU, MUTATION_SIGMA = 0.0, 1.0
FIGURES = Path(__file__).resolve().parent.parent / "figures"


def is_permutation(gene_list: list, alphabet: list) -> bool:
    """A chromosome is legal only if it is a rearrangement of the alphabet.

    Args:
        gene_list: claimed mutant.
        alphabet: the original symbols.

    Returns:
        True iff the mutant is a rearrangement of the alphabet.

    Example:
        Random deviation: 1935 invalid mutants of 2000.
    """
    return sorted(gene_list) == sorted(alphabet)


def positions_changed(original: list, mutant: list) -> int:
    """How many positions hold a different symbol after the mutation.

    Args:
        original, mutant: equal-length sequences.

    Returns:
        How many positions hold a different symbol.

    Example:
        Bit flip: exactly 1 every time. Exchange: exactly 2.
        Shift: 4.663 on average.
    """
    return sum(1 for a, b in zip(original, mutant) if a != b)


def travel(original: list, mutant: list) -> float:
    """Mean distance, in positions, that a symbol moved.

    The counterpart of |dx| in the first example: `positions_changed` says how
    much of the chromosome was disturbed, `travel` says how far the disturbance
    carried anything. Two operators can score the same on one and not the other,
    which is the finding the next step is built on.

    Args:
        original, mutant: permutations of the same symbols.

    Returns:
        Mean distance, in positions, that a symbol moved.

    Example:
        Inversion 1.308 > shuffle 0.876 > exchange 0.733 = shift 0.733.
        Exchange and shift give identical total travel on 90 of 90
        position pairs.
    """
    where = {symbol: index for index, symbol in enumerate(mutant)}
    return statistics.fmean(abs(where[symbol] - index)
                            for index, symbol in enumerate(original))


def mutate_random_deviation(gene_list: list[float], mu: float, sigma: float,
                            p: float) -> list[float]:
    """Step 3's operator, unchanged, about to be pointed at the wrong encoding.

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
            mutant[i] = mutant[i] + random.gauss(mu, sigma)
    return mutant


def mutation_bit_flip(gene_list: list[int]) -> list[int]:
    """Gridin's operator for a BINARY chromosome: pick a position, flip it.

    Worth pausing on. On a binary gene there is no such thing as a small
    perturbation - 0 and 1 are adjacent and there is nothing between them - so
    the operator has no sigma and no step size at all. The only dial left is how
    many bits to flip, which is the rate dial and nothing else.

    Args:
        gene_list: a binary chromosome.

    Returns:
        A copy with exactly one bit flipped.

    Example:
        Changes exactly 1 position every time; there is no sigma
        on a binary gene.
    """
    mutant = copy.deepcopy(gene_list)
    position = random.randint(0, len(mutant) - 1)
    mutant[position] = (mutant[position] + 1) % 2
    return mutant


def mutation_exchange(gene_list: list, positions: tuple[int, int] | None = None
                      ) -> list:
    """Gridin's operator for a PERMUTATION: swap the symbols at two positions.

    This is the minimal legal move. Any mutation of a permutation has to be a
    rearrangement of what is already there, and a swap is the smallest
    rearrangement that changes anything at all.

    `positions` is optional so that the next step can ask what an operator does
    to a GIVEN pair of positions rather than to a random one. Left as None it
    behaves exactly as the book's version.

    Args:
        gene_list: a permutation.
        positions: the pair to swap, or None to draw one.

    Returns:
        A copy with two symbols swapped. Always a permutation.

    Example:
        Changes exactly 2 positions; travel 0.733, identical to shift
        on 90 of 90 pairs.
    """
    mutant = copy.deepcopy(gene_list)
    if positions is None:
        positions = random.sample(range(len(mutant)), 2)
    i, j = positions
    mutant[i], mutant[j] = mutant[j], mutant[i]
    return mutant


def rearrangement_report(gene_list: list, operator, trials: int = TRIALS) -> dict:
    """Apply one operator `trials` times and describe the damage.

    The instrument of the first example, re-cut for orderings: re-seed, fire the
    operator many times at the same individual, and report distributions rather
    than one example.

    Args:
        gene_list: the individual being mutated.
        operator: a mutation that takes a gene list.
        trials: 2000, reseeded from SEED = 52.

    Returns:
        illegal count, positions-changed distribution, travel
        distribution, unchanged count.

    Example:
        Random deviation: 1935 illegal of 2000, and 0 of the 65
        survivors had any gene touched.
    """
    random.seed(SEED)
    changed, distances, unchanged, illegal = [], [], 0, 0
    for _ in range(trials):
        mutant = operator(gene_list)
        if not is_permutation(mutant, gene_list):
            illegal += 1
            continue
        count = positions_changed(gene_list, mutant)
        changed.append(count)
        distances.append(travel(gene_list, mutant))
        if count == 0:
            unchanged += 1
    return {"trials": trials,
            "illegal": illegal,
            "legal": len(changed),
            "mean_changed": statistics.fmean(changed) if changed else 0.0,
            "max_changed": max(changed) if changed else 0,
            "mean_travel": statistics.fmean(distances) if distances else 0.0,
            "unchanged": unchanged,
            "changed": changed}


tour = list(range(1, CHROMOSOME_LENGTH + 1))
print(f"A chromosome that is an ORDER, not a vector of values:\n    {tour}")
print("Ten places, visited in this sequence. Every symbol appears exactly once,")
print("and that constraint is part of what a solution IS.")

# ----- the illegal operator ---------------------------------------------------
print(f"\nApply step 3's random deviation to it, {TRIALS} times, at p = 0.3:\n")
random.seed(SEED)
legal, legal_and_touched, examples = 0, 0, []
for _ in range(TRIALS):
    mutant = mutate_random_deviation(tour, MUTATION_MU, MUTATION_SIGMA, 0.3)
    if len(examples) < 2 and mutant != tour:
        examples.append(mutant)
    if is_permutation(mutant, tour):
        legal += 1
        if mutant != tour:
            legal_and_touched += 1
print(f"    {tour}")
for mutant in examples:
    print("    " + str([round(g, 2) for g in mutant]))
print(f"\n    {TRIALS - legal} of {TRIALS} mutants "
      f"({100 * (TRIALS - legal) / TRIALS:.1f}%) were not permutations.")
print(f"    Of the {legal} that were, {legal_and_touched} had any gene touched "
      f"at all.")
print("    So the operator's only legal output is the one where the coin never")
print("    came up: it is not a weak mutation on this encoding, it is no")
print("    mutation on this encoding.")

# The survival rate is exactly the probability of touching nothing.
print(f"\nSurvival rate against p, {TRIALS} draws each:\n")
print("       p | still a permutation | predicted (1-p)^n")
print("    -----+---------------------+------------------")
survival = []
for percent in range(0, 101, 20):
    p = percent / 100
    random.seed(SEED)
    valid = sum(1 for _ in range(TRIALS)
                if is_permutation(mutate_random_deviation(tour, MUTATION_MU,
                                                          MUTATION_SIGMA, p),
                                  tour))
    predicted = (1 - p) ** CHROMOSOME_LENGTH
    survival.append((p, valid / TRIALS, predicted))
    print(f"    {p:4.1f} |     {valid:5d} of {TRIALS}   |"
          f"       {predicted:12.4f}")
worst = max(abs(rate - predicted) for _, rate, predicted in survival)
print(f"\n    Measured and predicted never differ by more than {worst:.4f}.")
print("    A permutation cannot be perturbed. It can only be rearranged.")

# ----- the two legal operators ------------------------------------------------
print("\nTwo encodings, two answers.\n")
random.seed(SEED)
bits = [random.randint(0, 1) for _ in range(CHROMOSOME_LENGTH)]
random.seed(SEED)
flipped = mutation_bit_flip(bits)
print(f"    binary chromosome: {bits}")
print(f"    after bit flip:    {flipped}")
random.seed(SEED)
flip_changes = [positions_changed(bits, mutation_bit_flip(bits))
                for _ in range(TRIALS)]
print(f"    over {TRIALS} flips: always exactly "
      f"{min(flip_changes)} position changed "
      f"(min {min(flip_changes)}, max {max(flip_changes)}).")
print("    There is no step size to choose. On a binary gene, the smallest")
print("    possible change is the largest possible change.")

random.seed(SEED)
swapped = mutation_exchange(tour)
print(f"\n    permutation:       {tour}")
print(f"    after exchange:    {swapped}")
report = rearrangement_report(tour, mutation_exchange)
print(f"    over {report['trials']} exchanges: {report['illegal']} illegal "
      f"results,")
print(f"    always exactly {report['mean_changed']:.3f} positions changed "
      f"(max {report['max_changed']}),")
print(f"    mean travel {report['mean_travel']:.3f} positions per symbol.")
print("    Exchange is to a permutation what the smallest sigma is to a real")
print("    gene: the local move, the one that refines rather than explores.")

print("\nThe reach dial has to be rebuilt from scratch for this encoding, and")
print("the next step builds it: three operators that disturb more of the")
print("chromosome than a swap does, and an instrument that says by how much.")

# The picture: what the illegal operator does, and what the legal one does.
fig, (ax_valid, ax_swap) = plt.subplots(1, 2, figsize=(11, 3.8))
ax_valid.plot([p for p, _, _ in survival], [100 * r for _, r, _ in survival],
              "o-", color="tab:red", label=f"measured over {TRIALS} draws")
ax_valid.plot([p for p, _, _ in survival], [100 * q for _, _, q in survival],
              "--", color="tab:blue",
              label=f"(1 - p) ^ {CHROMOSOME_LENGTH}")
ax_valid.set_xlabel("per-gene mutation probability p")
ax_valid.set_ylabel("% of mutants still a permutation")
ax_valid.set_title("Random deviation on a permutation:\nthe only legal output "
                   "is the untouched one", fontsize=10)
ax_valid.grid(True, linestyle=":", alpha=0.5)
ax_valid.legend(fontsize=8)

ax_swap.hist(report["changed"], bins=range(0, CHROMOSOME_LENGTH + 2),
             align="left", rwidth=0.8, color="tab:green")
ax_swap.set_xticks(range(0, CHROMOSOME_LENGTH + 1))
ax_swap.set_xlabel("positions changed by one exchange")
ax_swap.set_ylabel(f"count of {report['trials']} mutations")
ax_swap.set_title("Exchange: the minimal legal move,\nand it is minimal every "
                  "time", fontsize=10)
ax_swap.grid(True, axis="y", linestyle=":", alpha=0.5)

fig.suptitle("Changing the encoding changes which mutations exist")
FIGURES.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURES / "permutation_mutation_01_illegal_noise.png",
            dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved to {FIGURES}/permutation_mutation_01_illegal_noise.png")

