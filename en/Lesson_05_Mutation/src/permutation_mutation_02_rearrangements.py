"""
Lesson 05 - Step 2 of the second example: three bigger rearrangements
======================================================================
NEW IN THIS STEP: inversion, shift and shuffle, and the comparison that puts
all four permutation operators on one scale.

Exchange is the minimal legal move. Gridin gives three larger ones, and all
three work the same way: pick two positions and rearrange the block between
them. Inversion reverses it, shift rotates it by one place, shuffle randomises
it. The question this step answers is the one the first example asked about
sigma - how far does each of them reach? - and the answer needs two numbers,
not one.

The two numbers disagree, and that is the result. The operator that disturbs
the most positions moves the symbols the least. The operator that sounds most
destructive does nothing at all 13.6% of the time.

CHANGES FROM permutation_mutation_01_illegal_noise.py
Introduce them in this order:
    1. mutation_inversion()   reverse the block: the same symbols, read backwards
    2. mutation_shift()       rotate the block by one place: everyone moves, barely
    3. mutation_shuffle()     randomise the block, which sometimes means not changing it
    4. compare_operators()    the four legal operators on one scale, and the exact check

Run it:  python permutation_mutation_02_rearrangements.py

mutation_inversion, mutation_shift and mutation_shuffle are the legal moves on
an order. By positions disturbed: shift 4.663 > inversion 4.230 > shuffle
3.708 > exchange 2.000. By distance travelled: inversion 1.308 > shuffle 0.876
> exchange 0.733 = shift 0.733. Shuffle returns the individual unchanged 271
of 2000 times.
"""
import copy
import random
import statistics
from math import copysign
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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


def mutation_bit_flip(gene_list: list[int]) -> list[int]:
    """Gridin's operator for a BINARY chromosome: pick a position, flip it.

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
    """Swap the symbols at two positions: the minimal legal move.

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


# --- NEW (1) mutation_inversion() ---------------------------------------------
def mutation_inversion(gene_list: list, positions: tuple[int, int] | None = None
                       ) -> list:
    """Reverse the block between two positions.

    The block's contents are untouched as a SET; only the direction of reading
    changes. On a route this is the classic move: undo one crossing of the path
    without disturbing anything outside the reversed leg. Note that it moves the
    two ends of the block furthest and leaves the middle almost where it was.

    Args:
        gene_list: a permutation.
        positions: the slice to reverse, or None to draw one.

    Returns:
        A copy with one slice reversed. Always a permutation.

    Example:
        Highest travel of the four (1.308) at 4.230 positions changed.
    """
    mutant = copy.deepcopy(gene_list)
    source = copy.deepcopy(gene_list)
    if positions is None:
        positions = random.sample(range(len(mutant)), 2)
    low, high = sorted(positions)
    for offset in range(0, high - low + 1):
        mutant[low + offset] = source[high - offset]
    return mutant
# ------------------------------------------------------------------------------


# --- NEW (2) mutation_shift() -------------------------------------------------
def mutation_shift(gene_list: list, positions: tuple[int, int] | None = None
                   ) -> list:
    """Take the symbol at one position and re-insert it at another.

    Everything between the two positions slides one place to make room. So the
    number of positions this disturbs is the whole span - which sounds violent -
    while every symbol except one moves exactly one place, which is not.

    Args:
        gene_list: a permutation.
        positions: (from, to), or None to draw them.

    Returns:
        A copy with one symbol moved. Always a permutation.

    Example:
        Most positions disturbed (4.663) but travel identical to
        exchange (0.733).
    """
    mutant = copy.deepcopy(gene_list)
    if positions is None:
        positions = random.sample(range(len(mutant)), 2)
    source, target = positions
    travelling = mutant[source]
    direction = int(copysign(1, target - source))
    for index in range(source, target, direction):
        mutant[index] = mutant[index + direction]
    mutant[target] = travelling
    return mutant
# ------------------------------------------------------------------------------


# --- NEW (3) mutation_shuffle() -----------------------------------------------
def mutation_shuffle(gene_list: list, positions: tuple[int, int] | None = None
                     ) -> list:
    """Randomise the order of the block between two positions.

    The only one of the four whose output is not determined by the two positions
    it drew - and the only one that can return the individual unchanged, because
    a random permutation of a short block is quite often the identity. That is
    measured below rather than asserted.

    Args:
        gene_list: a permutation.
        positions: the slice to shuffle, or None to draw one.

    Returns:
        A copy with one slice randomly permuted. Always a permutation.

    Example:
        Returns the individual unchanged 271 of 2000 times (13.6%).
    """
    mutant = copy.deepcopy(gene_list)
    if positions is None:
        positions = random.sample(range(len(mutant)), 2)
    low, high = sorted(positions)
    block = mutant[low:high + 1]
    random.shuffle(block)
    mutant[low:high + 1] = block
    return mutant
# ------------------------------------------------------------------------------


def rearrangement_report(gene_list: list, operator, trials: int = TRIALS) -> dict:
    """Apply one operator `trials` times and describe the damage.

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
            "mean_changed": statistics.fmean(changed) if changed else 0.0,
            "max_changed": max(changed) if changed else 0,
            "mean_travel": statistics.fmean(distances) if distances else 0.0,
            "max_travel": max(distances) if distances else 0.0,
            "unchanged": unchanged,
            "changed": changed}


# --- NEW (4) compare_operators() ----------------------------------------------
OPERATORS = (("exchange", mutation_exchange),
             ("inversion", mutation_inversion),
             ("shift", mutation_shift),
             ("shuffle", mutation_shuffle))


def compare_operators(gene_list: list, trials: int = TRIALS) -> list[dict]:
    """Every legal operator, same individual, same seed, same instrument.

    This is the counterpart of step 4's regime sweep in the first example: the
    rows are comparable because the only thing that differs between them is
    which operator was called.

    Args:
        gene_list: the permutation under test.
        trials: 2000.

    Returns:
        One row per rearrangement operator, both reach measurements.

    Example:
        The two rankings disagree: by positions, shift wins; by
        travel, inversion wins.
    """
    rows = []
    for name, operator in OPERATORS:
        report = rearrangement_report(gene_list, operator, trials)
        report["name"] = name
        rows.append(report)
    return rows


def travel_over_all_pairs(gene_list: list, operator) -> dict[tuple[int, int], float]:
    """Every ordered pair of distinct positions, fed to one operator.

    No randomness at all: this asks what the operator DOES, not what it does on
    average. Only meaningful for the three operators whose output is fixed by
    the pair, which is why shuffle is left out of it.

    Args:
        gene_list: the permutation.
        operator: exchange or shift, with an explicit position pair.

    Returns:
        Travel for each of the 90 unordered pairs.

    Example:
        Exchange and shift give identical total travel on 90 of 90 pairs.
    """
    length = len(gene_list)
    return {(i, j): travel(gene_list, operator(gene_list, (i, j)))
            for i in range(length) for j in range(length) if i != j}
# ------------------------------------------------------------------------------


tour = list(range(1, CHROMOSOME_LENGTH + 1))
print(f"The same chromosome as step 1:\n    {tour}\n")

print("One example of each, from the same seed:")
for name, operator in OPERATORS:
    random.seed(SEED)
    print(f"    {name:<10} -> {operator(tour)}")

rows = compare_operators(tour)
row_format = " {:<10} | {:>17} | {:>11} | {:>11} | {:>11} | {:>9}"
header = row_format.format("operator", "positions changed", "most",
                           "mean travel", "most", "unchanged")
print(f"\nEach operator applied {TRIALS} times to that individual:\n")
print(header)
print("-" * len(header))
for r in rows:
    print(row_format.format(r["name"], f"{r['mean_changed']:.3f}",
                            r["max_changed"], f"{r['mean_travel']:.3f}",
                            f"{r['max_travel']:.3f}",
                            f"{r['unchanged']} / {r['trials']}"))

illegal = sum(r["illegal"] for r in rows)
print(f"\nIllegal results across all {len(rows) * TRIALS} mutations: {illegal}.")
print("Every one of these operators only rearranges, so validity is structural")
print("rather than checked - there is no repair step anywhere in this file.")

# The two orderings, and their disagreement.
by_positions = sorted(rows, key=lambda r: -r["mean_changed"])
by_travel = sorted(rows, key=lambda r: -r["mean_travel"])
print("\nRanked by how much of the chromosome they disturb:")
print("    " + "  >  ".join(f"{r['name']} ({r['mean_changed']:.2f})"
                            for r in by_positions))
print("Ranked by how far they actually move a symbol:")
print("    " + "  >  ".join(f"{r['name']} ({r['mean_travel']:.2f})"
                            for r in by_travel))
if [r["name"] for r in by_positions] != [r["name"] for r in by_travel]:
    widest = by_positions[0]
    furthest = by_travel[0]
    print(f"\nThe two orders disagree. {widest['name'].capitalize()} touches the "
          f"most positions")
    print(f"({widest['mean_changed']:.3f} of {CHROMOSOME_LENGTH}) and moves "
          f"symbols {widest['mean_travel']:.3f} places on average;")
    print(f"{furthest['name']} touches {furthest['mean_changed']:.3f} and moves "
          f"them {furthest['mean_travel']:.3f}.")
    print("Reach on a permutation is not one quantity. 'How much is disturbed'")
    print("and 'how far anything gets' are different questions with different")
    print("answers, and a slide that shows only the first is misleading.")
else:
    print("\nThe two orders agree, so on this chromosome one number would do.")

# The exact statement behind the two shift and exchange rows.
exchange_travel = travel_over_all_pairs(tour, mutation_exchange)
shift_travel = travel_over_all_pairs(tour, mutation_shift)
identical = sum(1 for pair in exchange_travel
                if abs(exchange_travel[pair] - shift_travel[pair]) < 1e-12)
print(f"\nWhy exchange and shift share a mean travel, exactly and not by luck:")
print(f"over all {len(exchange_travel)} ordered pairs of distinct positions, the "
      f"two operators")
print(f"produce the same total travel in {identical} of them.")
print("Swapping positions i and j moves two symbols |i - j| places each;")
print("shifting from i to j moves one symbol |i - j| places and each of the")
print("|i - j| symbols in between exactly one place. Both come to 2|i - j|.")
print("Same distance covered, spread over 2 symbols or over |i - j| + 1 of them.")

# The one operator that can do nothing.
lazy = max(rows, key=lambda r: r["unchanged"])
print(f"\nAnd the last column: {lazy['name']} returned the individual unchanged "
      f"{lazy['unchanged']} times")
print(f"in {lazy['trials']} ({100 * lazy['unchanged'] / lazy['trials']:.1f}%). "
      "It draws two positions and randomises the")
print("block between them; when that block is short, a random order of it is")
print("often the order it already had. The most violent-sounding operator on")
print("the list is the only one with a real chance of doing nothing.")

print("\n--- both halves of the lesson, in one sentence ------------------------")
print("Reach is the single dial mutation has, whatever the encoding: sigma for")
print("a real gene, the size of the rearranged block for a permutation. And in")
print("both halves the widest setting was not the best one. Step 4's table")
print("reads 96 of 100 runs at sigma 1.0 against 82 at sigma 3.0; this table")
print("says the operator that disturbs the most positions is the one that")
print("moves symbols least. Bigger is a direction, not an improvement.")
print("What these operators do inside a search, on a real route, is Lesson 10.")

# The picture: the two measurements of reach, side by side, per operator.
names = [r["name"] for r in rows]
positions = np.arange(len(names))
fig, ax = plt.subplots(figsize=(8.5, 4.2))
ax.bar(positions - 0.2, [r["mean_changed"] for r in rows], width=0.4,
       color="tab:blue", label="positions changed (of 10)")
ax.bar(positions + 0.2, [r["mean_travel"] for r in rows], width=0.4,
       color="tab:orange", label="mean travel per symbol (positions)")
for index, r in enumerate(rows):
    if r["unchanged"]:
        ax.annotate(f"does nothing\n{100 * r['unchanged'] / r['trials']:.1f}% "
                    f"of the time",
                    (index, r["mean_changed"]), textcoords="offset points",
                    xytext=(0, 12), ha="center", fontsize=8, color="tab:red")
ax.set_xticks(positions)
ax.set_xticklabels(names)
ax.set_ylabel("positions")
ax.set_title(f"Two measurements of reach, {TRIALS} mutations of one "
             f"{CHROMOSOME_LENGTH}-symbol permutation")
ax.grid(True, axis="y", linestyle=":", alpha=0.5)
ax.legend(fontsize=9)
FIGURES.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURES / "permutation_mutation_02_rearrangements.png",
            dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved to {FIGURES}/permutation_mutation_02_rearrangements.png")

