"""
Lesson 04 - Step 1: Two parents, and how to measure what an operator does to them
=================================================================================
Crossover is the operator that takes two chromosomes and returns two new ones.
Nothing else in a genetic algorithm creates a combination that was not there
before, so the whole lesson turns on one question: GIVEN THESE TWO PARENTS,
WHAT IS THE SET OF CHILDREN AN OPERATOR CAN PRODUCE?

This step builds nothing but the instrument that answers it, and applies it to
the cheapest possible operator - the one that hands the parents straight back.
Every one of the seven operators in steps 2 to 7 is measured against that row.

The parents are Gridin's own pair for chapter 4 (SEED = 3, six genes drawn
uniformly from the box), so the numbers here line up with the book's.

Run it:  python offspring_01_two_parents.py

create_parents() rebuilds Gridin's pair (SEED = 3): [2.38 5.44 3.70 6.04 6.26
0.66] and [0.13 8.37 2.59 2.34 9.96 4.70]. offspring_report() fires an
operator 2000 times on that pair. crossover_clone() is the zero row: 2
children reached, 0.00% new genes, 100.00% clones, 100.00% legal. Choosing
freely between the parents allows 2^6 = 64 chromosomes.
"""
import random
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt


SEED = 3                         # Gridin's seed for chapter 4; every measurement resets to it
GENE_COUNT = 6                   # a chromosome is six real numbers
GENE_MIN, GENE_MAX = 0.0, 10.0   # the box a gene has to stay inside
TRIALS = 2000                    # draws per measurement, for stable percentages
FIGURES = Path(__file__).resolve().parent.parent / "figures"

Chromosome = list[float]
Operator = Callable[[list, list], tuple[list, list]]
Legality = Callable[[list], bool]


def create_parents() -> tuple[Chromosome, Chromosome]:
    """The two parents every operator in this lesson is handed.

    They are the parents of Gridin's chapter 4 - same seed, same six genes -
    so anything printed here can be compared with the book directly.
    random.seed() is called inside the function rather than at module level so
    that every measurement starts from the same pair, whatever ran before it.

    Returns:
        Gridin's two six-gene parents, SEED = 3, rounded to 2 decimals:
        [2.38 5.44 3.70 6.04 6.26 0.66] and
        [0.13 8.37 2.59 2.34 9.96 4.70].
    """
    random.seed(SEED)
    p1 = [round(random.uniform(GENE_MIN, GENE_MAX), 2) for _ in range(GENE_COUNT)]
    p2 = [round(random.uniform(GENE_MIN, GENE_MAX), 2) for _ in range(GENE_COUNT)]
    return p1, p2


def is_legal_real(chromosome: Chromosome) -> bool:
    """A real-valued chromosome is legal when every gene is inside the box.

    That is the entire constraint on this representation, and it is what lets
    the tables below ever call a child *illegal*: an operator that leaves the
    box has produced something the problem cannot evaluate.

    Args:
        chromosome: a list of real genes.

    Returns:
        True iff every gene is inside [GENE_MIN, GENE_MAX] = [0, 10].

    Example:
        Blend at alpha = 0.5 is only 50.08% legal until clamp repairs it.
    """
    return all(GENE_MIN <= gene <= GENE_MAX for gene in chromosome)


def show(chromosome) -> str:
    """One chromosome on one line, so parents and children line up in columns.

    Args:
        chromosome: reals or integer stops.

    Returns:
        A single-line string so parents and children line up in columns.
    """
    parts = [f"{gene:6.2f}" if isinstance(gene, float) else f"{gene:>6d}"
             for gene in chromosome]
    return "[" + " ".join(parts) + "]"


def crossover_clone(parent1: list, parent2: list) -> tuple[list, list]:
    """The zero-reach baseline: the children ARE the parents.

    Nobody would use this. It is the yardstick: every row of every table below
    is interesting exactly to the extent that it differs from this one.

    Args:
        parent1, parent2: the two chromosomes.

    Returns:
        Shallow copies of the parents. The zero-reach baseline.

    Example:
        2 children reached, 0.00% new genes, 100.00% clones, 100.00% legal.
    """
    return list(parent1), list(parent2)


def offspring_report(label: str, operator: Operator,
                     parent1: list, parent2: list,
                     is_legal: Legality, trials: int = TRIALS) -> dict:
    """Run one operator `trials` times on the same pair and describe the cloud.

    Four numbers, and the whole lesson is read off them:

      children  how many DIFFERENT chromosomes came out. For the cut-and-swap
                operators this is the operator's entire reachable set, small
                enough to be listed; for the arithmetic ones it is just the
                number of draws, which is the point.
      new       % of gene slots holding a value NEITHER parent had in that
                slot - values the operator computed rather than copied.
      clone     % of children that are an exact copy of a parent: the operator
                ran and nothing happened.
      legal     % of children the problem can actually read.

    Always reseeded from SEED, so the rows of a table are comparable by
    construction rather than by hope.

    Args:
        label: row name.
        operator: (p1, p2) -> (c1, c2).
        parent1, parent2: the fixed pair.
        is_legal: is_legal_real or is_legal_tour.
        trials: 2000, reseeded from SEED = 3.

    Returns:
        children reached, % new genes, % clones, % legal, and the set
        of reachable chromosomes.

    Example:
        One-point reaches 10 children, 0.00% new; blend at alpha = 0.5
        puts a new value in 99.57% of gene slots.
    """
    random.seed(SEED)
    seen: set[tuple] = set()
    slots = new_values = clones = legal = produced = 0
    for _ in range(trials):
        for child in operator(parent1, parent2):
            produced += 1
            seen.add(tuple(child))
            if list(child) == list(parent1) or list(child) == list(parent2):
                clones += 1
            if is_legal(child):
                legal += 1
            for index, gene in enumerate(child):
                slots += 1
                if gene != parent1[index] and gene != parent2[index]:
                    new_values += 1
    return {"label": label, "children": len(seen), "new": 100 * new_values / slots,
            "clone": 100 * clones / produced, "legal": 100 * legal / produced,
            "reachable": seen}


def print_report_header() -> None:
    """One header for every comparison table in steps 1 to 7.

    Returns:
        None. Prints the shared header of every comparison table
        in steps 1 to 7.
    """
    print(f"{'operator':28s} {'children':>8} {'new':>7} {'clone':>7} {'legal':>7}")
    print(f"{'':28s} {'reached':>8} {'genes':>7} {'of a':>7} {'':>7}")
    print(f"{'':28s} {'':>8} {'':>7} {'parent':>7} {'':>7}")


def print_report_row(row: dict) -> None:
    """Print one measured row of the comparison table.

    Args:
        row: a dict from offspring_report.

    Returns:
        None. Prints one measured line.
    """
    print(f"{row['label']:28s} {row['children']:8d} {row['new']:6.2f}% "
          f"{row['clone']:6.2f}% {row['legal']:6.2f}%")


parent1, parent2 = create_parents()

print("The two parents, gene by gene:")
print(f"    parent 1   {show(parent1)}")
print(f"    parent 2   {show(parent2)}")
print(f"    distance   {show([round(abs(a - b), 2) for a, b in zip(parent1, parent2)])}")
print("\nEvery operator in this lesson is handed exactly these two chromosomes")
print("and nothing else. So there is one question to ask of each of them: what")
print("is the set of children it can produce from this pair?")

print(f"\nThe baseline, {TRIALS} draws:")
baseline = offspring_report("clone (no crossover)", crossover_clone,
                            parent1, parent2, is_legal_real)
print_report_header()
print_report_row(baseline)
print(f"\nRead the row: {baseline['children']} different chromosomes ever came")
print(f"out, {baseline['new']:.2f}% of gene slots held a value neither parent")
print(f"had there, {baseline['clone']:.2f}% of the children were a parent, and")
print(f"{baseline['legal']:.2f}% of them were legal. That is what ZERO looks")
print("like on this instrument. Every row in steps 2 to 7 is a departure from it.")

corners = 2 ** GENE_COUNT
print(f"\nOne number to keep in mind. A child built only by CHOOSING, gene by")
print(f"gene, between the two parents can be one of {corners} chromosomes -")
print(f"2 choices at each of {GENE_COUNT} positions. Steps 2 and 3 ask how much")
print("of that set each cut-and-swap operator can actually reach. Step 4 asks")
print("what happens when an operator is allowed to compute a value instead of")
print("choosing one, and the answer is a different kind of number entirely.")


fig, ax = plt.subplots(figsize=(9, 4))
positions = range(1, GENE_COUNT + 1)
ax.fill_between(positions, parent1, parent2, color="tab:blue", alpha=0.15,
                label="between the parents")
ax.plot(positions, parent1, "o-", color="tab:blue", label="parent 1")
ax.plot(positions, parent2, "s-", color="tab:red", label="parent 2")
ax.axhline(GENE_MIN, color="grey", linewidth=0.8, linestyle="--")
ax.axhline(GENE_MAX, color="grey", linewidth=0.8, linestyle="--")
ax.set_xticks(list(positions))
ax.set_xlabel("gene position")
ax.set_ylabel("gene value")
ax.set_ylim(GENE_MIN - 1.5, GENE_MAX + 1.5)
ax.set_title("The two parents, and the region between them")
ax.legend(loc="lower right", fontsize=9)
ax.grid(True, linestyle=":", alpha=0.5)

FIGURES.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURES / "offspring_01_two_parents.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved to {FIGURES}/offspring_01_two_parents.png")

