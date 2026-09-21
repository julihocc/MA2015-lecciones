"""
Lesson 04 - Step 3: More cut points, and what the family can never do
=====================================================================
NEW IN THIS STEP: crossover_n_point(), crossover_uniform(), and the family table.

If one cut is good, are three better? Measured on this pair the answer is no,
and the reason is worth more than the answer. What separates the members of this
family is not how many cuts they make but how many of the 64 combinations of the
parents' genes they can actually reach: uniform crossover reaches all of them,
the n-point operators reach a handful.

There is also one thing none of them can do, and the `new genes` column says so
in every row.

CHANGES FROM offspring_02_one_point.py
Introduce them in this order:
    1. crossover_n_point()       more cuts, the same idea
    2. crossover_uniform()       a cut decision per gene, the limit of 'more cuts'
    3. the reach table           the whole family on step 1's instrument
    4. the pattern figure        every reachable inheritance pattern, as a grid

Run it:  python offspring_03_more_cuts.py

crossover_n_point() cuts n times and swaps alternate segments;
crossover_uniform() decides gene by gene. Reach is not monotone in cuts:
one-point 10, 2-point 12, 3-point 8, uniform 64 of 64. Uniform's price: 3.20%
of its children are a parent handed back.
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


def crossover_one_point(parent1: list, parent2: list) -> tuple[list, list]:
    """Cut both parents at the same point and swap the tails.

    The oldest crossover there is, and the one every textbook draws. Note what
    it can and cannot do: gene i of a child is gene i of one parent or gene i
    of the other, never anything else. The operator chooses WHICH parent, never
    WHAT value.

    Args:
        parent1, parent2: equal-length chromosomes.

    Returns:
        Two children that swapped tails at one cut in 1 .. L-1.

    Example:
        On these six-gene parents, 2000 draws reach 10 children;
        enumerating the 5 cut points reaches the same 10. 0.00% new genes.
    """
    point = random.randint(1, len(parent1) - 1)
    child1, child2 = list(parent1), list(parent2)
    child1[point:], child2[point:] = parent2[point:], parent1[point:]
    return child1, child2


# --- NEW (1) crossover_n_point() ----------------------------------------------
N_POINTS = (2, 3)                # how many cuts the n-point operator makes


def crossover_n_point(parent1: list, parent2: list, n: int) -> tuple[list, list]:
    """Cut n times and swap alternate segments.

    Same idea as one-point, more cuts. The cut positions are sampled from
    range(1, len - 1), which is Gridin's choice: the last gene can never start
    a segment. Keep that in mind when reading the reachable counts - they are
    counts for THIS implementation, not for the idea in the abstract.

    Args:
        parent1, parent2: equal-length chromosomes.
        n: number of cuts, sampled from range(1, len-1) as in Gridin.

    Returns:
        Two children with alternate segments swapped.

    Example:
        2-point reaches 12 children here, 3-point only 8 — fewer than
        one-point's 10 — because three cuts out of four interior
        positions admit only four segment patterns.
    """
    points = sorted(random.sample(range(1, len(parent1) - 1), n) + [0, len(parent1)])  # Gridin omits the last gene as a cut start
    child1, child2 = list(parent1), list(parent2)
    for i in range(n + 1):
        if i % 2 == 0:
            continue
        child1[points[i]:points[i + 1]] = parent2[points[i]:points[i + 1]]
        child2[points[i]:points[i + 1]] = parent1[points[i]:points[i + 1]]
    return child1, child2
# ------------------------------------------------------------------------------


# --- NEW (2) crossover_uniform() ----------------------------------------------
UNIFORM_RATE = 0.5               # chance of swapping each gene independently


def crossover_uniform(parent1: list, parent2: list,
                      rate: float = UNIFORM_RATE) -> tuple[list, list]:
    """Decide gene by gene which parent it comes from.

    This is the limit of "more cuts": a cut decision at every position. It is
    the only member of the family that can reach EVERY combination of the
    parents' genes, and the table below shows it doing exactly that.

    Args:
        parent1, parent2: equal-length chromosomes.
        rate: per-gene swap probability, 0.5 here.

    Returns:
        Two children; each gene independently came from one parent
        or the other.

    Example:
        Reaches 64 of 64 possible combinations; 3.20% of children are
        a parent handed back. On tours it is legal 0.95% of the time,
        and every one of those is a parent.
    """
    child1, child2 = list(parent1), list(parent2)
    for i in range(len(parent1)):
        if random.random() < rate:
            child1[i], child2[i] = parent2[i], parent1[i]
    return child1, child2
# ------------------------------------------------------------------------------


parent1, parent2 = create_parents()
print(f"parent 1   {show(parent1)}")
print(f"parent 2   {show(parent2)}")


# --- NEW (3) the reach table --------------------------------------------------
print(f"\nThe cut-and-swap family, {TRIALS} draws each:")
rows = [offspring_report("clone (no crossover)", crossover_clone,
                         parent1, parent2, is_legal_real),
        offspring_report("one-point", crossover_one_point,
                         parent1, parent2, is_legal_real)]
for n in N_POINTS:
    rows.append(offspring_report(f"{n}-point",
                                 lambda a, b, n=n: crossover_n_point(a, b, n),
                                 parent1, parent2, is_legal_real))
rows.append(offspring_report(f"uniform (rate {UNIFORM_RATE})", crossover_uniform,
                             parent1, parent2, is_legal_real))
print_report_header()
for row in rows:
    print_report_row(row)

by_label = {row["label"]: row for row in rows}
corners = 2 ** GENE_COUNT
one_point = by_label["one-point"]
uniform = by_label[f"uniform (rate {UNIFORM_RATE})"]
cuts = [by_label[f"{n}-point"] for n in N_POINTS]

print(f"\nThe `new genes` column is {max(row['new'] for row in rows):.2f}% in")
print("every row. No member of this family ever computes a gene value; the")
print("whole family does nothing but choose a parent per position. That is the")
print("family's definition, and the reason it is safe on any representation")
print("whose legality is a property of each gene on its own.")

print("\nThe `children reached` column is where they differ, and it does not do")
print("what the phrase 'more crossover points' suggests:")
for row in [one_point] + cuts + [uniform]:
    print(f"    {row['label']:24s} reaches {row['children']:3d} of {corners}")
worst = min([one_point] + cuts, key=lambda row: row["children"])
print(f"\nMore cuts is not more reach. {worst['label']} reaches only")
print(f"{worst['children']}, fewer than one-point's {one_point['children']},")
print(f"because n cut points sampled from the {GENE_COUNT - 2} interior")
print("positions admit fewer distinct segment patterns than a single free cut")
print("does. The count depends on the chromosome length, so do not memorise it")
print("- the lesson is that 'number of cuts' is not the quantity that matters.")

print(f"\nUniform crossover reaches {uniform['children']}, which is exactly")
print(f"{corners}: every combination of the parents' genes, the complete set.")
print("It is the limit of 'more cuts' - a cut decision at every position. The")
print(f"price is printed beside it: {uniform['clone']:.2f}% of its children are a")
print("parent handed straight back, because 'swap nothing' and 'swap everything'")
print(f"are 2 of the {corners} patterns it draws from. Keep that number; step 6")
print("turns it into the most surprising row in the lesson.")
# ------------------------------------------------------------------------------


# --- NEW (4) the pattern figure -----------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
for ax, row in zip(axes, [one_point, cuts[-1], uniform]):
    grid = [[0 if gene == parent1[i] else 1 for i, gene in enumerate(child)]
            for child in sorted(row["reachable"])]
    ax.imshow(grid, aspect="auto", cmap="coolwarm", vmin=0, vmax=1,
              interpolation="nearest")
    ax.set_title(f"{row['label']}\n{len(grid)} of {2 ** GENE_COUNT} patterns",
                 fontsize=10)
    ax.set_xticks(range(GENE_COUNT))
    ax.set_xticklabels(range(1, GENE_COUNT + 1))
    ax.set_xlabel("gene position")
    ax.set_yticks([])
axes[0].set_ylabel("one reachable child per row")
fig.suptitle("Which parent each gene came from: blue = parent 1, red = parent 2",
             fontsize=10)

FIGURES.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURES / "offspring_03_more_cuts.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved to {FIGURES}/offspring_03_more_cuts.png")
# ------------------------------------------------------------------------------

