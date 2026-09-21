"""
Lesson 04 - Step 2: One-point crossover, and the size of its reach
==================================================================
NEW IN THIS STEP: crossover_one_point(), and the enumeration of its children.

The textbook operator: cut both parents at the same point, swap the tails. It is
worth starting here not because it is the best but because its reachable set is
small enough to write down in full - which is the only way to see what the
operator can and cannot do.

CHANGES FROM offspring_01_two_parents.py
Introduce them in this order:
    1. crossover_one_point()     cut once, swap the tails - the oldest operator there is
    2. the reachable set         list every child this operator can produce from this pair
    3. the cut figure            the five children drawn over the parents they came from

Run it:  python offspring_02_one_point.py

crossover_one_point() cuts both parents at the same index and swaps the tails.
2000 draws produce 10 distinct children; enumerating the 5 cut points produces
the same 10. 0.00% new genes: the operator chooses which parent, never what
value.
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


# --- NEW (1) crossover_one_point() --------------------------------------------
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
# ------------------------------------------------------------------------------


parent1, parent2 = create_parents()
print(f"parent 1   {show(parent1)}")
print(f"parent 2   {show(parent2)}")


# --- NEW (2) the reachable set ------------------------------------------------
print(f"\nOne-point crossover, {TRIALS} draws:")
baseline = offspring_report("clone (no crossover)", crossover_clone,
                            parent1, parent2, is_legal_real)
one_point = offspring_report("one-point", crossover_one_point,
                             parent1, parent2, is_legal_real)
print_report_header()
for row in (baseline, one_point):
    print_report_row(row)

# The reachable set is small enough to write out by hand, so write it out: the
# cut point is an integer from 1 to GENE_COUNT - 1, and nothing else is random.
enumerated = set()
print(f"\nThere are only {GENE_COUNT - 1} places to cut, so the whole reachable")
print("set can be listed:")
for point in range(1, GENE_COUNT):
    kid1 = parent1[:point] + parent2[point:]
    kid2 = parent2[:point] + parent1[point:]
    enumerated.update({tuple(kid1), tuple(kid2)})
    print(f"    cut after gene {point}:  {show(kid1)}   {show(kid2)}")

print(f"\n{TRIALS} draws produced {one_point['children']} distinct children;")
print(f"enumerating the cut points gives {len(enumerated)}. Same set: "
      f"{'yes' if enumerated == one_point['reachable'] else 'NO'}.")
print(f"So this operator's entire reach is {len(enumerated)} chromosomes, out")
print(f"of the {2 ** GENE_COUNT} that choosing freely between the parents allows.")
print(f"\nAnd {one_point['new']:.2f}% of gene slots held a value neither parent")
print("had in that slot. That is the sentence to take away: one-point crossover")
print("RECOMBINES, it does not INVENT. Gene i of a child is gene i of one parent")
print("or gene i of the other. The operator chooses which parent, never what")
print(f"value - which is also why every child is legal ({one_point['legal']:.2f}%)")
print("although the operator knows nothing about the box.")
# ------------------------------------------------------------------------------


# --- NEW (3) the cut figure ---------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 4))
positions = range(1, GENE_COUNT + 1)
ax.fill_between(positions, parent1, parent2, color="tab:blue", alpha=0.12)
for point in range(1, GENE_COUNT):
    kid = parent1[:point] + parent2[point:]
    ax.plot(positions, kid, "-", color="tab:green", alpha=0.8, linewidth=1.0)
    ax.annotate(f"cut {point}", (point + 0.5, kid[point]), fontsize=7,
                color="tab:green", textcoords="offset points", xytext=(0, 6))
ax.plot(positions, parent1, "o-", color="tab:blue", linewidth=2, label="parent 1")
ax.plot(positions, parent2, "s-", color="tab:red", linewidth=2, label="parent 2")
ax.plot([], [], "-", color="tab:green",
        label="the five children that start as parent 1")
ax.set_xticks(list(positions))
ax.set_xlabel("gene position")
ax.set_ylabel("gene value")
ax.set_title("One-point crossover: every child follows one parent, then the other")
ax.legend(loc="lower right", fontsize=8)
ax.grid(True, linestyle=":", alpha=0.5)

FIGURES.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURES / "offspring_02_one_point.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved to {FIGURES}/offspring_02_one_point.png")

# Same band, second figure: the mechanism the profile plot does not show.
# One cut after gene 3, two parent bars, two child bars. The existing figure
# is kept; this one is the textbook drawing of the operator.
SCHEMATIC_CUT = 3
child_from_p1 = parent1[:SCHEMATIC_CUT] + parent2[SCHEMATIC_CUT:]
child_from_p2 = parent2[:SCHEMATIC_CUT] + parent1[SCHEMATIC_CUT:]
rows = (
    ("parent 1", parent1, "tab:blue"),
    ("parent 2", parent2, "tab:red"),
    ("child 1  (cut after gene 3)", child_from_p1, "tab:green"),
    ("child 2  (cut after gene 3)", child_from_p2, "tab:olive"),
)
fig2, axes = plt.subplots(4, 1, figsize=(9, 4.8), sharex=True)
for axis, (label, genes, colour) in zip(axes, rows):
    for index, gene in enumerate(genes):
        from_parent1 = gene == parent1[index]
        from_parent2 = gene == parent2[index]
        face = "tab:blue" if from_parent1 and not from_parent2 else (
            "tab:red" if from_parent2 and not from_parent1 else colour
        )
        axis.bar(index + 1, 1, width=0.9, color=face, edgecolor="black", linewidth=0.6)
        axis.text(index + 1, 0.5, f"{gene:.2f}", ha="center", va="center",
                  fontsize=8, color="white", fontweight="bold")
    axis.axvline(SCHEMATIC_CUT + 0.5, color="black", linestyle="--", linewidth=1.2)
    axis.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8)
    axis.set_yticks([])
    axis.set_ylim(0, 1)
    axis.set_xlim(0.4, GENE_COUNT + 0.6)
axes[-1].set_xticks(list(range(1, GENE_COUNT + 1)))
axes[-1].set_xlabel("gene position")
axes[0].set_title("One-point crossover: cut after gene 3, swap the tails")
fig2.tight_layout()
fig2.savefig(FIGURES / "offspring_02_one_point_schematic.png", dpi=150,
             bbox_inches="tight")
plt.close(fig2)
print(f"Schematic saved to {FIGURES}/offspring_02_one_point_schematic.png")
# ------------------------------------------------------------------------------

