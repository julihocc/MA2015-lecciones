"""
Lesson 04 - Step 4: Arithmetic on genes, and the first illegal children
=======================================================================
NEW IN THIS STEP: clamp(), crossover_blend(), crossover_linear().

Every operator so far chose between the parents' genes. These two compute new
ones - which is only possible because a gene here is a NUMBER, and numbers can
be averaged, interpolated and extrapolated. That single assumption about the
representation is what step 6 will take away.

Two consequences arrive together. The `new genes` column of the instrument
leaves zero for the first time, and the `legal` column leaves 100% for the first
time: a computed value has no reason to stay inside the box.

Lesson 01 step 4 already showed BLX-alpha and the inward collapse at alpha = 0.
This step does not re-derive it; it measures it, and puts a price on the cure.

CHANGES FROM offspring_03_more_cuts.py
Introduce them in this order:
    1. clamp()                   a computed gene has no reason to respect the box
    2. crossover_blend()         BLX-alpha: the first operator that computes a value
    3. crossover_linear()        the deterministic sibling - always the same two children
    4. the reach table           how far past the parents each operator actually goes
    5. the blend figure          where the children land, alpha 0 against alpha 0.5

Run it:  python offspring_04_blend.py

clamp() exists because arithmetic genes can leave [0, 10]. crossover_blend()
draws each child gene from an interval stretched by alpha; crossover_linear()
steps a fixed fraction along the parent line. Blend at alpha = 0.5 puts a new
value in 99.57% of gene slots and is only 50.08% legal until clamp repairs
them. Linear reaches exactly 2 children, 100.00% new genes.
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
    points = sorted(random.sample(range(1, len(parent1) - 1), n) + [0, len(parent1)])
    child1, child2 = list(parent1), list(parent2)
    for i in range(n + 1):
        if i % 2 == 0:
            continue
        child1[points[i]:points[i + 1]] = parent2[points[i]:points[i + 1]]
        child2[points[i]:points[i + 1]] = parent1[points[i]:points[i + 1]]
    return child1, child2


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


# --- NEW (1) clamp() ----------------------------------------------------------
def clamp(gene: float, low: float = GENE_MIN, high: float = GENE_MAX) -> float:
    """Arithmetic on genes can produce anything; the search space has edges.

    Lesson 01 introduced this for exactly the same reason. It is repeated here
    because from this step on the operators COMPUTE gene values instead of
    copying them, and a computed value has no reason to respect the box.

    Args:
        gene: raw gene (float).
        low, high: closed bounds of the search interval.

    Returns:
        gene projected onto [low, high].

    Example:
        At blend alpha = 0.5 only 50.08% of children are legal until
        clamp repairs them; with clamp, 100% are legal and 49.63% of
        genes still sit beyond the parents.
    """
    return max(low, min(high, gene))
# ------------------------------------------------------------------------------


# --- NEW (2) crossover_blend() ------------------------------------------------
BLEND_ALPHAS = (0.0, 0.5)        # step 4 runs the operator at both


def crossover_blend(parent1: Chromosome, parent2: Chromosome, alpha: float,
                    keep_inside: bool = True) -> tuple[Chromosome, Chromosome]:
    """Blend crossover (BLX-alpha): draw each child gene from a widened interval.

    For parental interval [m, M] of width d > 0, each raw gene is uniform on
    [m-alpha*d, M+alpha*d]; its exact outside-parent probability is
    2*alpha/(1+2*alpha). A finite run measures sampled reach, not support.

    For each position take the interval between the two parents' genes and
    stretch it by alpha of its own width at both ends, then draw uniformly
    inside it. With alpha = 0 the interval is exactly the segment between the
    parents and every child lies between them - Lesson 01 showed the population
    collapsing inwards for that reason. With alpha > 0 the interval reaches
    past both parents, which is what makes this the first operator in the
    lesson that can leave the box at all.

    `keep_inside=False` switches clamp() off, so that step 4 can show how often
    the operator would have left the box if nobody had repaired it.

    Args:
        parent1, parent2: real-valued chromosomes.
        alpha: BLX extension. 0.0 interpolates; 0.5 is this step's
            interesting setting.
        keep_inside: if False, clamp is skipped so the illegal share
            can be measured.

    Returns:
        Two children with computed gene values.

    Example:
        alpha = 0: 0.00% of genes beyond the parents, 100% legal.
        alpha = 0.5: 49.63% beyond, 50.08% legal; with clamp, 100%
        legal and still 49.63% beyond. 99.57% of gene slots are new.
    """
    child1, child2 = list(parent1), list(parent2)
    for i in range(len(parent1)):
        distance = abs(parent2[i] - parent1[i])
        low = min(parent1[i], parent2[i]) - alpha * distance
        high = max(parent1[i], parent2[i]) + alpha * distance
        child1[i] = round(low + random.random() * (high - low), 2)
        child2[i] = round(low + random.random() * (high - low), 2)
    if keep_inside:  # False leaves clamp off so the illegal share can be measured
        child1 = [clamp(gene) for gene in child1]
        child2 = [clamp(gene) for gene in child2]
    return child1, child2
# ------------------------------------------------------------------------------


# --- NEW (3) crossover_linear() -----------------------------------------------
LINEAR_ALPHA = 0.3               # how far along the parent-to-parent line to step


def crossover_linear(parent1: Chromosome, parent2: Chromosome,
                     alpha: float = LINEAR_ALPHA) -> tuple[Chromosome, Chromosome]:
    """Step a fixed fraction alpha along the line joining the two parents.

    The deterministic sibling of blend: no random draw anywhere, so the same
    pair of parents always yields the same pair of children. It computes new
    gene values - and it can only ever compute these two.

    Args:
        parent1, parent2: real-valued chromosomes.
        alpha: fraction along the parent-to-parent line, 0.3 here.

    Returns:
        The same two children every time this pair is handed in.

    Example:
        100.00% new genes and exactly 2 reachable children. On a
        tour: legal 0.00%.
    """
    child1, child2 = list(parent1), list(parent2)
    for i in range(len(parent1)):
        difference = parent2[i] - parent1[i]
        child1[i] = clamp(round(parent1[i] + alpha * difference, 2))
        child2[i] = clamp(round(parent2[i] - alpha * difference, 2))
    return child1, child2
# ------------------------------------------------------------------------------


parent1, parent2 = create_parents()
print(f"parent 1   {show(parent1)}")
print(f"parent 2   {show(parent2)}")


# --- NEW (4) the reach table --------------------------------------------------
def share_outside(operator: Operator) -> float:
    """% of gene slots holding a value outside the interval spanned by the parents.

    The instrument of step 1 counts values that are NEW. This counts values
    that are BEYOND - the difference between interpolating between the parents
    and extrapolating past them, which is the whole argument for alpha > 0.

    Args:
        operator: a real-valued crossover.

    Returns:
        % of gene slots holding a value outside the interval spanned
        by the two parents — interpolating vs extrapolating.

    Example:
        Blend alpha = 0 gives 0.00%; alpha = 0.5 gives 49.63%.
    """
    random.seed(SEED)
    outside = slots = 0
    for _ in range(TRIALS):
        for child in operator(parent1, parent2):
            for index, gene in enumerate(child):
                slots += 1
                low = min(parent1[index], parent2[index])
                high = max(parent1[index], parent2[index])
                if gene < low or gene > high:
                    outside += 1
    return 100 * outside / slots


operators = [
    ("clone (no crossover)", crossover_clone),
    (f"uniform (rate {UNIFORM_RATE})", crossover_uniform),
]
for alpha in BLEND_ALPHAS:
    operators.append((f"blend alpha={alpha}, no clamp",
                      lambda a, b, alpha=alpha: crossover_blend(a, b, alpha,
                                                                keep_inside=False)))
operators.append((f"blend alpha={BLEND_ALPHAS[-1]} + clamp",
                  lambda a, b: crossover_blend(a, b, BLEND_ALPHAS[-1])))
operators.append((f"linear alpha={LINEAR_ALPHA}", crossover_linear))

print(f"\nThe arithmetic operators, {TRIALS} draws each:")
rows = [offspring_report(label, operator, parent1, parent2, is_legal_real)
        for label, operator in operators]
for row, (label, operator) in zip(rows, operators):
    row["outside"] = share_outside(operator)
print_report_header()
for row in rows:
    print_report_row(row)

print(f"\n{'operator':28s} {'genes beyond the parents':>26}")
for row in rows:
    print(f"{row['label']:28s} {row['outside']:25.2f}%")

by_label = {row["label"]: row for row in rows}
flat = by_label[f"blend alpha={BLEND_ALPHAS[0]}, no clamp"]
wide = by_label[f"blend alpha={BLEND_ALPHAS[-1]}, no clamp"]
repaired = by_label[f"blend alpha={BLEND_ALPHAS[-1]} + clamp"]
linear = by_label[f"linear alpha={LINEAR_ALPHA}"]

print(f"\nThe `new genes` column has moved off zero for the first time:")
print(f"{wide['new']:.2f}% of gene slots hold a value neither parent had. Not")
print("100%, because a draw that rounds to two decimals can land exactly on a")
print("parent's value; but for practical purposes every gene is computed, not")
print(f"copied, and the operator reached {wide['children']} distinct children in")
print(f"{TRIALS} draws - that is 2 per draw, all of them different. The")
print("reachable set is no longer something you can enumerate.")

print(f"\nalpha = {BLEND_ALPHAS[0]} and alpha = {BLEND_ALPHAS[-1]} differ in one")
print("number and in one behaviour:")
print(f"    alpha = {BLEND_ALPHAS[0]}:  {flat['outside']:5.2f}% of genes beyond "
      f"the parents, {flat['legal']:6.2f}% legal")
print(f"    alpha = {BLEND_ALPHAS[-1]}:  {wide['outside']:5.2f}% of genes beyond "
      f"the parents, {wide['legal']:6.2f}% legal")
print(f"With alpha = {BLEND_ALPHAS[0]} the children are always between the")
print("parents. That is the collapse Lesson 01 showed: a population bred that")
print("way can only shrink towards its own middle, and every child is legal")
print("precisely because it never goes anywhere new.")
print(f"With alpha = {BLEND_ALPHAS[-1]} the operator finally explores - and")
print(f"immediately breaks the box: only {wide['legal']:.2f}% of its children")
print("are legal. Exploration and legality are the same trade here, which is")
print(f"why clamp() exists. The repaired row is legal {repaired['legal']:.2f}% of")
print(f"the time and still puts {repaired['outside']:.2f}% of genes beyond the")
print("parents, so the repair costs almost none of the reach.")

print(f"\nLinear crossover is the odd one out: {linear['new']:.2f}% new genes,")
print(f"none of them beyond the parents ({linear['outside']:.2f}%), and exactly")
print(f"{linear['children']} children reachable. It computes values - and it can")
print("only ever compute these two, because there is no random draw in it at")
print("all. New is not the same as varied.")
# ------------------------------------------------------------------------------


# --- NEW (5) the blend figure -------------------------------------------------
position = max(range(GENE_COUNT), key=lambda i: abs(parent1[i] - parent2[i]))
fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for ax, alpha in zip(axes, BLEND_ALPHAS):
    random.seed(SEED)
    values = []
    for _ in range(TRIALS):
        for child in crossover_blend(parent1, parent2, alpha, keep_inside=False):
            values.append(child[position])
    ax.hist(values, bins=40, color="tab:green", alpha=0.75)
    ax.axvline(parent1[position], color="tab:blue", linewidth=2, label="parent 1")
    ax.axvline(parent2[position], color="tab:red", linewidth=2, label="parent 2")
    ax.axvline(GENE_MIN, color="black", linestyle="--", linewidth=1,
               label="wall of the box")
    outside = sum(1 for v in values
                  if v < min(parent1[position], parent2[position])
                  or v > max(parent1[position], parent2[position]))
    ax.set_title(f"alpha = {alpha}: {100 * outside / len(values):.1f}% beyond "
                 f"the parents", fontsize=10)
    ax.set_xlabel(f"value drawn for gene {position + 1}")
    ax.grid(True, linestyle=":", alpha=0.5)
axes[0].set_ylabel("draws")
axes[0].legend(fontsize=8)

FIGURES.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURES / "offspring_04_blend.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved to {FIGURES}/offspring_04_blend.png")
# ------------------------------------------------------------------------------

