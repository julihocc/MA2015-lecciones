"""
Lesson 04 - Step 5: Letting the operator look at what it made
=============================================================
NEW IN THIS STEP: objective(), crossover_fitness_driven().

Every operator so far was blind: it produced two children and handed them over
without ever asking whether they were any good. Gridin's fitness-driven variant
adds one line - rank the two parents and the two children together, return the
best two - and with it the guarantee that crossover can never make the pair
worse.

The guarantee is real. This step measures what it costs, and the cost is not
subtle.

(Gridin's own listing has a bug here: it builds both children from the first
child's genes, so its second child is a duplicate. Fixed in the code below.)

CHANGES FROM offspring_04_blend.py
Introduce them in this order:
    1. objective()               a chromosome finally has a value, not only a shape
    2. crossover_fitness_driven() make two children, keep the best two of the four
    3. the survivors             what the rule buys, and what it quietly costs
    4. the fitness figure        what each rule hands back, as a distribution

Run it:  python offspring_05_fitness_driven.py

objective() scores a six-gene chromosome as the mean of Lesson 01's landscape.
crossover_fitness_driven() blends, then keeps the best two of the four
chromosomes involved. Blind blend is worse than its parents in 41.70% of
draws; the fitness-driven rule is worse in 0.00%, and pays for it: 16.50% of
draws return the parents unchanged.
"""
import math
import random
import statistics
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
    if keep_inside:
        child1 = [clamp(gene) for gene in child1]
        child2 = [clamp(gene) for gene in child2]
    return child1, child2


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


# --- NEW (1) objective() ------------------------------------------------------
def objective(chromosome: Chromosome) -> float:
    """What a real-valued chromosome is worth. Higher is better.

    Lesson 01's landscape, sin(x) - 0.2|x|, averaged over the six genes, so the
    number stays on the scale that lesson used and a chromosome finally has a
    value rather than only a shape.

    Args:
        chromosome: six real genes in [0, 10].

    Returns:
        Mean of sin(g) - 0.2*|g| over the genes. Larger is better.

    Example:
        The two parents of this lesson score as printed; fitness-driven
        crossover is worse than that parental mean in 0.00% of draws.
    """
    return sum(math.sin(g) - 0.2 * abs(g) for g in chromosome) / len(chromosome)
# ------------------------------------------------------------------------------


# --- NEW (2) crossover_fitness_driven() ---------------------------------------
def crossover_fitness_driven(parent1: Chromosome, parent2: Chromosome,
                             alpha: float) -> tuple[Chromosome, Chromosome]:
    """Blend, then keep the best two of the four chromosomes involved.

    Gridin's fitness-driven crossover. The operator stops being blind: it looks
    at what it made and refuses to hand back anything worse than what it was
    given. That guarantee is real, and step 5 measures what it costs.

    (The book's own listing builds both children from the first child's genes,
    so its `child2` is a duplicate of `child1`. Fixed here.)

    Args:
        parent1, parent2: real-valued chromosomes.
        alpha: blend extension, 0.5 here.

    Returns:
        The best two of {parent1, parent2, child1, child2} by objective.

    Example:
        Blind blend is worse than the parents in 41.70% of draws; this
        rule is worse in 0.00%, returns the parents in 16.50%, and only
        56.10% of what it returns is a child.
    """
    child1, child2 = crossover_blend(parent1, parent2, alpha)
    ranked = sorted([list(parent1), list(parent2), child1, child2],
                    key=objective, reverse=True)
    return ranked[0], ranked[1]
# ------------------------------------------------------------------------------


parent1, parent2 = create_parents()
print(f"parent 1   {show(parent1)}")
print(f"parent 2   {show(parent2)}")


# --- NEW (3) the survivors ----------------------------------------------------
print(f"\nThe parents now have a value: f(parent 1) = {objective(parent1):+.4f}, "
      f"f(parent 2) = {objective(parent2):+.4f}")
PARENTS_MEAN = (objective(parent1) + objective(parent2)) / 2
print(f"Their mean is {PARENTS_MEAN:+.4f}. That is the number the operator has")
print("to beat.")

alpha = BLEND_ALPHAS[-1]


def pair_quality(operator: Operator) -> dict:
    """Mean value of the pair an operator returns, and how often it returns
    the parents unchanged.

    Args:
        operator: a crossover on the real-valued parents.

    Returns:
        Mean pair-fitness, how often the parents come back unchanged,
        and how often a returned chromosome is a true child.

    Example:
        The fitness-driven rule's 16.50% unchanged / 56.10% children
        split is this function's output.
    """
    random.seed(SEED)
    values, unchanged, children_kept, kept = [], 0, 0, 0
    for _ in range(TRIALS):
        first, second = operator(parent1, parent2)
        values.append((objective(first) + objective(second)) / 2)
        returned = {tuple(first), tuple(second)}
        if returned == {tuple(parent1), tuple(parent2)}:
            unchanged += 1
        for survivor in (first, second):
            kept += 1
            if list(survivor) != parent1 and list(survivor) != parent2:
                children_kept += 1
    worse = sum(1 for value in values if value < PARENTS_MEAN)
    return {"mean": statistics.mean(values),
            "unchanged": 100 * unchanged / TRIALS,
            "children": 100 * children_kept / kept,
            "worse": 100 * worse / TRIALS}


blind = pair_quality(lambda a, b: crossover_blend(a, b, alpha))
driven = pair_quality(lambda a, b: crossover_fitness_driven(a, b, alpha))

print(f"\n{TRIALS} draws of each, at alpha = {alpha}:")
print(f"{'operator':24s} {'mean f of':>11} {'draws worse':>12} {'pair = the':>11} "
      f"{'survivors':>10}")
print(f"{'':24s} {'the pair':>11} {'than the':>12} {'parents':>11} {'that are':>10}")
print(f"{'':24s} {'returned':>11} {'parents':>12} {'unchanged':>11} {'children':>10}")
print(f"{'the parents themselves':24s} {PARENTS_MEAN:+11.4f} {0.0:11.2f}% "
      f"{100.0:10.2f}% {0.0:9.2f}%")
print(f"{'blend, blind':24s} {blind['mean']:+11.4f} {blind['worse']:11.2f}% "
      f"{blind['unchanged']:10.2f}% {blind['children']:9.2f}%")
print(f"{'blend, fitness-driven':24s} {driven['mean']:+11.4f} {driven['worse']:11.2f}% "
      f"{driven['unchanged']:10.2f}% {driven['children']:9.2f}%")

print(f"\nBlind blend happens to average {blind['mean']:+.4f} here, above the")
print(f"{PARENTS_MEAN:+.4f} it was given - but that is this pair on this")
print("landscape, not a property of the operator, and the next column says so:")
print(f"in {blind['worse']:.2f}% of draws the pair it returns is worse than the")
print("pair it was handed. Blind crossover offers no guarantee whatsoever.")

print(f"\nThe fitness-driven rule averages {driven['mean']:+.4f} and is worse")
print(f"than the parents in {driven['worse']:.2f}% of draws - it cannot be,")
print("because the parents are two of the four candidates it ranks. That")
print("guarantee is exactly what it costs:")
print(f"    in {driven['unchanged']:.2f}% of draws it hands back the two parents")
print("    unchanged - the operator ran and the population did not move")
print(f"    only {driven['children']:.2f}% of the chromosomes it returns are")
print(f"    children at all, against {blind['children']:.2f}% for blind blend")
print("\nSo the rule converts exploration into safety at a measurable rate. On a")
print("hard landscape that is a bargain; on a run that has already converged it")
print("is a machine for standing still. Lesson 03 measured this trade for")
print("selection; this is the same trade, moved inside the operator.")
# ------------------------------------------------------------------------------


# --- NEW (4) the fitness figure -----------------------------------------------
random.seed(SEED)
blind_values, driven_values = [], []
for _ in range(TRIALS):
    first, second = crossover_blend(parent1, parent2, alpha)
    blind_values.extend([objective(first), objective(second)])
random.seed(SEED)
for _ in range(TRIALS):
    first, second = crossover_fitness_driven(parent1, parent2, alpha)
    driven_values.extend([objective(first), objective(second)])

fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(blind_values, bins=50, alpha=0.6, color="tab:green",
        label="blind blend: whatever came out")
ax.hist(driven_values, bins=50, alpha=0.6, color="tab:purple",
        label="fitness-driven: the best two of four")
ax.axvline(objective(parent1), color="tab:blue", linewidth=2, label="parent 1")
ax.axvline(objective(parent2), color="tab:red", linewidth=2, label="parent 2")
ax.set_xlabel("f of a returned chromosome")
ax.set_ylabel("count")
ax.set_title("What each rule hands back, over "
             f"{TRIALS} draws of the same two parents")
ax.legend(fontsize=8)
ax.grid(True, linestyle=":", alpha=0.5)

FIGURES.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURES / "offspring_05_fitness_driven.png", dpi=150,
            bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved to {FIGURES}/offspring_05_fitness_driven.png")
# ------------------------------------------------------------------------------

