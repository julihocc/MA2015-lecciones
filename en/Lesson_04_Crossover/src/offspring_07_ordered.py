"""
Lesson 04 - Step 7: Ordered crossover, and the divide read in both directions
=============================================================================
NEW IN THIS STEP: crossover_order(), and the comparison the lesson was built for.

Ordered crossover (OX1) is the answer to step 6: it never swaps gene values
position by position, so a permutation goes in and a permutation comes out, by
construction rather than by luck. This step checks that it is legal, checks what
it actually inherits, and then does the thing that closes the lesson - hands OX1
the real-valued parents of step 1 and watches what happens.

It does not crash. That is the finding.

Note on Gridin's listing: his order.py indexes an array BY GENE VALUE
(`spaces1[t2[i]]`), so on real-valued genes it raises TypeError instead. The
implementation below is the usual formulation of OX1 and fails silently, which
is the more instructive failure and the more dangerous one.

CHANGES FROM offspring_06_the_divide.py
Introduce them in this order:
    1. crossover_order()         OX1: copy a swath, fill the rest in the other parent's order
    2. the adjacency test        what OX1 actually inherits, against a random route
    3. the other direction       OX1 handed the real-valued parents of step 1
    4. the whole comparison      seven operators, two representations, one table
    5. the divide figure         the useful column of that table, side by side

Run it:  python offspring_07_ordered.py

crossover_order() (OX1) copies a swath and fills the rest in the other
parent's order, so a child of two routes is a route by construction: legal
100.00%, keeping 7.17 of 9 parent legs. Handed the real-valued parents it does
not crash: it reaches 2 children, 100.00% of them a parent. Operators useful
on both representations: 0.
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


STOP_COUNT = 9                   # nine delivery stops, as in Gridin's order.py
STOP_XY = [(4.5, 5.6), (9.2, 4.7), (5.1, 5.9), (1.8, 5.1), (6.3, 7.9),
           (0.9, 3.0), (0.9, 8.1), (6.9, 0.4), (9.8, 9.6)]

Tour = list[int]


def create_tours() -> tuple[Tour, Tour]:
    """Two routes through the nine stops - the parents of the second half.

    A route is a chromosome too: nine genes, one per position in the round.
    What has changed is not the length or the type but the CONSTRAINT: the nine
    genes have to be the nine stops, each exactly once.

    Returns:
        Two random routes through the nine stops, SEED = 3, the size
        Gridin uses in order.py.
    """
    random.seed(SEED)
    return (random.sample(range(1, STOP_COUNT + 1), STOP_COUNT),
            random.sample(range(1, STOP_COUNT + 1), STOP_COUNT))


def tour_length(tour: Tour) -> float:
    """Total distance of a closed round through the stops. Lower is better.

    Args:
        tour: a permutation of 1..9.

    Returns:
        Closed-round Euclidean length. Lower is better.

    Example:
        OX1 children average 52.95 against 51.23 for 2000 random
        routes: inheriting 7.17 parent legs is visible, inheriting
        quality is not.
    """
    return sum(math.dist(STOP_XY[tour[i] - 1], STOP_XY[tour[(i + 1) % len(tour)] - 1])
               for i in range(len(tour)))


def is_legal_tour(chromosome) -> bool:
    """A route is legal when it visits each of the nine stops exactly once.

    Compare this with is_legal_real(): there, legality was a property of each
    gene on its own. Here it is a property of the chromosome AS A WHOLE, and no
    operator that treats genes independently can be trusted to preserve it.

    Args:
        chromosome: claimed to be a route.

    Returns:
        True iff it is a permutation of 1..9.

    Example:
        8 of 8 one-point cut points visit one stop twice and another
        never. Legal children of one-point / 3-point / blend / linear:
        0.00%.
    """
    return sorted(chromosome) == list(range(1, STOP_COUNT + 1))


def tour_faults(chromosome) -> tuple[list, list]:
    """Which stops a broken route repeats, and which it never visits.

    Args:
        chromosome: a broken route.

    Returns:
        (repeated stops, missing stops), so the printed illegal
        children can be read rather than just counted.
    """
    repeated = sorted({s for s in chromosome if list(chromosome).count(s) > 1})
    missing = [s for s in range(1, STOP_COUNT + 1) if s not in list(chromosome)]
    return repeated, missing


# --- NEW (1) crossover_order() ------------------------------------------------
def crossover_order(parent1: list, parent2: list) -> tuple[list, list]:
    """Ordered crossover (OX1): copy a swath, fill the rest in the other's order.

    Pick two cut points. Child 1 takes the swath between them from parent 2
    verbatim, and fills the remaining positions - walking forward from the end
    of the swath, wrapping around - with the stops of parent 1 in the order
    parent 1 visits them, skipping the ones the swath already used. Child 2 is
    the same with the roles exchanged.

    Nothing here swaps gene values position by position, which is why the
    result is a permutation by construction rather than by luck. Note the
    assumption it hides: both parents must carry the SAME SET of values. Step 7
    ends by breaking that assumption on purpose.

    Args:
        parent1, parent2: sequences of the same length. Legal on
            permutations only if they carry the same set of values.

    Returns:
        Two children. On two tours: permutations by construction.
        On the real-valued parents: the parents themselves, silently.

    Example:
        Legal 100.00% on routes, 7.17 of 9 parent legs kept. Handed
        the real-valued pair it reaches 2 children, 100.00% of them
        a parent — the fill overwrites the swath when nothing is
        shared to skip.
    """
    length = len(parent1)
    start, end = sorted(random.randrange(length) for _ in range(2))

    def build(donor: list, keeper: list) -> list:
        child = [None] * length
        child[start:end + 1] = donor[start:end + 1]
        taken = set(child[start:end + 1])
        slot = (end + 1) % length
        for step in range(length):
            gene = keeper[(end + 1 + step) % length]
            if gene not in taken:
                child[slot] = gene
                slot = (slot + 1) % length
        return child

    return build(parent2, parent1), build(parent1, parent2)
# ------------------------------------------------------------------------------


parent1, parent2 = create_parents()
print(f"parent 1   {show(parent1)}")
print(f"parent 2   {show(parent2)}")


# --- NEW (2) the adjacency test -----------------------------------------------
tour1, tour2 = create_tours()
print(f"\nroute 1   {show(tour1)}   length {tour_length(tour1):.2f}")
print(f"route 2   {show(tour2)}   length {tour_length(tour2):.2f}")

random.seed(SEED)
print("\nFour ordered crossovers of those two routes:")
for _ in range(4):
    kid1, kid2 = crossover_order(tour1, tour2)
    print(f"    {show(kid1)}  legal={is_legal_tour(kid1)}    "
          f"{show(kid2)}  legal={is_legal_tour(kid2)}")


def edges(tour: Tour) -> set:
    """The unordered pairs of stops a route visits back to back.

    This is what a route really is, as far as its length is concerned: an
    ordering matters only through the adjacencies it creates.

    Args:
        tour: a permutation of stops.

    Returns:
        The unordered adjacencies the length actually depends on.

    Example:
        OX1 keeps 7.17 of 9 parent legs against 3.71 for a random route.
    """
    return {frozenset((tour[i], tour[(i + 1) % len(tour)])) for i in range(len(tour))}


parent_edges = edges(tour1) | edges(tour2)
random.seed(SEED)
kept, lengths = [], []
for _ in range(TRIALS):
    for child in crossover_order(tour1, tour2):
        kept.append(len(edges(child) & parent_edges))
        lengths.append(tour_length(child))
random.seed(SEED)
random_kept, random_lengths = [], []
for _ in range(TRIALS):
    tour = random.sample(range(1, STOP_COUNT + 1), STOP_COUNT)
    random_kept.append(len(edges(tour) & parent_edges))
    random_lengths.append(tour_length(tour))

parents_length = (tour_length(tour1) + tour_length(tour2)) / 2
print(f"\nWhat OX1 inherits, over {TRIALS} draws:")
print(f"    parent-to-parent legs kept by an OX1 child:  "
      f"{statistics.mean(kept):.2f} of {STOP_COUNT}")
print(f"    parent-to-parent legs kept by a random route: "
      f"{statistics.mean(random_kept):.2f} of {STOP_COUNT}")
print("A child keeps most of the legs its parents drove, which is exactly what")
print("recombination is supposed to mean and exactly what a repaired one-point")
print("child would not do.")

print(f"\nRoute length, on the other hand, says almost nothing here:")
print(f"    the two parents average       {parents_length:.2f}")
print(f"    OX1 children average          {statistics.mean(lengths):.2f}")
print(f"    {TRIALS} random routes average    {statistics.mean(random_lengths):.2f}")
print("On nine stops the good and the bad routes are close together, and these")
print("parents are not good ones, so a random route beats their children on")
print("average. Inheriting structure is not the same as inheriting quality -")
print("crossover supplies the first, and only selection supplies the second.")
# ------------------------------------------------------------------------------


# --- NEW (3) the other direction ----------------------------------------------
print("\nAnd now the same question in the other direction. Ordered crossover is")
print("handed the two REAL-VALUED parents of step 1:")
crossed = offspring_report("ordered (OX1)", crossover_order,
                           parent1, parent2, is_legal_real)
print_report_header()
print_report_row(crossed)
random.seed(SEED)
sample1, sample2 = crossover_order(parent1, parent2)
print(f"    child 1   {show(sample1)}")
print(f"    child 2   {show(sample2)}")
print(f"\nIt did not crash, and {crossed['legal']:.2f}% of what it produced is")
print(f"legal. It also reached {crossed['children']} distinct children and")
print(f"{crossed['clone']:.2f}% of them are a parent. Those two facts are the")
print("same fact: OX1 fills the positions outside its swath with the other")
print("parent's genes IN THE ORDER THAT PARENT HOLDS THEM, skipping the values")
print("the swath already used - and when the two parents share no values at")
print("all, nothing is ever skipped and the fill overwrites the swath. The")
print("operator returns the parents.")
print("\nThis is the worse half of the divide. A duplicated stop is loud: the")
print("program stops or the score is nonsense and somebody investigates. An")
print("operator that quietly returns its inputs breaks nothing, logs nothing,")
print("and simply makes the run stop searching.")
# ------------------------------------------------------------------------------


# --- NEW (4) the whole comparison ---------------------------------------------
def share_useful(operator: Operator, first: list, second: list,
                 is_legal: Legality) -> float:
    """% of children that are legal AND not a copy of a parent.

    Neither column on its own catches both halves of the divide: illegal
    children fail the first test, and an operator that hands the parents back
    fails the second. This multiplies the two, and it is the column the last
    table of the lesson is read from.

    Args:
        operator: any crossover of this lesson.
        first, second: the pair it is handed (reals or tours).
        is_legal: is_legal_real or is_legal_tour.

    Returns:
        % of children that are legal AND not a parent copy.

    Example:
        The last table of the lesson: operators useful on both
        representations — 0.
    """
    random.seed(SEED)
    useful = produced = 0
    for _ in range(TRIALS):
        for child in operator(first, second):
            produced += 1
            if is_legal(child) and list(child) != list(first) \
                    and list(child) != list(second):
                useful += 1
    return 100 * useful / produced


common = [
    ("clone (no crossover)", crossover_clone),
    ("one-point", crossover_one_point),
    (f"{N_POINTS[-1]}-point", lambda a, b: crossover_n_point(a, b, N_POINTS[-1])),
    (f"uniform (rate {UNIFORM_RATE})", crossover_uniform),
    (f"blend alpha={BLEND_ALPHAS[-1]} + clamp",
     lambda a, b: crossover_blend(a, b, BLEND_ALPHAS[-1])),
    (f"linear alpha={LINEAR_ALPHA}", crossover_linear),
    ("ordered (OX1)", crossover_order),
]
real_only = [("fitness-driven blend",
              lambda a, b: crossover_fitness_driven(a, b, BLEND_ALPHAS[-1]))]

print(f"\nSeven operators, two representations, {TRIALS} draws in every cell.")
print("`useful` is the share of children that are legal AND not a parent:")
print(f"\n{'operator':28s} {'REAL-VALUED, 6 genes':>28} {'ROUTE, 9 stops':>26}")
print(f"{'':28s} {'legal':>13} {'useful':>14} {'legal':>13} {'useful':>12}")
summary = []
for label, operator in common:
    real = offspring_report(label, operator, parent1, parent2, is_legal_real)
    tour = offspring_report(label, operator, tour1, tour2, is_legal_tour)
    real_useful = share_useful(operator, parent1, parent2, is_legal_real)
    tour_useful = share_useful(operator, tour1, tour2, is_legal_tour)
    summary.append((label, real["legal"], real_useful, tour["legal"], tour_useful))
    print(f"{label:28s} {real['legal']:12.2f}% {real_useful:13.2f}% "
          f"{tour['legal']:12.2f}% {tour_useful:11.2f}%")
for label, operator in real_only:
    real = offspring_report(label, operator, parent1, parent2, is_legal_real)
    real_useful = share_useful(operator, parent1, parent2, is_legal_real)
    print(f"{label:28s} {real['legal']:12.2f}% {real_useful:13.2f}% "
          f"{'n/a':>13} {'n/a':>12}")
print(f"{'':28s} fitness-driven has no route column: it has to SCORE its")
print(f"{'':28s} children, and an illegal route has no length.")

both = [row for row in summary if row[2] > 0 and row[4] > 0]
print(f"\nRows with a useful score above zero on BOTH sides: {len(both)}.")
print("That is the lesson, in one integer. Read the columns separately and each")
print("one looks ordinary; read them together and they do not overlap:")
for label, _, real_useful, _, tour_useful in summary:
    if label.startswith("clone"):
        continue
    side = "real-valued genes" if real_useful > tour_useful else "routes"
    print(f"    {label:28s} works on {side}, and only there")
print("\nCrossover is not a menu of interchangeable operators of differing")
print("strength. It is two families, chosen for you by the representation, and")
print("the failures they produce look completely different: illegal chromosomes")
print("on one side of the line, silent no-ops on the other.")
# ------------------------------------------------------------------------------


# --- NEW (5) the divide figure ------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4.6))
labels = [row[0] for row in summary]
spots = range(len(labels))
ax.bar([s - 0.2 for s in spots], [row[2] for row in summary], width=0.4,
       color="tab:blue", label="real-valued genes")
ax.bar([s + 0.2 for s in spots], [row[4] for row in summary], width=0.4,
       color="tab:orange", label="route of nine stops")
ax.set_xticks(list(spots))
ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
ax.set_ylabel("% of children legal and not a parent")
ax.set_ylim(0, 105)
ax.set_title("The divide: no operator is useful on both representations")
ax.legend(fontsize=9)
ax.grid(True, axis="y", linestyle=":", alpha=0.5)

FIGURES.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURES / "offspring_07_ordered.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigure saved to {FIGURES}/offspring_07_ordered.png")
# ------------------------------------------------------------------------------
print("A genetic algorithm on a route, with these operators in the loop,")
print("is Lesson 10. This lesson stops at what they do to one pair.")

