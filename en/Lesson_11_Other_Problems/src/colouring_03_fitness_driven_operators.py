"""
Lesson 11 - Colouring 3: Operators that consult the fitness function
====================================================================
NEW IN THIS STEP: crossover_n_point(), the two fitness-driven operators, and a
measurement of all of them on the same population.

On a plateau, a blind operator is a coin toss. Gridin's answer for this problem
is to let the operators look before they leap: a crossover that keeps the best
two of {both children, both parents}, and a mutation that tries a few random
recolourings and keeps one only if it lowers the conflict count. Both are
greedy, and greed is a real choice with a real cost, so measure it rather than
assert it.

CHANGES FROM colouring_02_random_population.py
Introduce them in this order:
    1. crossover_n_point()                      the plain two-point operator
    2. crossover_fitness_driven_n_point()       children must beat their parents
    3. mutation_fitness_driven_random_change()  keep a recolour only if it helps
    4. the experiment                           all three, on one population

Run it:  python colouring_03_fitness_driven_operators.py

Plain two-point crossover moves the population mean by +0.03 edges; the
fitness-driven version by −2.32. Greedy recolouring moves the mean by
−1.02 but leaves 206 of 500 individuals untouched, and its best (11) is
worse than a blind recolour's best (9).
"""
from collections import Counter
from pathlib import Path
import random
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import igraph

SEED = 7
POPULATION_SIZE = 500
FIGURES = Path(__file__).resolve().parent.parent / "figures"
PALETTE = ["#d62728", "#2ca02c", "#1f77b4"]

# The instance is Gridin's, kept verbatim so the lesson can be compared with the
# book: 30 vertices, and the edges below.
EDGES: List[Tuple[int, int]] = [
    (0, 2), (3, 4), (3, 5), (0, 5), (1, 4), (2, 10), (14, 8), (15, 3), (16, 9), (17, 11), (18, 7),
    (19, 7), (6, 3), (7, 6), (8, 0), (11, 5), (11, 9), (12, 1), (13, 4), (20, 5), (21, 17), (22, 15),
    (23, 20), (24, 20), (25, 17), (26, 24), (27, 10), (28, 15), (29, 21), (12, 14), (16, 20), (29, 19),
    (27, 22), (26, 9), (25, 26), (23, 10), (6, 28), (14, 13), (0, 12), (3, 2), (19, 14), (22, 10), (1, 18),
    (7, 21), (15, 12), (11, 1), (23, 28), (6, 11), (9, 25), (17, 9), (24, 16), (27, 28), (18, 20), (19, 21),
    (1, 14), (22, 29), (17, 4), (8, 13), (7, 23), (16, 28), (5, 9), (12, 29), (27, 21), (1, 23)
]
COLOURS = 3


def vertices(edges: Sequence[Tuple[int, int]] = EDGES) -> List[int]:
    """Every vertex mentioned by an edge, in order.

    Args:
        edges: the 64-edge instance.

    Returns:
        Sorted unique vertex ids. This graph has 30.
    """
    return sorted({end for edge in edges for end in edge})


def conflicts(colouring: Sequence[int],
              edges: Sequence[Tuple[int, int]] = EDGES) -> int:
    """Edges whose two ends share a colour. Zero is the whole objective.

    Args:
        colouring: one colour in 0..2 per vertex.
        edges: the instance edges.

    Returns:
        The conflict count.
    """
    return sum(1 for one, two in edges if colouring[one] == colouring[two])


def degrees(edges: Sequence[Tuple[int, int]] = EDGES) -> Dict[int, int]:
    """How many edges touch each vertex.

    Args:
        edges: the instance edges.

    Returns:
        Vertex id to degree.
    """
    counter: Counter = Counter()
    for one, two in edges:
        counter[one] += 1
        counter[two] += 1
    return dict(counter)


def draw(axis, colouring: Sequence[int], title: str) -> None:
    """Draw the instance with igraph's layout and matplotlib's , so the
    figure is saved rather than shown and the script runs unattended.

    Args:
        axis: a matplotlib axes.
        colouring: one colour in 0..2 per vertex.
        title: axes title.

    Returns:
        None.
    """
    graph = igraph.Graph(n=len(vertices()), edges=EDGES, directed=False)
    layout = graph.layout_kamada_kawai()
    broken = {index for index, (one, two) in enumerate(EDGES)
              if colouring[one] == colouring[two]}
    igraph.plot(
        graph, target=axis, layout=layout,
        vertex_color=[PALETTE[colour] for colour in colouring],
        vertex_size=22, vertex_label=[str(v) for v in vertices()],
        vertex_label_size=7, vertex_frame_width=0.5,
        edge_color=["#000000" if index in broken else "#bbbbbb"
                    for index in range(len(EDGES))],
        edge_width=[2.2 if index in broken else 0.7 for index in range(len(EDGES))],
    )
    axis.set_title(title)


def random_population(size: int) -> List[List[int]]:
    """Independent uniform colours for every vertex of every individual.

    Args:
        size: number of colourings. 500 here.

    Returns:
        ``size`` lists of 30 colours in 0..2.
    """
    return [[random.randrange(COLOURS) for _ in vertices()] for _ in range(size)]


# --- NEW (1) crossover_n_point() ----------------------------------------------
def crossover_n_point(first: Sequence[int], second: Sequence[int],
                      points: int = 2) -> Tuple[List[int], List[int]]:
    """Swap alternating segments. Any colouring is legal as a chromosome, so
    unlike lesson 10 there is nothing to repair - only quality to lose.

    Args:
        first, second: parent colourings.
        points: number of cut points. 2 here.

    Returns:
        Two children. Quality can go either way.

    Example:
        Plain two-point crossover moves the mean by +0.03 edges.
    """
    cuts = sorted(random.sample(range(1, len(first) - 1), points) + [0, len(first)])
    child_one, child_two = list(first), list(second)
    for index in range(1, points + 1, 2):
        left, right = cuts[index], cuts[index + 1]
        child_one[left:right] = second[left:right]
        child_two[left:right] = first[left:right]
    return child_one, child_two
# ------------------------------------------------------------------------------


# --- NEW (2) crossover_fitness_driven_n_point() -------------------------------
def crossover_fitness_driven_n_point(first: Sequence[int], second: Sequence[int],
                                     points: int = 2) -> List[List[int]]:
    """Produce the two children, then return the best two of the four. A pair of
    parents can therefore never be replaced by something worse.

    Args:
        first, second: parent colourings.
        points: number of cut points. 2 here.

    Returns:
        The two lowest-conflict members of {children, parents}.

    Example:
        The fitness-driven version moves the mean by −2.32 edges.
    """
    child_one, child_two = crossover_n_point(first, second, points)
    family = [list(child_one), list(child_two), list(first), list(second)]
    return sorted(family, key=conflicts)[:2]
# ------------------------------------------------------------------------------


# --- NEW (3) mutation_fitness_driven_random_change() --------------------------
def mutation_fitness_driven_random_change(colouring: Sequence[int],
                                          tries: int = 3) -> List[int]:
    """Recolour one random vertex, up to `tries` times, and return the first
    attempt that strictly improves. Otherwise leave the individual alone.

    Args:
        colouring: a parent colouring.
        tries: how many recolour attempts. 3 here.

    Returns:
        A strictly better colouring, or a copy of the parent.

    Example:
        Mean conflicts fall by 1.02, but 206 of 500 individuals are left
        untouched; the best (11) is worse than a blind recolour's best (9).
    """
    current = conflicts(colouring)
    for _ in range(tries):
        mutant = list(colouring)
        mutant[random.randrange(len(mutant))] = random.randrange(COLOURS)
        if conflicts(mutant) < current:
            return mutant
    return list(colouring)
# ------------------------------------------------------------------------------


# --- NEW (4) the experiment ---------------------------------------------------
def mean(values: Sequence[float]) -> float:
    """Arithmetic mean of a non-empty sequence.

    Args:
        values: conflict counts or other numbers.

    Returns:
        ``sum(values) / len(values)``.
    """
    return sum(values) / len(values)


all_vertices = vertices()
random.seed(SEED)
population = random_population(POPULATION_SIZE)
before = [conflicts(colouring) for colouring in population]
pairs = list(zip(population[::2], population[1::2]))

plain_children: List[List[int]] = []
for one, two in pairs:
    plain_children += list(crossover_n_point(one, two))
driven_children: List[List[int]] = []
for one, two in pairs:
    driven_children += crossover_fitness_driven_n_point(one, two)

blind_mutants = []
for colouring in population:
    mutant = list(colouring)
    mutant[random.randrange(len(mutant))] = random.randrange(COLOURS)
    blind_mutants.append(mutant)
driven_mutants = [mutation_fitness_driven_random_change(c) for c in population]
unchanged = sum(1 for before_one, after_one in zip(population, driven_mutants)
                if before_one == after_one)
# ------------------------------------------------------------------------------

plain = [conflicts(child) for child in plain_children]
driven = [conflicts(child) for child in driven_children]
blind = [conflicts(mutant) for mutant in blind_mutants]
greedy = [conflicts(mutant) for mutant in driven_mutants]

print("Lesson 11 - Colouring 3: operators that consult the fitness function")
print(f"Seed {SEED}, population {POPULATION_SIZE}, {len(pairs)} parent pairs")
print(f"Parents:                       mean {mean(before):.2f} conflicts, best {min(before)}")
print(f"Plain two-point crossover:     mean {mean(plain):.2f} conflicts, best {min(plain)}")
print(f"Fitness-driven crossover:      mean {mean(driven):.2f} conflicts, best {min(driven)}")
print(f"Blind single recolour:         mean {mean(blind):.2f} conflicts, best {min(blind)}")
print(f"Fitness-driven recolour:       mean {mean(greedy):.2f} conflicts, best {min(greedy)}")
print(f"Fitness-driven recolour left {unchanged} of {POPULATION_SIZE} individuals untouched, "
      f"having tried and rejected every recolouring it sampled")
print(f"Plain crossover moves the mean by {mean(plain) - mean(before):+.2f} edges; "
      f"the fitness-driven version by {mean(driven) - mean(before):+.2f}.")
print(f"Best individual after mutation: blind {min(blind)}, greedy {min(greedy)}; greed moved"
      f" the mean by {mean(greedy) - mean(before):+.2f} edges and the best by {min(greedy) - min(blind):+d}.")
print(f"Neither operator produced a legal colouring: "
      f"{sum(1 for c in plain + driven + blind + greedy if c == 0)} of "
      f"{len(plain) + len(driven) + len(blind) + len(greedy)} results had 0 conflicts.")
print("Greed is not free, and the cost is visible in the two functions above:")
print("crossover_fitness_driven_n_point() counts conflicts on four colourings per")
print("pair - two children and both parents - where the plain operator counts none,")
print("and mutation_fitness_driven_random_change() counts one plus up to three more")
print(f"per individual. On this population that is at least {4 * len(pairs):,} extra counts")
print("for the crossover alone.")

FIGURES.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.boxplot([before, plain, driven, blind, greedy], vert=True,
           tick_labels=["parents", "plain\ncrossover", "driven\ncrossover",
                        "blind\nrecolour", "driven\nrecolour"])
ax.set(ylabel="conflicting edges",
       title="Greedy operators move the population down; neither reaches 0")
fig.tight_layout()
fig.savefig(FIGURES / "colouring_03_fitness_driven_operators.png", dpi=160)
plt.close(fig)
