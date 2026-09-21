"""
Lesson 11 - Colouring 2: The baseline luck provides
===================================================
NEW IN THIS STEP: random_population() and a 500-colouring sample.

A random colouring assigns each of the three colours independently, so each edge
has one chance in three of being broken. That gives an expected conflict count
of |E|/3 before running anything, and the sample should land on it. What matters
is the other end of the histogram: the best of 500 random colourings, and how
far it still is from the 0 that the problem demands.

CHANGES FROM colouring_01_the_graph.py
Introduce them in this order:
    1. random_population()  draw the whole starting population at once
    2. the sample           500 colourings, their conflicts, and the theory line

Run it:  python colouring_02_random_population.py

500 random colourings yield 0 legal ones; the sample mean is 21.51
conflicts against the predicted |E|/3 = 21.33, and the best draw still
breaks 11 edges (17.2%).
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

    Example:
        500 random colourings have mean 21.51 against |E|/3 = 21.33; the
        best still breaks 11 edges.
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


# --- NEW (1) random_population() ----------------------------------------------
def random_population(size: int) -> List[List[int]]:
    """Independent uniform colours for every vertex of every individual.

    Args:
        size: number of colourings. 500 here.

    Returns:
        ``size`` lists of 30 colours in 0..2.

    Example:
        0 of 500 are legal; the best still breaks 11 edges (17.2%).
    """
    return [[random.randrange(COLOURS) for _ in vertices()] for _ in range(size)]
# ------------------------------------------------------------------------------


# --- NEW (2) the sample -------------------------------------------------------
all_vertices = vertices()
random.seed(SEED)
population = random_population(POPULATION_SIZE)
counts = [conflicts(colouring) for colouring in population]
expected = len(EDGES) / COLOURS
legal = [colouring for colouring in population if conflicts(colouring) == 0]
best_index = min(range(len(counts)), key=lambda index: counts[index])
# ------------------------------------------------------------------------------

print("Lesson 11 - Colouring 2: the baseline luck provides")
print(f"Seed {SEED}, population {POPULATION_SIZE}")
print(f"Legal colourings drawn:  {len(legal)}")
print(f"Conflicts: min {min(counts)}, mean {sum(counts) / len(counts):.2f}, max {max(counts)}")
print(f"Expected conflicts |E|/{COLOURS}: {expected:.2f}")
print(f"Sample mean is off the expectation by "
      f"{abs(sum(counts) / len(counts) - expected):.2f} edges")
print(f"Best of {POPULATION_SIZE} draws still breaks {min(counts)} of {len(EDGES)} edges "
      f"({min(counts) / len(EDGES):.1%})")
print(f"Search space: {COLOURS ** len(all_vertices):,} colourings. Unlike the equation")
print("box, this one cannot be walked, which is the case for using a search at all.")

FIGURES.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(counts, bins=range(min(counts), max(counts) + 2), color="#4c72b0", edgecolor="white")
ax.axvline(expected, color="black", linestyle="--", label=f"|E|/{COLOURS} = {expected:.2f}")
ax.axvline(min(counts), color="crimson", linestyle="--", label=f"best draw = {min(counts)}")
ax.set(xlabel="conflicting edges", ylabel="colourings",
       title=f"{POPULATION_SIZE} random colourings; {len(legal)} of them are legal")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "colouring_02_random_population.png", dpi=160)
plt.close(fig)
