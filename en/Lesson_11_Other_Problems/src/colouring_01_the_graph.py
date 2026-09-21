"""
Lesson 11 - Colouring 1: When the constraint is the whole objective
===================================================================
NEW IN THIS STEP: EDGES, vertices(), conflicts(), draw() and a random colouring.

Graph colouring is the opposite of the equation system. There the objective was
a smooth-ish quantity to drive down; here the only thing being measured is a
violated constraint. Fitness is the negated number of edges whose two ends share
a colour, so a legal colouring and an optimal colouring are the same object and
there is nothing to optimise once you reach it.

That shape has a consequence the rest of this sequence keeps running into: near
the answer the objective is flat. One conflict and two conflicts look almost
identical to the search, and zero is not reachable by getting gradually closer.

Run it:  python colouring_01_the_graph.py

30 vertices, 64 distinct edges, mean degree 4.27, and 3^30 colourings. A
random colouring at seed 7 breaks 26 of the 64 edges.
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
        edges: the 64-edge instance, default ``EDGES``.

    Returns:
        Sorted unique vertex ids. This graph has 30.

    Example:
        30 vertices, 3^30 = 205,891,132,094,649 colourings.
    """
    return sorted({end for edge in edges for end in edge})


def conflicts(colouring: Sequence[int],
              edges: Sequence[Tuple[int, int]] = EDGES) -> int:
    """Edges whose two ends share a colour. Zero is the whole objective.

    Args:
        colouring: one colour in 0..2 per vertex.
        edges: the instance edges.

    Returns:
        The conflict count. Fitness is its negation.

    Example:
        A random colouring at seed 7 breaks 26 of the 64 edges.
    """
    return sum(1 for one, two in edges if colouring[one] == colouring[two])


def degrees(edges: Sequence[Tuple[int, int]] = EDGES) -> Dict[int, int]:
    """How many edges touch each vertex.

    Args:
        edges: the instance edges.

    Returns:
        Vertex id to degree. Degrees run from 3 to 6; the mean is 4.27.
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
        None. Conflicting edges are drawn bold.
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


all_vertices = vertices()
degree = degrees()
random.seed(SEED)
random_colouring = [random.randrange(COLOURS) for _ in all_vertices]
broken_edges = [edge for edge in EDGES
                if random_colouring[edge[0]] == random_colouring[edge[1]]]

print("Lesson 11 - Colouring 1: when the constraint is the whole objective")
print(f"Vertices:            {len(all_vertices)}")
print(f"Edges:               {len(EDGES)} ({len({tuple(sorted(e)) for e in EDGES})} distinct)")
print(f"Degree:              min {min(degree.values())}, max {max(degree.values())}, "
      f"mean {sum(degree.values()) / len(all_vertices):.2f}")
print(f"Colours available:   {COLOURS}")
print(f"Chromosome:          {len(all_vertices)} genes, each in 0..{COLOURS - 1}, "
      f"{COLOURS ** len(all_vertices):,} colourings in total")
print(f"One random colouring at seed {SEED}: {random_colouring}")
print(f"  conflicting edges: {conflicts(random_colouring)}")
print(f"  fitness:           {-conflicts(random_colouring)} (0 would be a legal colouring)")
print(f"  first three broken edges: {broken_edges[:3]}")
print("Fitness is bounded above by 0 and that bound is the answer, not a target.")

FIGURES.mkdir(exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
draw(axes[0], [2] * len(all_vertices), "The instance, uncoloured")
draw(axes[1], random_colouring,
     f"A random 3-colouring: {conflicts(random_colouring)} conflicting edges (bold)")
for axis in axes:
    axis.set_axis_off()
fig.tight_layout()
fig.savefig(FIGURES / "colouring_01_the_graph.png", dpi=160)
plt.close(fig)
