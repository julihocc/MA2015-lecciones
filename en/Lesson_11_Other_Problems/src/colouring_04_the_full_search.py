"""
Lesson 11 - Colouring 4: The search that does not finish, and the one that does
===============================================================================
NEW IN THIS STEP: selection_rank_with_elite(), run(), exact_colouring() and a
sweep over four seeds.

This is the step the lesson is built for. The genetic algorithm is assembled and
run, and on most seeds it stops one or two edges short of a legal colouring and
stays there for the rest of its budget - the plateau that step 1 predicted.
Then ten lines of backtracking settle the same question outright, and also prove
that two colours are impossible, which no amount of running the GA could have
established.

The result below is the measured one, not the planned one. The sequence was
designed expecting the GA to reach zero, because that is what Gridin reports.
It does so in one of the four seeds tried here. Both facts are the lesson.

CHANGES FROM colouring_03_fitness_driven_operators.py
Introduce them in this order:
    1. selection_rank_with_elite()  rank selection, two elites carried through
    2. run()                        the loop, stopping the moment conflicts hit 0
    3. exact_colouring()            backtracking that answers the question outright
    4. the seed sweep               how often the search actually reaches zero

Run it:  python colouring_04_the_full_search.py

1 of 4 seeds reaches a legal colouring (seed 11, generation 15). Seeds 7
and 16 stall at 2 conflicts and seed 1 at 1 conflict. Backtracking finds a
proper 3-colouring in 40 search nodes and proves no 2-colouring exists in
4 nodes.
"""
from collections import Counter
from pathlib import Path
import random
import time
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import igraph

SEED = 7
POPULATION_SIZE = 500
CROSSOVER_PROBABILITY = 0.5
MUTATION_PROBABILITY = 0.5
MAX_GENERATIONS = 200
ELITE_SIZE = 2
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
        The conflict count. 1 of 4 GA seeds reaches 0.
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
    """Draw the instance with igraph's layout and matplotlib's .

    Args:
        axis: a matplotlib axes.
        colouring: one colour in 0..2 per vertex.
        title: axes title.

    Returns:
        None.
    """
    """Draw the instance with igraph's layout and matplotlib's , so the
    figure is saved rather than shown and the script runs unattended."""
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


def crossover_n_point(first: Sequence[int], second: Sequence[int],
                      points: int = 2) -> Tuple[List[int], List[int]]:
    """Swap alternating segments. Any colouring is legal as a chromosome, so
    unlike lesson 10 there is nothing to repair - only quality to lose.

    Args:
        first, second: parent colourings.
        points: number of cut points. 2 here.

    Returns:
        Two children.
    """
    cuts = sorted(random.sample(range(1, len(first) - 1), points) + [0, len(first)])
    child_one, child_two = list(first), list(second)
    for index in range(1, points + 1, 2):
        left, right = cuts[index], cuts[index + 1]
        child_one[left:right] = second[left:right]
        child_two[left:right] = first[left:right]
    return child_one, child_two


def crossover_fitness_driven_n_point(first: Sequence[int], second: Sequence[int],
                                     points: int = 2) -> List[List[int]]:
    """Produce the two children, then return the best two of the four. A pair of
    parents can therefore never be replaced by something worse.

    Args:
        first, second: parent colourings.
        points: number of cut points. 2 here.

    Returns:
        The two lowest-conflict members of {children, parents}.
    """
    child_one, child_two = crossover_n_point(first, second, points)
    family = [list(child_one), list(child_two), list(first), list(second)]
    return sorted(family, key=conflicts)[:2]


def mutation_fitness_driven_random_change(colouring: Sequence[int],
                                          tries: int = 3) -> List[int]:
    """Recolour one random vertex, up to `tries` times, and return the first
    attempt that strictly improves. Otherwise leave the individual alone.

    Args:
        colouring: a parent colouring.
        tries: how many recolour attempts. 3 here.

    Returns:
        A strictly better colouring, or a copy of the parent.
    """
    current = conflicts(colouring)
    for _ in range(tries):
        mutant = list(colouring)
        mutant[random.randrange(len(mutant))] = random.randrange(COLOURS)
        if conflicts(mutant) < current:
            return mutant
    return list(colouring)


# --- NEW (1) selection_rank_with_elite() --------------------------------------
def selection_rank_with_elite(population: List[List[int]],
                              elite_size: int = 2) -> List[List[int]]:
    """Rank selection with elitism. Conflict counts sit in a narrow band, so
    rank - which ignores the size of the gaps - is the sensible pressure here.

    Args:
        population: colourings, any order.
        elite_size: how many of the current best are copied through. 2 here.

    Returns:
        A new list of the same length.
    """
    ordered = sorted(population, key=conflicts)
    rank_distance = 1.0 / len(population)
    ranks = [1.0 - index * rank_distance for index in range(len(population))]
    ranks_sum = sum(ranks)
    selected = [list(individual) for individual in ordered[:elite_size]]
    for _ in range(len(ordered) - elite_size):
        threshold = random.random() * ranks_sum
        running = 0.0
        for index, rank in enumerate(ranks):
            running += rank
            if running > threshold:
                selected.append(list(ordered[index]))
                break
    return selected
# ------------------------------------------------------------------------------


# --- NEW (2) run() ------------------------------------------------------------
def run(seed: int) -> Tuple[List[int], int, List[int]]:
    """Return the best colouring found, the generation reached, and the best
    conflict count after each generation.

    Args:
        seed: the only input that differs across the four runs.

    Returns:
        ``(best, generation, history)``. Stops at 0 conflicts or 200 generations.

    Example:
        Seed 11 reaches a legal colouring at generation 15; seeds 7 and 16
        stall at 2 conflicts and seed 1 at 1.
    """
    random.seed(seed)
    population = random_population(POPULATION_SIZE)
    best = min(population, key=conflicts)
    history = [conflicts(best)]
    generation = 0
    while generation < MAX_GENERATIONS and conflicts(best) > 0:
        generation += 1
        parents = selection_rank_with_elite(population, ELITE_SIZE)
        crossed: List[List[int]] = []
        for one, two in zip(parents[::2], parents[1::2]):
            if random.random() < CROSSOVER_PROBABILITY:
                crossed += crossover_fitness_driven_n_point(one, two)
            else:
                crossed += [one, two]
        population = [mutation_fitness_driven_random_change(individual)
                      if random.random() < MUTATION_PROBABILITY else individual
                      for individual in crossed]
        champion = min(population, key=conflicts)
        if conflicts(champion) < conflicts(best):
            best = champion
        history.append(conflicts(best))
    return best, generation, history
# ------------------------------------------------------------------------------


# --- NEW (3) exact_colouring() ------------------------------------------------
def exact_colouring(k: int) -> Tuple[List[int], int]:
    """Backtracking search for a proper k-colouring. Returns the colouring (or
    an empty list if none exists) and the number of search nodes visited.

    The symmetry cut - never open a colour numbered more than one above the
    highest already used - is what keeps this in the tens of nodes.

    Args:
        k: number of colours to try. 3 is possible; 2 is not.

    Returns:
        ``(colouring, nodes)``. Empty colouring means no k-colouring exists.

    Example:
        A proper 3-colouring in 40 nodes; no 2-colouring in 4 nodes.
    """
    neighbours: Dict[int, List[int]] = {v: [] for v in vertices()}
    for one, two in EDGES:
        neighbours[one].append(two)
        neighbours[two].append(one)
    assignment = [-1] * len(vertices())
    visited = 0

    def extend() -> bool:
        nonlocal visited
        visited += 1
        open_vertices = [v for v in vertices() if assignment[v] == -1]
        if not open_vertices:
            return True
        # Most constrained first: the vertex with the fewest colours still free.
        chosen = min(open_vertices,
                     key=lambda v: (-len({assignment[u] for u in neighbours[v]
                                          if assignment[u] >= 0}),
                                    -len(neighbours[v])))
        forbidden = {assignment[u] for u in neighbours[chosen] if assignment[u] >= 0}
        ceiling = min(k, max(assignment) + 2)
        for colour in range(ceiling):
            if colour in forbidden:
                continue
            assignment[chosen] = colour
            if extend():
                return True
            assignment[chosen] = -1
        return False

    return (list(assignment), visited) if extend() else ([], visited)
# ------------------------------------------------------------------------------


# --- NEW (4) the seed sweep ---------------------------------------------------
SEEDS = [7, 1, 16, 11]
ga_start = time.perf_counter()
results = {seed: run(seed) for seed in SEEDS}
ga_seconds = time.perf_counter() - ga_start
solved = [seed for seed, (best, _, _) in results.items() if conflicts(best) == 0]

exact_start = time.perf_counter()
proper, nodes_three = exact_colouring(3)
two_colour, nodes_two = exact_colouring(2)
exact_seconds = time.perf_counter() - exact_start
# ------------------------------------------------------------------------------

all_vertices = vertices()
print("Lesson 11 - Colouring 4: the search that does not finish, and the one that does")
print(f"Population {POPULATION_SIZE}, generation cap {MAX_GENERATIONS}, "
      f"elite {ELITE_SIZE}, seeds {SEEDS}")
for seed in SEEDS:
    best, generation, history = results[seed]
    print(f"  seed {seed:<3} stopped at generation {generation:<4} "
          f"best {conflicts(best)} conflicts   "
          f"(after 10 generations it was already at {history[min(10, len(history) - 1)]})")
print(f"Reached a legal colouring:  {len(solved)} of {len(SEEDS)} seeds {solved}")
print(f"Genetic search wall time:   {ga_seconds:.1f} s for {len(SEEDS)} runs")
print()
print("The same question, answered exactly by backtracking:")
print(f"  proper 3-colouring:       {'found' if proper else 'none exists'}, "
      f"{nodes_three} search nodes")
print(f"  conflicts in it:          {conflicts(proper) if proper else 'n/a'}")
print(f"  proper 2-colouring:       {'found' if two_colour else 'none exists'}, "
      f"{nodes_two} search nodes")
print(f"  wall time for both:       {exact_seconds * 1000:.2f} ms")
print()
print(f"So the instance is 3-colourable, and the genetic algorithm missed it on "
      f"{len(SEEDS) - len(solved)} of {len(SEEDS)} seeds.")
print("The plateau is why. Once the population is at one or two conflicts, almost")
print("every recolouring is neutral or worse, the fitness-driven mutation refuses")
print("nearly everything it tries, and elitism holds the population still.")
print("Reading the run alone, you could not tell a stuck search from an impossible")
print("instance - lesson 08's problem, in a case where the truth is one call away.")

FIGURES.mkdir(exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for seed in SEEDS:
    _, _, history = results[seed]
    axes[0].plot(history, label=f"seed {seed}")
axes[0].axhline(0, color="black", linestyle="--", label="legal colouring")
axes[0].set(xlabel="generation", ylabel="conflicting edges",
            title=f"{len(solved)} of {len(SEEDS)} seeds reach zero")
axes[0].legend(fontsize=8)
draw(axes[1], proper, f"The exact 3-colouring: {conflicts(proper)} conflicts, "
                      f"{nodes_three} search nodes")
axes[1].set_axis_off()
fig.tight_layout()
fig.savefig(FIGURES / "colouring_04_the_full_search.png", dpi=160)
plt.close(fig)
