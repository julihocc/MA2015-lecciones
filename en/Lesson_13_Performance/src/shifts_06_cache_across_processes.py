"""
Lesson 13 - Step 6: Bringing the cache and the meter back across the boundary
==============================================================================
NEW IN THIS STEP: the parent keeps the cache, and the workers report their cost.

Step 5 finished with a pooled run whose counters read zero evaluations and zero
work units, for a run that had plainly done both. Worker processes have separate
memory under both Windows ``spawn`` and Unix ``fork`` (copy-on-write): the cache
each fills and the counters each increments are private and disappear with the pool.

That is not only a measurement problem. Each worker builds a private cache, the
same repeated genome is answered once per worker that happens to receive it, and
none of those answers outlives the pool - so the cache of step 3 quietly stopped
working the moment the pool was introduced.

Both problems have one fix, and it is a rule worth remembering beyond this
lesson: *the parent owns the state, the workers own only the arithmetic.* Look
the genome up here, dispatch only the misses, and have each worker hand back
what its call cost so the meter can be added up at home.

CHANGES FROM shifts_05_parallel_evaluation.py
Introduce them in this order:
    1. evaluate_with_cost()  a worker returns the fitness AND the work it took
    2. evaluate_batch()      hits are answered here; only misses are dispatched
                             (step 3's cached_fitness() is absorbed into it)
    3. the_report            the pooled run and the sequential run, side by side

Run it:  python shifts_06_cache_across_processes.py

Pooled run now reports 291 / 8 hits / 61,110, matching the sequential
run on all three, answer still −87. Step 5's private per-worker caches
had been discarding every hit.
"""
import json
import os
import random
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt

SEED = 3
EMPLOYEES = 5
DAYS = 7
SHIFTS_PER_DAY = 3
SHIFTS = DAYS * SHIFTS_PER_DAY
GENOME_LENGTH = EMPLOYEES * SHIFTS
POPULATION_SIZE = 30
ELITE_SIZE = 2
CROSSOVER_PROBABILITY = 0.8
CROSSOVER_POINTS = 3
MUTATION_PROBABILITY = 0.5
MAX_GENERATIONS = 10
# (minimum, maximum) staff for the morning, day and evening shift.
SHIFT_LIMITS = ((1, 4), (2, 5), (1, 2))
# How many following shifts an employee must rest after each shift type.
RELAX_AFTER = (1, 1, 3)
EMPTY_SHIFT_PENALTY = 100
RELAX_WEIGHT = 5
FIGURES = Path(__file__).resolve().parent.parent / "figures"

# A snapshot is an ordinary file, so it belongs somewhere a student can open it
# and read the genes as text.
SNAPSHOTS = Path(__file__).resolve().parent.parent / "snapshots"
# Small enough to behave on any teaching machine; the script prints how many
# CPUs it actually found.
WORKERS = min(4, os.cpu_count() or 1)
# A deliberately expensive variant of the same fitness, used once at the end to
# show that the pool's verdict depends entirely on the cost of one call.
HEAVY_REPEATS = 20

# The meter. Nothing in this lesson is claimed without reading it.
COST: Dict[str, int] = {"calls": 0, "work": 0, "built": 0}


def shift_deviation(genome: List[int]) -> int:
    """Penalty for every shift staffed outside its allowed band.

    Walks all EMPLOYEES x SHIFTS cells, which is where the cost of an
    evaluation comes from and why the work counter lives in the inner loop.

    Args:
        genome: 105 bits, employee-major.

    Returns:
        Non-negative staffing penalty. One empty shift costs 100.

    Example:
        Together with rest violations this is 210 work units per call.
    """
    penalty = 0
    for s in range(SHIFTS):
        on_duty = 0
        for e in range(EMPLOYEES):
            COST["work"] += 1
            on_duty += genome[e * SHIFTS + s]
        low, high = SHIFT_LIMITS[s % SHIFTS_PER_DAY]
        penalty += max(low - on_duty, 0) + max(on_duty - high, 0)
        if on_duty == 0:
            penalty += EMPTY_SHIFT_PENALTY
    return penalty


def relax_violations(genome: List[int]) -> int:
    """Count the times an employee is put back on duty while still resting.

    Args:
        genome: 105 bits, employee-major.

    Returns:
        Number of rest violations. Weighted by 5 in the full objective.

    Example:
        The full-objective champion has 17 rest violations and scores -87.
    """
    violations = 0
    for e in range(EMPLOYEES):
        resting = 0
        for s in range(SHIFTS):
            COST["work"] += 1
            if genome[e * SHIFTS + s] == 1:
                if resting > 0:
                    violations += 1
                resting = RELAX_AFTER[s % SHIFTS_PER_DAY]
            else:
                resting = max(0, resting - 1)
    return violations


# One entry per distinct genome ever built. A hit is an evaluation that never
# happens; the counter is what lets the report below state a saving instead of
# claiming one.
CACHE: Dict[Tuple[int, ...], int] = {}
COST["hits"] = 0


def roster_fitness(genome: List[int]) -> int:
    """The objective, to be MAXIMISED. A perfect roster scores 0.

    Args:
        genome: 105 bits, employee-major.

    Returns:
        ``-(staffing + 5 * rest)``. Seed 3 reaches -87 on this objective.

    Example:
        Steps 1-6 all report best fitness -87 with this function.
    """
    COST["calls"] += 1
    return -(shift_deviation(genome) + RELAX_WEIGHT * relax_violations(genome))


class Individual:
    """One candidate roster and its fitness, which now arrives from outside.

    Evaluating inside the constructor was the right thing to do in step 2 and
    is the wrong thing to do here: an evaluation buried in an object's birth
    cannot be batched, and what cannot be batched cannot be dispatched. The
    fitness is still computed exactly once per genome - just not here.
    """

    def __init__(self, gene_list: List[int], fitness: int) -> None:
        self.gene_list = list(gene_list)
        self.fitness = fitness
        COST["built"] += 1


# --- NEW (1) evaluate_with_cost() ---------------------------------------------
def evaluate_with_cost(genome: List[int]) -> Tuple[int, int]:
    """Evaluate one genome and report what it cost.

    This is what runs in a worker. It reads the worker's own copy of COST to
    measure itself, then hands the delta back with the fitness: a worker cannot
    update the parent's meter, but it can tell the parent what to add.
    """
    before = COST["work"]
    value = roster_fitness(genome)
    return value, COST["work"] - before
# ------------------------------------------------------------------------------


# --- NEW (2) evaluate_batch() -------------------------------------------------
def evaluate_batch(genomes: Sequence[List[int]], pool: Optional[Pool] = None) -> List[int]:
    """Answer from the parent's cache; dispatch only what is genuinely new.

    Order is preserved by construction: every genome keeps its own slot, and
    the misses are written back into the slots they came from.
    """
    values: List[Optional[int]] = []
    misses: List[int] = []
    for i, genome in enumerate(genomes):
        key = tuple(genome)
        if key in CACHE:
            COST["hits"] += 1
            values.append(CACHE[key])
        else:
            values.append(None)
            misses.append(i)

    if misses:
        pending = [genomes[i] for i in misses]
        if pool is None:
            # Same process: the counters have already been incremented in place.
            results = [evaluate_with_cost(genome) for genome in pending]
        else:
            results = list(pool.map(evaluate_with_cost, pending))
            # Different processes: their counters died with them, so add the
            # totals they reported to ours.
            COST["calls"] += len(results)
            COST["work"] += sum(work for _, work in results)
        for i, (value, _) in zip(misses, results):
            CACHE[tuple(genomes[i])] = value
            values[i] = value
    return [value for value in values if value is not None]
# ------------------------------------------------------------------------------


Bred = Union[Individual, List[int]]


def materialise(items: List[Bred], pool: Optional[Pool] = None) -> List[Individual]:
    """Turn a generation of survivors and new genomes into individuals.

    Survivors come through untouched and cost nothing - that is section 13.1
    still doing its work. Only the raw genomes are sent to be evaluated.
    """
    pending = [i for i, item in enumerate(items) if not isinstance(item, Individual)]
    values = evaluate_batch([items[i] for i in pending], pool)
    result: List[Individual] = list(items)  # type: ignore[arg-type]
    for i, value in zip(pending, values):
        result[i] = Individual(items[i], value)  # type: ignore[arg-type]
    return result


def fitness_of(ind: Individual) -> int:
    """Reading a fitness is now a lookup. Not one call site had to change.

    Args:
        ind: a candidate roster with ``ind.fitness`` already stored.

    Returns:
        The stored integer. SELECT and STATS cost 0 from step 2 on.
    """
    return ind.fitness


def dump_population(population: List[Individual], path: Path) -> None:
    """Write only the genes. That is the whole of the book's snapshot.

    Args:
        population: the current generation.
        path: ``snapshots/shifts_population.json``.

    Returns:
        None. Restoring from genes alone costs 30 evaluations / 6,300 work.
    """
    path.parent.mkdir(exist_ok=True)
    with path.open("w") as handle:
        json.dump([ind.gene_list for ind in population], handle)


def restore_population(path: Path) -> List[Individual]:
    """Rebuild the individuals. Every constructor call is an evaluation.

    Args:
        path: the gene file written by ``dump_population``.

    Returns:
        30 individuals. Costs 30 evaluations unless the cache is restored first.
    """
    with path.open() as handle:
        return [Individual(gene_list) for gene_list in json.load(handle)]


def dump_cache(path: Path) -> None:
    """Write the fitness table as well. JSON has no tuples, so keys go as lists.

    Args:
        path: ``snapshots/shifts_cache.json``.

    Returns:
        None. The cache file is 14.0x the size of the gene file.
    """
    path.parent.mkdir(exist_ok=True)
    with path.open("w") as handle:
        json.dump([[list(key), value] for key, value in CACHE.items()], handle)


def restore_cache(path: Path) -> None:
    """Put the table back before anything is rebuilt, or it will not be used.

    Args:
        path: the cache file written by ``dump_cache``.

    Returns:
        None. Restoring the cache first makes population restore cost 0 / 0.
    """
    with path.open() as handle:
        for key, value in json.load(handle):
            CACHE[tuple(key)] = value


def create_random_genome() -> List[int]:
    """One random 105-bit genome, not yet an Individual.

    Returns:
        A bit list. Evaluation is deferred until ``materialise``.
    """
    return [random.choice([0, 1]) for _ in range(GENOME_LENGTH)]


def selection_rank_with_elite(population: List[Individual]) -> List[Individual]:
    """Rank selection, elites first. The sort alone is one evaluation each.

    Args:
        population: the current generation.

    Returns:
        A new list of the same length. Two elites are copied through.

    Example:
        Step 1 spends 300 SELECT evaluations on the sort; step 2 spends 0.
    """
    ordered = sorted(population, key=fitness_of, reverse=True)
    step = 1 / len(ordered)
    ranks = [1 - i * step for i in range(len(ordered))]
    total = sum(ranks)
    selected = ordered[:ELITE_SIZE]
    for _ in range(len(ordered) - ELITE_SIZE):
        shave = random.random() * total
        running = 0.0
        for i, rank in enumerate(ranks):
            running += rank
            if running > shave:
                selected.append(ordered[i])
                break
    return selected


def crossover_n_point(p1: List[int], p2: List[int], n: int) -> Tuple[List[int], List[int]]:
    """Swap alternating segments of two bit strings.

    Args:
        p1, p2: parent genomes of length 105.
        n: number of cut points. 3 here.

    Returns:
        Two children, not yet evaluated.
    """
    cuts = sorted(random.sample(range(1, len(p1) - 1), n) + [0, len(p1)])
    c1, c2 = list(p1), list(p2)
    for i in range(1, n + 1, 2):
        c1[cuts[i]:cuts[i + 1]] = p2[cuts[i]:cuts[i + 1]]
        c2[cuts[i]:cuts[i + 1]] = p1[cuts[i]:cuts[i + 1]]
    return c1, c2


def mutation_bit_flip(genome: List[int]) -> List[int]:
    """Flip one random bit.

    Args:
        genome: a 105-bit roster.

    Returns:
        A copy with one bit inverted. Hit rate falls as this rate rises.
    """
    mutant = list(genome)
    pos = random.randint(0, len(genome) - 1)
    mutant[pos] = 1 - mutant[pos]
    return mutant


def crossover_operation(population: List[Individual]) -> List["Bred"]:
    """Children leave here as gene lists. Nothing is evaluated yet."""
    offspring: List[Bred] = []
    for ind1, ind2 in zip(population[::2], population[1::2]):
        if random.random() < CROSSOVER_PROBABILITY:
            g1, g2 = crossover_n_point(ind1.gene_list, ind2.gene_list, CROSSOVER_POINTS)
            offspring.extend([g1, g2])
        else:
            offspring.extend([ind1, ind2])
    return offspring


def mutation_operation(population: List["Bred"]) -> List["Bred"]:
    """A mutant is a gene list; anything untouched stays the object it was."""
    offspring: List[Bred] = []
    for item in population:
        genome = item.gene_list if isinstance(item, Individual) else item
        if random.random() < MUTATION_PROBABILITY:
            offspring.append(mutation_bit_flip(genome))
        else:
            offspring.append(item)
    return offspring


def generation_stats(population: List[Individual], best: Individual) -> Tuple[Individual, float]:
    """Bookkeeping. Count how many times it has to read a fitness value.

    Args:
        population: the current generation.
        best: the best-ever individual so far.

    Returns:
        ``(best, mean)``. Step 1 spends 630 STATS evaluations here.
    """
    champion = max(population, key=fitness_of)
    if fitness_of(best) < fitness_of(champion):
        best = champion
    mean = sum(fitness_of(ind) for ind in population) / len(population)
    return best, mean


def run(pool: Optional[Pool] = None) -> Tuple[Individual, List[Dict[str, int]]]:
    """The same algorithm, with one batch evaluation per generation."""
    random.seed(SEED)
    population = materialise(
        [create_random_genome() for _ in range(POPULATION_SIZE)], pool)
    best = population[0]
    history: List[Dict[str, int]] = []
    for gen in range(1, MAX_GENERATIONS + 1):
        c0, b0 = COST["calls"], COST["built"]
        selected = selection_rank_with_elite(population)
        c1 = COST["calls"]
        population = materialise(
            mutation_operation(crossover_operation(selected)), pool)
        c2 = COST["calls"]
        best, mean = generation_stats(population, best)
        best_fitness = fitness_of(best)
        history.append({
            "gen": gen,
            "select": c1 - c0,
            "breed": c2 - c1,
            "stats": COST["calls"] - c2,
            "built": COST["built"] - b0,
            "best": best_fitness,
            "mean": round(mean, 1),
        })
    return best, history


if __name__ == "__main__":
    # Windows spawn re-imports this file; a Pool at import time deadlocks.
    best_individual, history = run()


    # --- NEW (3) the_report -------------------------------------------------------
    STEP_5_POOLED = (0, 0, 0)

    sequential = (COST["calls"], COST["hits"], COST["work"])
    sequential_best = [row["best"] for row in history]

    print("Lesson 13, step 6 - the cache and the meter cross the boundary")
    print("==============================================================")
    print(f"This machine reports {os.cpu_count()} CPUs; the pool below uses {WORKERS} "
          "worker processes.\n")

    CACHE.clear()
    COST.update(calls=0, work=0, built=0, hits=0)
    with Pool(processes=WORKERS) as pool:
        pooled_best, pooled_history = run(pool)
    pooled = (COST["calls"], COST["hits"], COST["work"])

    print("  run                        | evaluations | cache hits | work units")
    print("  ---------------------------+-------------+------------+-----------")
    print(f"  sequential                 | {sequential[0]:11d} | {sequential[1]:10d} |"
          f" {sequential[2]:10d}")
    print(f"  pooled, step 5             | {STEP_5_POOLED[0]:11d} | {STEP_5_POOLED[1]:10d} |"
          f" {STEP_5_POOLED[2]:10d}")
    print(f"  pooled, step 6             | {pooled[0]:11d} | {pooled[1]:10d} |"
          f" {pooled[2]:10d}")

    if pooled == sequential:
        print(f"\nThe pooled run and the sequential run now agree on all three numbers.")
        print("The work was done in other processes and the account of it came home.")
    else:
        print(f"\nThe two runs disagree: {sequential} against {pooled}. Something is "
              "still")
        print("being counted in a process that then throws the count away.")

    print(f"\nStep 5's pooled run reported {STEP_5_POOLED[0]} evaluations and "
          f"{STEP_5_POOLED[2]} work units for exactly this")
    print("search. Nothing about the search changed between then and now; what "
          "changed is")
    print(f"that somebody is keeping the books. And the {pooled[1]} cache hits are the "
          "other half of")
    print("the repair: in step 5 each worker kept a private cache that died with the")
    print("generation, so those repeats were paid for again. Here the lookup happens "
          "in")
    print("the parent, before anything is dispatched, and a hit costs no message at "
          "all.")

    same_answer = fitness_of(pooled_best) == fitness_of(best_individual)
    same_path = [row["best"] for row in pooled_history] == sequential_best
    print(f"\nBest fitness: {fitness_of(best_individual)} sequential, "
          f"{fitness_of(pooled_best)} pooled. Identical answer: {same_answer};")
    print(f"identical best-so-far in all {len(history)} generations: {same_path}.")
    print("\nThe rule this step is really about: the parent owns the state, the "
          "workers")
    print("own only the arithmetic. Every global a worker touches is a copy, and "
          "every")
    print("copy is thrown away - which is a bug when it is a cache, and a lie when "
          "it is")
    print("a counter.")

    labels = ["sequential", "pooled\n(step 5)", "pooled\n(step 6)"]
    work_values = [sequential[2], STEP_5_POOLED[2], pooled[2]]
    hit_values = [sequential[1], STEP_5_POOLED[1], pooled[1]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(labels, work_values, color=["tab:orange", "tab:red", "tab:green"])
    axes[0].set_title("Work units the parent can account for")
    axes[0].set_ylabel("work units")
    for i, v in enumerate(work_values):
        axes[0].text(i, v, str(v), ha="center", va="bottom")
    axes[1].bar(labels, hit_values, color=["tab:orange", "tab:red", "tab:green"])
    axes[1].set_title("Cache hits that survived the run")
    axes[1].set_ylabel("hits")
    for i, v in enumerate(hit_values):
        axes[1].text(i, v, str(v), ha="center", va="bottom")
    fig.suptitle("Step 6: the parent owns the state, the workers own the arithmetic")
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "shifts_06_cache_across_processes.png", dpi=130)
    plt.close(fig)
    print("\nFigure saved to figures/shifts_06_cache_across_processes.png")
    # ------------------------------------------------------------------------------

