"""
Lesson 13 - Step 5: Evaluating a generation in a process pool
==============================================================
NEW IN THIS STEP: batch evaluation, and a multiprocessing pool behind it.

This is the book's section 13.5, and the first technique in this lesson that
does not remove any work at all. Caching removes evaluations. A pool removes
none: it runs the same evaluations somewhere else, at the same time.

To dispatch work you must first have it in one place, and steps 2 to 4 had it
in the worst possible place - inside a constructor, one individual at a time.
So the real content of this step is not `Pool`; it is the restructuring that
makes a pool possible. Breeding now hands back *genomes*, a whole generation of
them is evaluated in one call, and only then do the individuals get built.
Survivors that were neither crossed nor mutated are carried through as the same
objects, so section 13.1's saving survives the rewrite intact.

Batching turned out to save something this step was not built to save, and the
report says so rather than hiding it. In steps 2 to 4 a child that was crossed
and then mutated was evaluated twice: once as the crossover product, once as
the mutant, and the first value was thrown away unused. Deferring evaluation
until breeding is finished evaluates only what survives to the end of it. The
same run drops from 409 evaluations to 291 - a 28.9% saving that has nothing to
do with parallelism and everything to do with evaluating late.

Two things this step measures, and one it refuses to.

  * It measures that both paths give the same answer, generation by generation.
  * It measures the parent process's counters after a pooled run - and finds
    them at zero, which is a fact about processes, not about arithmetic.
  * It prints one clock reading, and marks it for what it is: a measurement of
    the machine that ran the script, not a property of the algorithm. Step 3's
    hit rates are the same on every computer in the room. The seconds below
    are not.

CHANGES FROM shifts_04_snapshot.py
Introduce them in this order:
    1. Individual             the constructor takes a fitness; it stops evaluating
    2. evaluate_batch()       one call evaluates a whole generation, pool or no pool
    3. materialise()          survivors and new genomes become one list of individuals
    4. crossover_operation()  hand back raw genomes for the children
    5. mutation_operation()   the same, and carry untouched individuals through
    6. run()                  take a pool, and evaluate each generation in one batch
    7. the_report             both paths, the same answer, and one hedged clock

Run it:  python shifts_05_parallel_evaluation.py

Batching alone drops the run from 409 to 291 evaluations (−28.9%),
because a crossed-then-mutated child used to be evaluated twice. The
pooled run gives the identical answer and the identical best-so-far in
all 10 generations — while the parent's counters read 0 evaluations,
0 work units.
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


def cached_fitness(genome: List[int]) -> int:
    """Answer from the table if this exact genome has been evaluated before.

    The key has to be hashable and has to be the *whole* genome: two rosters
    that differ in one bit are two different rosters.

    Args:
        genome: 105 bits, employee-major.

    Returns:
        The stored or freshly computed fitness. Hits increment COST["hits"].

    Example:
        19 hits in 428 builds at mutation 0.50; 14.4% at 0.10, 3.2% at 0.75.
    """
    key = tuple(genome)
    if key in CACHE:
        COST["hits"] += 1
        return CACHE[key]
    value = roster_fitness(genome)
    CACHE[key] = value
    return value


# --- NEW (1) Individual -------------------------------------------------------
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
# ------------------------------------------------------------------------------


# --- NEW (2) evaluate_batch() -------------------------------------------------
def evaluate_batch(genomes: Sequence[List[int]], pool: Optional[Pool] = None) -> List[int]:
    """Evaluate a whole generation. With a pool, in as many processes as it has.

    `pool.map` keeps the input order, so the pooled path returns exactly the
    list the sequential path would have returned, in exactly the same order.
    """
    if pool is None:
        return [cached_fitness(genome) for genome in genomes]
    return list(pool.map(cached_fitness, genomes))
# ------------------------------------------------------------------------------


# --- NEW (3) materialise() ----------------------------------------------------
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
# ------------------------------------------------------------------------------


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


# --- NEW (4) crossover_operation() --------------------------------------------
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
# ------------------------------------------------------------------------------


# --- NEW (5) mutation_operation() ---------------------------------------------
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
# ------------------------------------------------------------------------------


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


# --- NEW (6) run() ------------------------------------------------------------
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
# ------------------------------------------------------------------------------


# --- NEW (7) the_report -------------------------------------------------------
def heavy_fitness(genome: List[int]) -> int:
    """The same objective, HEAVY_REPEATS times over. Same answer, more seconds."""
    value = 0
    for _ in range(HEAVY_REPEATS):
        value = roster_fitness(genome)
    return value
# ------------------------------------------------------------------------------


if __name__ == "__main__":
    # Windows spawn re-imports this file; a Pool at import time deadlocks.
    best_individual, history = run()

    sequential_calls = COST["calls"]
    sequential_work = COST["work"]
    sequential_hits = COST["hits"]
    sequential_best = [row["best"] for row in history]

    print("Lesson 13, step 5 - evaluating a generation in a process pool")
    print("=============================================================")
    print(f"This machine reports {os.cpu_count()} CPUs; the pool below uses {WORKERS} "
          "worker processes.\n")
    STEP_3_CALLS = 409
    print(f"Sequential run: {sequential_calls} evaluations, {sequential_hits} cache hits,")
    print(f"                {sequential_work} work units, best fitness "
          f"{fitness_of(best_individual)}.")

    print(f"\nStep 3 ran this same search for {STEP_3_CALLS} evaluations. Batching brought it")
    print(f"to {sequential_calls}, a saving of {100 * (STEP_3_CALLS - sequential_calls) / STEP_3_CALLS:.1f}%, and not one process was started to get it: a "
          "child")
    print("that is crossed and then mutated used to be evaluated twice, and the "
          "first")
    print("value was discarded unread. Evaluating at the end of breeding instead of "
          "in")
    print("the middle of it removes that, and it is the same idea as step 2 - do not "
          "pay")
    print("for a number until something actually reads it.")

    CACHE.clear()
    COST.update(calls=0, work=0, built=0, hits=0)
    with Pool(processes=WORKERS) as pool:
        pooled_best, pooled_history = run(pool)
    pooled_best_series = [row["best"] for row in pooled_history]

    print(f"\nPooled run:     {COST['calls']} evaluations, {COST['hits']} cache hits,")
    print(f"                {COST['work']} work units, best fitness "
          f"{fitness_of(pooled_best)}.")

    same_answer = fitness_of(pooled_best) == fitness_of(best_individual)
    same_path = pooled_best_series == sequential_best
    print(f"\nSame final answer: {same_answer}. Same best-so-far in every one of the "
          f"{len(history)}")
    print(f"generations: {same_path}. `pool.map` preserves order, so the pooled path "
          "receives")
    print("exactly the list the sequential path would have received. Parallelism here "
          "is an")
    print("implementation detail of the arithmetic, not a change to the search.")

    print(f"\nAnd yet the parent process counted {COST['calls']} evaluations and "
          f"{COST['work']} work units for a run")
    print("that plainly did the work. The counters, the cache and every other global "
          "live")
    print("in the workers' own memory: a forked process gets a copy, and nothing it "
          "writes")
    print("comes back. That is the real price of a pool, and step 6 is about paying it")
    print("properly.")

    sample = [create_random_genome() for _ in range(512)]
    timings = {}
    for label, function in (("cheap", cached_fitness), ("heavy", heavy_fitness)):
        CACHE.clear()
        start = time.perf_counter()
        [function(genome) for genome in sample]
        sequential_seconds = time.perf_counter() - start
        CACHE.clear()
        with Pool(processes=WORKERS) as pool:
            start = time.perf_counter()
            pool.map(function, sample)
            pooled_seconds = time.perf_counter() - start
        timings[label] = (sequential_seconds, pooled_seconds)

    print(f"\nEvaluating the same {len(sample)} genomes, measured with a clock:\n")
    print("  fitness                        | sequential | with pool | ratio")
    print("  -------------------------------+------------+-----------+------")
    for label, (sequential_seconds, pooled_seconds) in timings.items():
        weight = 1 if label == "cheap" else HEAVY_REPEATS
        print(f"  {label:<5} fitness ({weight:2d} x 210 units) | {sequential_seconds:9.3f}s |"
              f" {pooled_seconds:8.3f}s | {sequential_seconds / pooled_seconds:5.2f}")

    print("\nREAD THAT TABLE WITH CARE. Those seconds measure the machine that ran this")
    print("script, this once - its cores, its load, its Python. They are not a "
          "property of")
    print("the genetic algorithm and they will not reproduce elsewhere. Every other "
          "number")
    print("in this lesson will.")
    cheap_ratio = timings["cheap"][0] / timings["cheap"][1]
    heavy_ratio = timings["heavy"][0] / timings["heavy"][1]
    print(f"\nOn this run the same pool scored {cheap_ratio:.2f} on the cheap fitness and "
          f"{heavy_ratio:.2f} on a fitness")
    print(f"{HEAVY_REPEATS} times heavier, on identical data.")
    if heavy_ratio > cheap_ratio:
        print("The heavier fitness got more out of the pool, which is the direction to")
        print("expect: shipping a genome to another process costs the same either way, "
              "so")
        print("only the work waiting at the other end changes.")
    else:
        print("The ordering did not come out that way this time, which is what a "
              "measurement")
        print("of a few hundredths of a second on a shared machine looks like. That is "
              "the")
        print("point of the warning above, not an exception to it.")
    print("A pool is worth having when one evaluation is expensive enough to dwarf "
          "the")
    print("cost of shipping it, and the only way to know is to measure, on the "
          "machine")
    print("that will run the job.")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(range(1, len(history) + 1), sequential_best, "o-", label="sequential")
    axes[0].plot(range(1, len(history) + 1), pooled_best_series, "x--", label="pooled")
    axes[0].set_title("Two paths, one search")
    axes[0].set_xlabel("generation")
    axes[0].set_ylabel("best fitness")
    axes[0].legend()
    axes[0].grid(True, linestyle=":", alpha=0.5)
    labels = ["cheap", "heavy"]
    axes[1].bar([i - 0.2 for i in range(2)], [timings[l][0] for l in labels],
                width=0.4, label="sequential", color="tab:orange")
    axes[1].bar([i + 0.2 for i in range(2)], [timings[l][1] for l in labels],
                width=0.4, label="with pool", color="tab:blue")
    axes[1].set_xticks(range(2))
    axes[1].set_xticklabels(["cheap (1x)", f"heavy ({HEAVY_REPEATS}x)"])
    axes[1].set_ylabel("seconds on THIS machine")
    axes[1].set_title("Machine-dependent, and shown as such")
    axes[1].legend()
    fig.suptitle("Step 5: a pool moves the work; it does not remove any")
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "shifts_05_parallel_evaluation.png", dpi=130)
    plt.close(fig)
    print("\nFigure saved to figures/shifts_05_parallel_evaluation.png")
    # ------------------------------------------------------------------------------

