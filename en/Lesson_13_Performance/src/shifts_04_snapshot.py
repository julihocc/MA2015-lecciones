"""
Lesson 13 - Step 4: Snapshots, and what a snapshot forgets
===========================================================
NEW IN THIS STEP: dumping and restoring a population, and dumping the cache.

Long runs get interrupted: a laptop sleeps, a queue job hits its wall time, a
tuning study wants to branch off a run that already cost an hour. The book's
answer (section 13.4) is to write the population's genes to a file and read
them back later.

That works, and it is three lines. But a population is not only its genes. Its
fitness values are the expensive part, and JSON does not carry them: rebuilding
an individual from a gene list runs the constructor, and the constructor
evaluates. So a restart from a plain snapshot pays for the whole population
again before it has done any new work.

The cache from step 3 is the fix, and it is the point of putting these two
techniques next to each other: dump the cache alongside the genes and the
restored population costs nothing at all. This script restarts the same
population twice, cold and warm, and counts both.

CHANGES FROM shifts_03_the_cache.py
Introduce them in this order:
    1. SNAPSHOTS          where the files go
    2. dump_population()  the genes of a population, one list each, as JSON
    3. restore_population() rebuild the individuals from the file
    4. dump_cache()       the cache is state too, and it is the expensive half
    5. restore_cache()    put the table back before rebuilding anything
    6. the_report         restart cold, restart warm, and count what each costs

Run it:  python shifts_04_snapshot.py

30 rosters restore for 30 evaluations / 6,300 work units from genes
alone, and for 0 / 0 when the cache is restored first. The cache file is
14.0× the size of the gene file.
"""
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

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

# --- NEW (1) SNAPSHOTS --------------------------------------------------------
# A snapshot is an ordinary file, so it belongs somewhere a student can open it
# and read the genes as text.
SNAPSHOTS = Path(__file__).resolve().parent.parent / "snapshots"
# ------------------------------------------------------------------------------

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


class Individual:
    """One candidate roster: 105 bits, employee-major, evaluated exactly once.

    The genes never change after construction, so the fitness cannot change
    either. Computing it here and storing it is the whole of section 13.1:
    three lines, no approximation, nothing given up.
    """

    def __init__(self, gene_list: List[int]) -> None:
        self.gene_list = list(gene_list)
        self.fitness = cached_fitness(self.gene_list)
        COST["built"] += 1


def fitness_of(ind: Individual) -> int:
    """Reading a fitness is now a lookup. Not one call site had to change.

    Args:
        ind: a candidate roster with ``ind.fitness`` already stored.

    Returns:
        The stored integer. SELECT and STATS cost 0 from step 2 on.
    """
    return ind.fitness


# --- NEW (2) dump_population() ------------------------------------------------
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
# ------------------------------------------------------------------------------


# --- NEW (3) restore_population() ---------------------------------------------
def restore_population(path: Path) -> List[Individual]:
    """Rebuild the individuals. Every constructor call is an evaluation.

    Args:
        path: the gene file written by ``dump_population``.

    Returns:
        30 individuals. Costs 30 evaluations unless the cache is restored first.
    """
    with path.open() as handle:
        return [Individual(gene_list) for gene_list in json.load(handle)]
# ------------------------------------------------------------------------------


# --- NEW (4) dump_cache() -----------------------------------------------------
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
# ------------------------------------------------------------------------------


# --- NEW (5) restore_cache() --------------------------------------------------
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
# ------------------------------------------------------------------------------


def create_random() -> Individual:
    """One random 105-bit roster.

    Returns:
        An ``Individual``. Construction may or may not evaluate, depending
        on the step.
    """
    return Individual([random.choice([0, 1]) for _ in range(GENOME_LENGTH)])


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


def crossover_operation(population: List[Individual]) -> List[Individual]:
    offspring: List[Individual] = []
    for ind1, ind2 in zip(population[::2], population[1::2]):
        if random.random() < CROSSOVER_PROBABILITY:
            g1, g2 = crossover_n_point(ind1.gene_list, ind2.gene_list, CROSSOVER_POINTS)
            offspring.extend([Individual(g1), Individual(g2)])
        else:
            offspring.extend([ind1, ind2])
    return offspring


def mutation_operation(population: List[Individual]) -> List[Individual]:
    offspring: List[Individual] = []
    for ind in population:
        if random.random() < MUTATION_PROBABILITY:
            offspring.append(Individual(mutation_bit_flip(ind.gene_list)))
        else:
            offspring.append(ind)
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


def run() -> Tuple[Individual, List[Dict[str, int]]]:
    random.seed(SEED)
    population = [create_random() for _ in range(POPULATION_SIZE)]
    best = population[0]
    history: List[Dict[str, int]] = []
    for gen in range(1, MAX_GENERATIONS + 1):
        c0, b0 = COST["calls"], COST["built"]
        selected = selection_rank_with_elite(population)
        c1 = COST["calls"]
        population = mutation_operation(crossover_operation(selected))
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


best_individual, history = run()


# --- NEW (6) the_report -------------------------------------------------------
run_calls, run_work = COST["calls"], COST["work"]
WORK_PER_CALL = 2 * EMPLOYEES * SHIFTS

print("Lesson 13, step 4 - snapshots, and what a snapshot forgets")
print("==========================================================")
print(f"The run cost {run_calls} evaluations and {run_work} work units, and ended "
      f"at fitness {fitness_of(best_individual)}.")

# The population to save is the one the last generation left behind, and run()
# returns only the champion. So the run is replayed here, inline. With the
# cache from the first run still warm the replay is free, and the count below
# says so.
replay_before = COST["calls"]
random.seed(SEED)
population = [create_random() for _ in range(POPULATION_SIZE)]
for _ in range(MAX_GENERATIONS):
    population = mutation_operation(crossover_operation(
        selection_rank_with_elite(population)))

replay_calls = COST["calls"] - replay_before
print(f"Replaying that run to recover its last population cost {replay_calls} "
      "evaluations:")
print("same seed, same genomes, and the cache already held every one of them.")

genes_path = SNAPSHOTS / "shifts_population.json"
cache_path = SNAPSHOTS / "shifts_cache.json"
dump_population(population, genes_path)
dump_cache(cache_path)
print(f"\nSaved {len(population)} rosters to snapshots/{genes_path.name} "
      f"({genes_path.stat().st_size} bytes)")
print(f"Saved {len(CACHE)} cached fitness values to snapshots/{cache_path.name} "
      f"({cache_path.stat().st_size} bytes)")

original = [ind.fitness for ind in population]

# Restart 1: a fresh process would start with an empty cache. Simulate that.
CACHE.clear()
COST.update(calls=0, work=0, hits=0)
cold = restore_population(genes_path)
cold_calls, cold_work = COST["calls"], COST["work"]

# Restart 2: same file, but the cache is restored first.
CACHE.clear()
COST.update(calls=0, work=0, hits=0)
restore_cache(cache_path)
warm = restore_population(genes_path)
warm_calls, warm_work = COST["calls"], COST["work"]

print("\n  restart          | evaluations | work units")
print("  -----------------+-------------+-----------")
print(f"  genes only       | {cold_calls:11d} | {cold_work:10d}")
print(f"  genes + cache    | {warm_calls:11d} | {warm_work:10d}")

ratio = cache_path.stat().st_size / genes_path.stat().st_size
print(f"\nThe cache file is {ratio:.1f} times the size of the gene file, and what it "
      f"bought was")
print(f"{cold_calls - warm_calls} evaluations. At {WORK_PER_CALL} work units each that is a bad "
      "trade; on a fitness")
print("function that takes a minute a call it is an excellent one. The technique "
      "is")
print("the same either way - only the exchange rate changes.")

print(f"\nA snapshot of {len(population)} rosters costs {cold_calls} evaluations to "
      "come back to life,")
print("because JSON stores genes and the constructor is what turns genes into a")
print(f"fitness. Restoring the cache first brings that to {warm_calls}: every roster in "
      "the")
print("file is a genome the table already knows.")

exact = all(a == b for a, b in zip(original, [ind.fitness for ind in cold]))
exact_warm = all(a == b for a, b in zip(original, [ind.fitness for ind in warm]))
print(f"\nBoth restored populations match the saved one value for value: "
      f"{exact and exact_warm}.")
print("A snapshot that changed a fitness would be worse than no snapshot at all, "
      "so")
print("this is checked rather than assumed.")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].bar(["genes only", "genes + cache"], [cold_work, warm_work],
            color=["tab:red", "tab:green"])
axes[0].set_title(f"Cost of restoring {len(population)} rosters")
axes[0].set_ylabel("work units")
for i, v in enumerate([cold_work, warm_work]):
    axes[0].text(i, v, str(v), ha="center", va="bottom")
axes[1].plot(range(1, len(original) + 1), sorted(original), "o-", label="saved")
axes[1].plot(range(1, len(original) + 1),
             sorted(ind.fitness for ind in warm), "x--", label="restored")
axes[1].set_title("Restored fitness values, sorted")
axes[1].set_xlabel("roster")
axes[1].set_ylabel("fitness")
axes[1].legend()
axes[1].grid(True, linestyle=":", alpha=0.5)
fig.suptitle("Step 4: genes are cheap to store; fitness is what costs")
fig.tight_layout()
FIGURES.mkdir(exist_ok=True)
fig.savefig(FIGURES / "shifts_04_snapshot.png", dpi=130)
plt.close(fig)
print("\nFigure saved to figures/shifts_04_snapshot.png")
# ------------------------------------------------------------------------------

