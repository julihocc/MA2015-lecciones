"""
Lesson 13 - Step 7: A coarse fitness, and what it costs to be cheap
====================================================================
NEW IN THIS STEP: a cheaper objective, and a switch that chooses one.

Every optimisation so far has been free. Section 13.1 stopped recomputing a
known number; the cache stopped recomputing a known genome; batching stopped
evaluating children that were about to be thrown away; the pool moved work
without changing it. Not one of them altered a single fitness value, and each
one was checked against the previous step's answer to prove it.

This last technique is not like that. The book's section 13.3 evaluates a
*coarse* version of the problem: cheaper, approximate, and no longer the thing
you actually want maximised. Here the roster's relaxation rule - the expensive
half of the objective, and the half that walks employees rather than shifts -
is simply dropped. One call costs 105 work units instead of 210.

Half price. The question this step exists to ask is what the other half was
buying, and the only honest way to answer it is to take the roster the cheap
search produced and score it on the real objective. That is what the report
does, and the number it gets is the reason this step is last.

CHANGES FROM shifts_06_cache_across_processes.py
Introduce them in this order:
    1. coarse_fitness()      the objective with its expensive half removed
    2. evaluate_with_cost()  the evaluator reads a switch: which objective to use
    3. the_report            run both, price both, and judge both on the real one

Run it:  python shifts_07_coarse_fitness.py

105 work units per call instead of 210, 51.0% less work for the run.
The coarse search reaches 0 on its own objective — and its roster
scores −160 on the real one against −87, with 32 rest violations
instead of 17.
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


# --- NEW (1) coarse_fitness() -------------------------------------------------
def coarse_fitness(genome: List[int]) -> int:
    """Half the objective: the staffing bands, and nothing about rest.

    `shift_deviation` walks the roster shift by shift; `relax_violations` walks
    it employee by employee and carries state as it goes. Dropping the second
    halves the cost of a call - and deletes the only term that knows an
    employee cannot work three shifts in a row.
    """
    COST["calls"] += 1
    return -shift_deviation(genome)
# ------------------------------------------------------------------------------


# --- NEW (2) evaluate_with_cost() ---------------------------------------------
# Which objective the search is actually maximising. It is a module-level name
# so that the workers inherit it when the pool forks, and so that there is
# exactly one place to look when the answers come out strange.
OBJECTIVE = roster_fitness


def _inherit_objective(objective) -> None:
    """Windows spawn does not fork; workers must be told which objective to use."""
    global OBJECTIVE
    OBJECTIVE = objective


def evaluate_with_cost(genome: List[int]) -> Tuple[int, int]:
    """Evaluate one genome with the current objective and report what it cost."""
    before = COST["work"]
    value = OBJECTIVE(genome)
    return value, COST["work"] - before
# ------------------------------------------------------------------------------


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
    exact_calls, exact_work = COST["calls"], COST["work"]
    exact_best_fitness = fitness_of(best_individual)
    exact_champion_genome = list(best_individual.gene_list)

    print("Lesson 13, step 7 - a coarse fitness, and what it costs to be cheap")
    print("===================================================================")
    print(f"Full objective:   {exact_calls} evaluations, {exact_work} work units, "
          f"{exact_work // exact_calls} units per call.")

    OBJECTIVE = coarse_fitness
    CACHE.clear()
    COST.update(calls=0, work=0, built=0, hits=0)
    with Pool(processes=WORKERS, initializer=_inherit_objective,
              initargs=(OBJECTIVE,)) as pool:
        coarse_best, coarse_history = run(pool)
    coarse_calls, coarse_work = COST["calls"], COST["work"]
    print(f"Coarse objective: {coarse_calls} evaluations, {coarse_work} work units, "
          f"{coarse_work // coarse_calls} units per call.")
    print(f"\nCost per call fell by {100 * (1 - (coarse_work / coarse_calls) / (exact_work / exact_calls)):.0f}%, exactly as designed: the coarse call "
          "walks the")
    print("roster once instead of twice.")

    # The only comparison that means anything is on the objective that is real.
    COST.update(calls=0, work=0)
    coarse_champion_exact = roster_fitness(coarse_best.gene_list)
    exact_champion_exact = roster_fitness(exact_champion_genome)
    coarse_violations = relax_violations(coarse_best.gene_list)
    exact_violations = relax_violations(exact_champion_genome)

    print("\n  search guided by   | its own score | scored on the FULL objective |"
          " rest violations")
    print("  -------------------+---------------+------------------------------+"
          "----------------")
    print(f"  the full objective | {fitness_of(best_individual):13d} |"
          f" {exact_champion_exact:28d} | {exact_violations:15d}")
    print(f"  the coarse one     | {fitness_of(coarse_best):13d} |"
          f" {coarse_champion_exact:28d} | {coarse_violations:15d}")

    gap = exact_champion_exact - coarse_champion_exact
    saving = 100 * (exact_work - coarse_work) / exact_work
    print(f"\nThe coarse search spent {saving:.1f}% less work and its roster scores "
          f"{coarse_champion_exact} on the")
    print(f"objective that matters, against {exact_champion_exact} for the search that "
          f"paid full price -")
    if gap > 0:
        print(f"a gap of {gap}. The coarse roster breaks the rest rule "
              f"{coarse_violations} times, because nothing")
        print("in the objective it was optimising had ever heard of that rule. It is "
              "not a")
        print("worse search; it is a search for something else, and the something else "
              "was")
        print("chosen because it was cheap.")
    elif gap == 0:
        print("no gap at all. On this instance the dropped term did not change where "
              "the")
        print("search ended up - which is a result about this roster, not a licence to")
        print("drop terms in general.")
    else:
        print(f"a gap of {-gap} in the coarse search's favour. That is worth stating "
              "plainly:")
        print("the cheaper objective found the better roster here, which happens when "
              "the")
        print("dropped term was mostly getting in the way early on.")

    print("\nThat is the note this course ends on. Every optimisation before this one "
          "was")
    print("free, and each was checked against the previous step's answer; this one is")
    print("cheaper than all of them and is the only one that had to be judged rather "
          "than")
    print("verified. An approximation is not a faster way of getting the same "
          "answer. It is a")
    print("different question, asked because it is cheaper to ask, and whether that "
          "was")
    print("a good trade is a measurement - the one printed above - and never an")
    print("assumption.")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(["full objective", "coarse objective"], [exact_work, coarse_work],
                color=["tab:blue", "tab:orange"])
    axes[0].set_title("Work units spent on the search")
    axes[0].set_ylabel("work units")
    for i, v in enumerate([exact_work, coarse_work]):
        axes[0].text(i, v, str(v), ha="center", va="bottom")
    axes[1].bar(["full objective", "coarse objective"],
                [exact_champion_exact, coarse_champion_exact],
                color=["tab:blue", "tab:orange"])
    axes[1].set_title("Both champions, scored on the FULL objective")
    axes[1].set_ylabel("true fitness (0 is perfect)")
    for i, v in enumerate([exact_champion_exact, coarse_champion_exact]):
        axes[1].text(i, v, str(v), ha="center", va="top")
    fig.suptitle("Step 7: the only optimisation in this lesson that is not free")
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "shifts_07_coarse_fitness.png", dpi=130)
    plt.close(fig)
    print("\nFigure saved to figures/shifts_07_coarse_fitness.png")
    # ------------------------------------------------------------------------------

