"""
Lesson 13 - Step 3: A cache for genomes that come back
=======================================================
NEW IN THIS STEP: a dictionary from genome to fitness, and a hit counter.

Step 2 put the bill at one evaluation per individual built. That is a floor
only for *new* genomes. A genetic algorithm does not produce new genomes all
the time: elites are copied forward, selection picks the same parents twice,
crossover between two copies of one roster returns that roster unchanged, and
a mutation can undo an earlier one. Every one of those is a genome the program
has already priced.

The book's section 13.2 keeps a dictionary from genome to fitness. A hit is an
evaluation that never happens, so a hit costs 0 work units rather than 210.
The cache changes nothing about the search: it returns exactly the value the
fitness function would have returned, so the answer must come out identical to
step 2, and the script checks that rather than assuming it.

The price is memory. The cache holds one entry per distinct genome ever seen,
and nothing in this design ever evicts one.

CHANGES FROM shifts_02_evaluate_once.py
Introduce them in this order:
    1. CACHE             one dictionary from genome to fitness, and a hit counter
    2. cached_fitness()  look the genome up before paying for it
    3. Individual        the constructor calls cached_fitness() - a changed line
    4. the_report        count hits, and check the cache did not alter the answer
    5. hit_rate_scan()   is a small hit rate the cache's fault, or the search's?

Run it:  python shifts_03_the_cache.py

19 hits in 428 builds, 409 distinct genomes, 4.4% less work, answer
unchanged at −87. Hit rate is the search's property: 14.4% at mutation
p=0.10, 6.1% at 0.25, 4.4% at 0.50, 3.2% at 0.75.
"""
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


# --- NEW (1) CACHE ------------------------------------------------------------
# One entry per distinct genome ever built. A hit is an evaluation that never
# happens; the counter is what lets the report below state a saving instead of
# claiming one.
CACHE: Dict[Tuple[int, ...], int] = {}
COST["hits"] = 0
# ------------------------------------------------------------------------------


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


# --- NEW (2) cached_fitness() -------------------------------------------------
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
# ------------------------------------------------------------------------------


class Individual:
    """One candidate roster: 105 bits, employee-major, evaluated exactly once.

    The genes never change after construction, so the fitness cannot change
    either. Computing it here and storing it is the whole of section 13.1:
    three lines, no approximation, nothing given up.

    Args:
        gene_list: 105 bits, employee-major. Fitness comes from ``cached_fitness``.

    Example:
        19 cache hits in 428 builds; answer still -87.
    """

    def __init__(self, gene_list: List[int]) -> None:
        """Look up or compute fitness, then store it.

        Args:
            gene_list: 105 bits, employee-major.
        """
        self.gene_list = list(gene_list)
        self.fitness = cached_fitness(self.gene_list)   # --- CHANGED --- was roster_fitness()
        COST["built"] += 1


def fitness_of(ind: Individual) -> int:
    """Reading a fitness is now a lookup. Not one call site had to change.

    Args:
        ind: a candidate roster with ``ind.fitness`` already stored.

    Returns:
        The stored integer. SELECT and STATS cost 0 from step 2 on.
    """
    return ind.fitness


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
    """Cross pairs with probability 0.8, else copy both parents through.

    Args:
        population: selected individuals.

    Returns:
        A new list of the same length.
    """
    offspring: List[Individual] = []
    for ind1, ind2 in zip(population[::2], population[1::2]):
        if random.random() < CROSSOVER_PROBABILITY:
            g1, g2 = crossover_n_point(ind1.gene_list, ind2.gene_list, CROSSOVER_POINTS)
            offspring.extend([Individual(g1), Individual(g2)])
        else:
            offspring.extend([ind1, ind2])
    return offspring


def mutation_operation(population: List[Individual]) -> List[Individual]:
    """Flip one bit of each individual with probability 0.5.

    Args:
        population: after crossover.

    Returns:
        A new list of the same length. Higher mutation lowers cache hits.
    """
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
    """Ten generations from seed 3, with a genome-to-fitness cache.

    Returns:
        ``(best, history)``. Hits are counted in COST["hits"].

    Example:
        19 hits in 428 builds, 4.4% less work, answer still -87.
    """
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


# --- NEW (4) the_report -------------------------------------------------------
STEP_2_CALLS = 428
STEP_2_WORK = 89880
STEP_2_BEST = -87
STEP_1_WORK_REF = 195300
WORK_PER_CALL = 2 * EMPLOYEES * SHIFTS

print("Lesson 13, step 3 - a cache for genomes that come back")
print("======================================================")
print("BREED still builds the same individuals; the column now counts only the")
print("ones it had to evaluate.\n")
print(" gen | new individuals | evaluated in BREED | best |   mean")
print("-----+-----------------+--------------------+------+-------")
for row in history:
    print(f" {row['gen']:3d} | {row['built']:15d} | {row['breed']:18d} |"
          f" {row['best']:4d} | {row['mean']:6.1f}")

built = COST["built"]
calls = COST["calls"]
hits = COST["hits"]
print(f"\n{built} individuals were built. {calls} of them needed an evaluation and "
      f"{hits} were")
print(f"answered from the cache, which holds {len(CACHE)} distinct genomes.")
print("\n                     step 2     step 3     saved")
print(f"  fitness calls    {STEP_2_CALLS:8d}   {calls:8d}   {100 * (STEP_2_CALLS - calls) / STEP_2_CALLS:5.1f}%")
print(f"  work units       {STEP_2_WORK:8d}   {COST['work']:8d}   {100 * (STEP_2_WORK - COST['work']) / STEP_2_WORK:5.1f}%")
print(f"\nEach hit saved {WORK_PER_CALL} work units, so the cache saved "
      f"{hits * WORK_PER_CALL} of them in total.")
if hits * WORK_PER_CALL == STEP_2_WORK - COST["work"]:
    print("That accounts for the whole difference: nothing else in the run changed.")

best_fitness = fitness_of(best_individual)
print(f"\nBest roster found: fitness {best_fitness}.")
if best_fitness == STEP_2_BEST:
    print(f"Step 2 ended at {STEP_2_BEST} as well. A cache returns the value the "
          "fitness function")
    print("would have returned, so an identical answer is the result to expect - and "
          "a")
    print("different one would mean the key was wrong, not that the cache was clever.")
else:
    print(f"Step 2 ended at {STEP_2_BEST}. The answer moved, so the cache key is "
          "wrong.")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].bar(["step 1", "step 2", "step 3"],
            [STEP_1_WORK_REF, STEP_2_WORK, COST["work"]],
            color=["tab:red", "tab:orange", "tab:green"])
axes[0].set_title("Work units for the same run")
axes[0].set_ylabel("work units")
for i, v in enumerate([STEP_1_WORK_REF, STEP_2_WORK, COST["work"]]):
    axes[0].text(i, v, str(v), ha="center", va="bottom")
axes[1].plot([r["gen"] for r in history], [r["built"] for r in history],
             "ko--", label="individuals built")
axes[1].plot([r["gen"] for r in history], [r["breed"] for r in history],
             "o-", color="tab:green", label="actually evaluated")
axes[1].set_title("The gap between the two lines is the cache")
axes[1].set_xlabel("generation")
axes[1].set_ylabel("count")
axes[1].legend()
axes[1].grid(True, linestyle=":", alpha=0.5)
fig.suptitle("Step 3: a hit is an evaluation that never happens")
fig.tight_layout()
FIGURES.mkdir(exist_ok=True)
fig.savefig(FIGURES / "shifts_03_the_cache.png", dpi=130)
plt.close(fig)
print("\nFigure saved to figures/shifts_03_the_cache.png")
# ------------------------------------------------------------------------------


# --- NEW (5) hit_rate_scan() --------------------------------------------------
def hit_rate_scan(probabilities: List[float]) -> List[Tuple[float, int, int]]:
    """Re-run the whole GA at several mutation rates and count cache hits.

    19 hits in 428 builds is a small return, and the honest question is whether
    that is a fact about caching or a fact about this search. Only a second
    measurement can answer it, so here is one.
    """
    global MUTATION_PROBABILITY
    rows = []
    for probability in probabilities:
        CACHE.clear()
        COST.update(calls=0, work=0, built=0, hits=0)
        MUTATION_PROBABILITY = probability
        run()
        rows.append((probability, COST["built"], COST["hits"]))
    return rows


print("\nThe same run, at four mutation probabilities. Everything else is held")
print("fixed, including the seed.\n")
print("  mutation p | individuals built | cache hits | hit rate")
print("  -----------+-------------------+------------+---------")
scan = hit_rate_scan([0.1, 0.25, 0.5, 0.75])
for probability, n_built, n_hits in scan:
    print(f"  {probability:10.2f} | {n_built:17d} | {n_hits:10d} |"
          f" {100 * n_hits / n_built:7.1f}%")
low_p, low_built, low_hits = scan[0]
high_p, high_built, high_hits = scan[-1]
print(f"\nAt p={low_p} the cache answers {100 * low_hits / low_built:.1f}% of builds; at "
      f"p={high_p} it answers {100 * high_hits / high_built:.1f}%.")
print("The dictionary is the same dictionary. What changes is how often the "
      "search")
print("comes back to a genome it has already seen, and that is a property of "
      "the")
print("operators, not of the cache. A cache is worth what the search's repeats "
      "are")
print("worth - which is why it has to be measured on the problem in hand.")
# ------------------------------------------------------------------------------

