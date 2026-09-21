"""
Lesson 13 - Step 2: Calculate the fitness function once
========================================================
NEW IN THIS STEP: fitness is computed in the constructor and stored.

This is the book's section 13.1, and it is the best deal in the whole course.
An individual's genes do not change after it is built, so its fitness cannot
change either. Step 1 nevertheless recomputed it every time a value was needed
- once per individual inside the sort, twice more per individual in the
statistics - and paid a full 210-cell evaluation each time.

Move the evaluation into `__init__` and the program stops paying for what it
already knows. Nothing else changes: not the seed, not the operators, not the
order of the random draws, not the answer. Only the bill.

Lesson 02, step 1 counted the other half of this same idea. Its run cost 107
fitness evaluations rather than the obvious 110, because an individual that is
selected but neither crossed nor mutated is the *same object* and is never
rebuilt. Lesson 02 has the count; this is the technique that makes the count
the truth - once fitness lives on the object, "not rebuilt" and "not
re-evaluated" are the same sentence.

CHANGES FROM shifts_01_the_cost_of_a_run.py
Introduce them in this order:
    1. Individual        evaluate once, in the constructor, and store the value
    2. fitness_of()      reading a fitness becomes a lookup; no call site changes
    3. the_report        count the same run again and check the answer did not move

Run it:  python shifts_02_evaluate_once.py

428 calls, 89,880 work units: 54.0% less, with SELECT and STATS at 0.
Best fitness still −87 — the saving cost nothing.
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


# --- NEW (1) Individual -------------------------------------------------------
class Individual:
    """One candidate roster: 105 bits, employee-major, evaluated exactly once.

    The genes never change after construction, so the fitness cannot change
    either. Computing it here and storing it is the whole of section 13.1:
    three lines, no approximation, nothing given up.

    Args:
        gene_list: 105 bits, employee-major.

    Example:
        428 calls instead of 930: SELECT and STATS drop to 0, answer still -87.
    """

    def __init__(self, gene_list: List[int]) -> None:
        """Evaluate once and store the value.

        Args:
            gene_list: 105 bits, employee-major.
        """
        self.gene_list = list(gene_list)
        self.fitness = roster_fitness(self.gene_list)
        COST["built"] += 1
# ------------------------------------------------------------------------------


# --- NEW (2) fitness_of() -----------------------------------------------------
def fitness_of(ind: Individual) -> int:
    """Reading a fitness is now a lookup. Not one call site had to change.

    Args:
        ind: a candidate roster with ``ind.fitness`` already stored.

    Returns:
        The stored integer. SELECT and STATS cost 0 from step 2 on.
    """
    return ind.fitness
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
        A new list of the same length.
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
    """Ten generations from seed 3, evaluating once in the constructor.

    Returns:
        ``(best, history)``. SELECT and STATS are now 0.

    Example:
        428 calls, 89,880 work units (54.0% less), best fitness still -87.
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

# --- NEW (3) the_report -------------------------------------------------------
# Step 1 printed these three numbers. They are quoted here, not recomputed, so
# this script can say whether the change was free and whether it was safe.
STEP_1_CALLS = 930
STEP_1_WORK = 195300
STEP_1_BEST = -87

print("Lesson 13, step 2 - calculate the fitness function once")
print("=======================================================")
print("Fitness calls, split by the phase that made them.\n")
print(" gen | new individuals | SELECT | BREED | STATS | best | mean")
print("-----+-----------------+--------+-------+-------+------+-------")
for row in history:
    print(f" {row['gen']:3d} | {row['built']:15d} | {row['select']:6d} |"
          f" {row['breed']:5d} | {row['stats']:5d} | {row['best']:4d} |"
          f" {row['mean']:6.1f}")

built = COST["built"]
calls = COST["calls"]
select_total = sum(r["select"] for r in history)
breed_total = sum(r["breed"] for r in history)
stats_total = sum(r["stats"] for r in history)
print(f"\nThe bill has moved to BREED, which is where the new genomes are. "
      f"SELECT now costs")
print(f"{select_total} calls and STATS {stats_total}: reading a stored value is not an "
      "evaluation, so both")
print(f"columns are zero all the way down, and BREED pays {breed_total} - one call per "
      "individual")
print("it builds.")
print("\n                     step 1     step 2     saved")
print(f"  fitness calls    {STEP_1_CALLS:8d}   {calls:8d}   {100 * (STEP_1_CALLS - calls) / STEP_1_CALLS:5.1f}%")
print(f"  work units       {STEP_1_WORK:8d}   {COST['work']:8d}   {100 * (STEP_1_WORK - COST['work']) / STEP_1_WORK:5.1f}%")
print(f"\nThe run still builds {built} individuals - the same {built} - and now calls "
      f"the fitness")
print(f"function {calls} times, exactly once each. That is the floor for this design: "
      "no")
print("scheme in this lesson evaluates a brand-new genome fewer than one time.")

best_fitness = fitness_of(best_individual)
print(f"\nBest roster found: fitness {best_fitness}.")
if best_fitness == STEP_1_BEST:
    print(f"Step 1 ended at {STEP_1_BEST} too. Same seed, same answer: this saving was "
          "paid for with")
    print("nothing at all, which is why it belongs at the start of the lesson and not "
          "the end.")
else:
    print(f"Step 1 ended at {STEP_1_BEST}. The answer moved, so this refactor was NOT "
          "behaviour-preserving.")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].bar(["step 1", "step 2"], [STEP_1_CALLS, calls], color=["tab:red", "tab:green"])
axes[0].set_title("Fitness calls for the same run")
axes[0].set_ylabel("calls")
for i, v in enumerate([STEP_1_CALLS, calls]):
    axes[0].text(i, v, str(v), ha="center", va="bottom")
axes[1].plot([r["gen"] for r in history], [r["best"] for r in history], "o-", label="best")
axes[1].plot([r["gen"] for r in history], [r["mean"] for r in history], "s--", label="mean")
axes[1].set_title("The search itself is untouched")
axes[1].set_xlabel("generation")
axes[1].set_ylabel("fitness")
axes[1].legend()
axes[1].grid(True, linestyle=":", alpha=0.5)
fig.suptitle("Step 2: the same search, at one evaluation per individual")
fig.tight_layout()
FIGURES.mkdir(exist_ok=True)
fig.savefig(FIGURES / "shifts_02_evaluate_once.png", dpi=130)
plt.close(fig)
print("\nFigure saved to figures/shifts_02_evaluate_once.png")
# ------------------------------------------------------------------------------

