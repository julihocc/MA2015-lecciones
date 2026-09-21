"""
Lesson 13 - Step 1: What a run actually costs
==============================================
NEW IN THIS STEP: the roster problem, a plain genetic algorithm, and two
counters that price it - fitness calls and work units.

This is the last lesson of the course, and it is the only one about the cost of
running a genetic algorithm rather than about the algorithm itself. Everything
that follows is an attempt to make this run cheaper. So the first thing to build
is not an optimisation: it is the meter.

Two counters run through every script of this lesson.

  * COST["calls"] counts calls to the fitness function.
  * COST["work"]  counts the elementary steps those calls take - one increment
    per (employee, shift) cell the evaluation looks at.

Work units matter more than seconds. A second is a fact about the laptop that
ran the script; a work unit is a fact about the algorithm, and it is identical
on every machine, every Python and every run with this seed. Only one script in
this lesson prints a clock reading, and it says loudly what that reading is
worth.

The code below is written the way the problem is usually written first: the
fitness function is called wherever a fitness value is needed. That is the
defect this lesson opens with, and step 2 removes it.

Run it:  python shifts_01_the_cost_of_a_run.py

The run builds 428 individuals and calls fitness 930 times — 2.17
evaluations per individual, split SELECT 300 / BREED 0 / STATS 630.
195,300 work units, best fitness −87.
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


class Individual:
    """One candidate roster: 105 bits, employee-major.

    Note what this class does NOT do: it does not evaluate anything. Fitness is
    obtained by asking for it, and asking costs a full evaluation every time.

    Args:
        gene_list: 105 bits, employee-major.

    Example:
        428 individuals are built; asking for fitness 930 times is why the
        run costs 2.17 evaluations per individual.
    """

    def __init__(self, gene_list: List[int]) -> None:
        """Store genes. Evaluation is deferred until ``fitness()`` is asked.

        Args:
            gene_list: 105 bits, employee-major.
        """
        self.gene_list = list(gene_list)
        COST["built"] += 1

    def fitness(self) -> int:
        """Recompute the objective from scratch.

        Returns:
            ``roster_fitness`` of these genes. Every call is a paid evaluation.
        """
        return roster_fitness(self.gene_list)


def fitness_of(ind: Individual) -> int:
    """The one place the rest of the program reads a fitness value.

    Args:
        ind: a candidate roster.

    Returns:
        The stored or recomputed fitness. Step 1 recomputes; later
        steps look it up.
    """
    return ind.fitness()


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
        A new list of the same length. Intermediate children are evaluated
        later if mutation replaces them.
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
    """Ten generations from seed 3, with fitness asked wherever it is needed.

    Returns:
        ``(best, history)``. History rows split SELECT / BREED / STATS.

    Example:
        930 calls, 195,300 work units, split 300 / 0 / 630, best fitness -87.
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

print("Lesson 13, step 1 - the meter, before any optimisation")
print("=====================================================")
print(f"Roster problem: {EMPLOYEES} employees x {DAYS} days x {SHIFTS_PER_DAY} "
      f"shifts = {GENOME_LENGTH} bits.")
print(f"One fitness call walks {2 * EMPLOYEES * SHIFTS} cells, so one call "
      f"costs {2 * EMPLOYEES * SHIFTS} work units.\n")
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
print(f"\nBREED costs {breed_total} calls: building an individual evaluates "
      "nothing here, because")
print("this version has no place to keep the value. SELECT and STATS pay "
      f"instead - {select_total} and")
print(f"{stats_total} calls - for numbers the program could have written down once.")
print(f"\nThe run created {built} individuals and called the fitness function "
      f"{calls} times.")
print(f"That is {calls / built:.2f} evaluations per individual ever built, and "
      "every one of")
print("them past the first recomputes a number the program already knew.")
print(f"Total cost: {COST['work']} work units.")
print(f"\nA lower bound is one evaluation per individual: {built} calls, "
      f"{built * 2 * EMPLOYEES * SHIFTS} work units.")
print(f"The gap between {calls} and {built} is pure waste, and step 2 collects "
      "all of it.")
print(f"\nBest roster found: fitness {fitness_of(best_individual)} "
      f"(0 would be a roster with no violation at all).")

gens = [row["gen"] for row in history]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].bar(gens, [r["select"] for r in history], label="SELECT", color="tab:orange")
axes[0].bar(gens, [r["stats"] for r in history],
            bottom=[r["select"] for r in history], label="STATS", color="tab:red")
axes[0].plot(gens, [r["built"] for r in history], "ko--", label="individuals built")
axes[0].set_title("Where the fitness calls go")
axes[0].set_xlabel("generation")
axes[0].set_ylabel("fitness calls")
axes[0].legend()
axes[1].plot(gens, [r["best"] for r in history], "o-", label="best")
axes[1].plot(gens, [r["mean"] for r in history], "s--", label="mean")
axes[1].set_title("The search itself")
axes[1].set_xlabel("generation")
axes[1].set_ylabel("fitness")
axes[1].legend()
axes[1].grid(True, linestyle=":", alpha=0.5)
fig.suptitle("Step 1: the run costs far more evaluations than it has individuals")
fig.tight_layout()
FIGURES.mkdir(exist_ok=True)
fig.savefig(FIGURES / "shifts_01_the_cost_of_a_run.png", dpi=130)
plt.close(fig)
print(f"\nFigure saved to figures/shifts_01_the_cost_of_a_run.png")

