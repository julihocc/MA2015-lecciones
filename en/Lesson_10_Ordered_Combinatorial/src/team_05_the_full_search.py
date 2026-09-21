"""
Lesson 10 - Team 5: The full roster search
===========================================
NEW IN THIS STEP: run(), an equal-budget baseline, and the selected roster.

CHANGES FROM team_04_ordered_operators.py
Introduce them in this order:
    1. run()                 repeat role-aware crossover, mutation, repair and elitism
    2. the baseline          compare against the same number of repaired random teams
    3. the roster            report names, roles, skill and price of the final answer

Run it:  python team_05_the_full_search.py

The GA reaches skill 991 at €597.5m and ties the best equal-evaluation
repaired-random roster.
"""
from pathlib import Path
import random
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd

SEED = 3
DATA = Path(__file__).with_name("players_20.csv")
FIGURES = Path(__file__).resolve().parent.parent / "figures"
ROLE_COUNTS = {"GK": 1, "DEF": 4, "MID": 3, "FWD": 3}
POOL_PER_ROLE = 12
BUDGET_EUR = 600_000_000
POPULATION_SIZE = 100
GENERATIONS = 50
OVERALL: List[int] = []
VALUE: List[int] = []
ROLE: List[str] = []
ROLE_INDEX: dict[str, List[int]] = {}


def primary_role(positions: str) -> str:
    """Map FIFA position codes onto one of four roster blocks.

    Args:
        positions: comma-separated FIFA codes; only the first is used.

    Returns:
        ``"GK"``, ``"DEF"``, ``"MID"`` or ``"FWD"``.
    """
    first = positions.split(",")[0].strip()
    if first == "GK": return "GK"
    if first in {"CB", "LB", "RB", "LWB", "RWB"}: return "DEF"
    if first in {"CM", "CDM", "CAM", "LM", "RM"}: return "MID"
    return "FWD"


def load_pool(path: Path = DATA) -> pd.DataFrame:
    """Load the pool and cache overall, value and role into module lists.

    Args:
        path: ``players_20.csv`` beside this script.

    Returns:
        48 rows, 12 per role. Side-effect: fills ``OVERALL``, ``VALUE``,
        ``ROLE`` and ``ROLE_INDEX`` so later calls avoid DataFrame lookups.
    """
    cols = ["sofifa_id", "short_name", "overall", "value_eur", "player_positions"]
    frame = pd.read_csv(path, usecols=cols).dropna(subset=["value_eur", "player_positions"])
    frame["role"] = frame.player_positions.map(primary_role)
    pool = pd.concat([frame[frame.role == role].sort_values(["overall", "value_eur"], ascending=[False, True]).head(POOL_PER_ROLE)
                      for role in ROLE_COUNTS], ignore_index=True)
    OVERALL[:] = [int(value) for value in pool.overall]
    VALUE[:] = [int(value) for value in pool.value_eur]
    ROLE[:] = pool.role.tolist()
    ROLE_INDEX.clear()
    ROLE_INDEX.update({role: [i for i, item_role in enumerate(ROLE) if item_role == role]
                       for role in ROLE_COUNTS})
    return pool


def random_team(pool: pd.DataFrame) -> List[int]:
    """Fill each ordered role block without duplicates.

    Args:
        pool: unused after ``load_pool`` has filled ``ROLE_INDEX``; kept
            so the signature matches earlier steps.

    Returns:
        Eleven distinct pool indices.
    """
    team: List[int] = []
    for role, count in ROLE_COUNTS.items():
        team.extend(random.sample(ROLE_INDEX[role], count))
    return team


def team_metrics(pool: pd.DataFrame, team: Sequence[int]) -> Tuple[int, int]:
    """Keep skill and price as separate evidence.

    Args:
        pool: unused after ``load_pool``; kept for signature stability.
        team: eleven pool indices.

    Returns:
        ``(skill, cost)``. The GA champion scores 991 at €597.5m.
    """
    return sum(OVERALL[i] for i in team), sum(VALUE[i] for i in team)


def repair_budget(pool: pd.DataFrame, source: Sequence[int]) -> List[int]:
    """Replace costly players within the same role block until the budget holds.

    Args:
        pool: unused after ``load_pool``; lookups go through ``VALUE``.
        source: an eleven-player roster that may overspend.

    Returns:
        A roster of eleven distinct players costing at most €600m.
    """
    team = list(source)
    while team_metrics(pool, team)[1] > BUDGET_EUR:
        options = []
        for slot, current in enumerate(team):
            for replacement in ROLE_INDEX[ROLE[current]]:
                if replacement in team:
                    continue
                saving = VALUE[current] - VALUE[replacement]
                loss = OVERALL[current] - OVERALL[replacement]
                if saving > 0:
                    options.append((loss / saving, -saving, slot, replacement))
        _, _, slot, replacement = min(options); team[slot] = int(replacement)
    return team


def role_crossover(pool: pd.DataFrame, first: Sequence[int], second: Sequence[int]) -> List[int]:
    """Recombine within blocks without duplicate players.

    Args:
        pool: unused after ``load_pool``.
        first, second: parent rosters with the same role layout.

    Returns:
        One child. Uniqueness is preserved; budget may break.
    """
    child: List[int] = []; offset = 0
    for role, count in ROLE_COUNTS.items():
        candidates = list(dict.fromkeys(list(first[offset:offset + count]) + list(second[offset:offset + count])))
        candidates += [i for i in ROLE_INDEX[role] if i not in candidates]
        child.extend(candidates[:count]); offset += count
    return child


def mutate(pool: pd.DataFrame, source: Sequence[int]) -> List[int]:
    """Replace one player with an unused player of the same role.

    Args:
        pool: unused after ``load_pool``.
        source: a roster, usually a child of crossover.

    Returns:
        A new eleven-player roster. Uniqueness holds; budget may break.
    """
    team = list(source); slot = random.randrange(len(team)); role = ROLE[team[slot]]
    team[slot] = random.choice([i for i in ROLE_INDEX[role] if i not in team]); return team


# --- NEW (1) run() ------------------------------------------------------------
def run(pool: pd.DataFrame) -> Tuple[List[int], List[int]]:
    """Fifty generations of role-aware crossover, mutation, repair and elitism.

    Args:
        pool: the candidate table, already cached into module lists.

    Returns:
        ``(best, history)``. History is best-so-far skill.

    Example:
        Skill 991 at €597.5m, tying the best equal-evaluation
        repaired-random roster.
    """
    random.seed(SEED); population = [repair_budget(pool, random_team(pool)) for _ in range(POPULATION_SIZE)]
    best = max(population, key=lambda t: team_metrics(pool, t)[0]); history = [team_metrics(pool, best)[0]]
    for _ in range(GENERATIONS):
        children = [best]
        while len(children) < POPULATION_SIZE:
            select = lambda: max(random.sample(population, 3), key=lambda t: team_metrics(pool, t)[0])
            child = role_crossover(pool, select(), select())
            if random.random() < 0.3: child = mutate(pool, child)
            children.append(repair_budget(pool, child))
        population = children; best = max(population, key=lambda t: team_metrics(pool, t)[0]); history.append(team_metrics(pool, best)[0])
    return best, history
# ------------------------------------------------------------------------------


# --- NEW (2) the baseline -----------------------------------------------------
pool = load_pool(); best, history = run(pool); evaluations = POPULATION_SIZE * (GENERATIONS + 1)
random.seed(SEED + 1); baseline = max((repair_budget(pool, random_team(pool)) for _ in range(evaluations)), key=lambda t: team_metrics(pool, t)[0])
# ------------------------------------------------------------------------------


# --- NEW (3) the roster -------------------------------------------------------
skill, cost = team_metrics(pool, best); baseline_skill, _ = team_metrics(pool, baseline)
selected = pool.loc[best, ["short_name", "role", "overall", "value_eur"]]
# ------------------------------------------------------------------------------

print("Lesson 10 - Team 5: the full roster search")
print(f"GA skill={skill}, cost=€{cost / 1e6:.1f}m; equal-budget random skill={baseline_skill}")
print(f"GA improvement over random: {skill - baseline_skill:+d} rating points")
print(f"Distinct players: {len(set(best))}/11; within budget: {cost <= BUDGET_EUR}")
print(selected.to_string(index=False))

FIGURES.mkdir(exist_ok=True); fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(history, label="best GA skill"); ax.axhline(baseline_skill, color="black", linestyle="--", label="best equal-budget random")
ax.set(xlabel="generation", ylabel="skill total", title="Roster search against an equal-evaluation baseline"); ax.legend()
fig.tight_layout(); fig.savefig(FIGURES / "team_05_the_full_search.png", dpi=160); plt.close(fig)

