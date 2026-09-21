"""
Lesson 10 - Team 4: Ordered operators
======================================
NEW IN THIS STEP: role_crossover(), mutate(), and one selected generation.

CHANGES FROM team_03_budget_repair.py
Introduce them in this order:
    1. role_crossover()      recombine within blocks without duplicate players
    2. mutate()              replace one player with an unused player of the same role
    3. one generation        measure quality while repair preserves the budget

Run it:  python team_04_ordered_operators.py

All 100 children remain legal; mean skill rises from 979.6 to 985.3.
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
    """Load the source-backed candidate pool, 12 players per role.

    Args:
        path: ``players_20.csv`` beside this script.

    Returns:
        48 rows, 12 per role block.
    """
    cols = ["sofifa_id", "short_name", "overall", "value_eur", "player_positions"]
    frame = pd.read_csv(path, usecols=cols).dropna(subset=["value_eur", "player_positions"])
    frame["role"] = frame.player_positions.map(primary_role)
    return pd.concat([frame[frame.role == role].sort_values(["overall", "value_eur"], ascending=[False, True]).head(POOL_PER_ROLE)
                      for role in ROLE_COUNTS], ignore_index=True)


def random_team(pool: pd.DataFrame) -> List[int]:
    """Fill each ordered role block without duplicates.

    Args:
        pool: the 48-player candidate table.

    Returns:
        Eleven distinct pool indices. Budget is not checked.
    """
    team: List[int] = []
    for role, count in ROLE_COUNTS.items(): team.extend(random.sample(pool.index[pool.role == role].tolist(), count))
    return team


def team_metrics(pool: pd.DataFrame, team: Sequence[int]) -> Tuple[int, int]:
    """Keep skill and price as separate evidence.

    Args:
        pool: the candidate table.
        team: eleven pool indices.

    Returns:
        ``(skill, cost)``.

    Example:
        Mean skill rises from 979.6 to 985.3 after one operator pass.
    """
    chosen = pool.loc[list(team)]; return int(chosen.overall.sum()), int(chosen.value_eur.sum())


def repair_budget(pool: pd.DataFrame, source: Sequence[int]) -> List[int]:
    """Replace costly players within the same role block until the budget holds.

    Args:
        pool: the candidate table.
        source: an eleven-player roster that may overspend.

    Returns:
        A roster of eleven distinct players costing at most €600m.
    """
    team = list(source)
    while team_metrics(pool, team)[1] > BUDGET_EUR:
        options = []
        for slot, current in enumerate(team):
            role = pool.loc[current, "role"]
            for replacement in pool.index[(pool.role == role) & (~pool.index.isin(team))]:
                saving = int(pool.loc[current, "value_eur"] - pool.loc[replacement, "value_eur"])
                loss = int(pool.loc[current, "overall"] - pool.loc[replacement, "overall"])
                if saving > 0: options.append((loss / saving, -saving, slot, replacement))
        _, _, slot, replacement = min(options); team[slot] = int(replacement)
    return team


# --- NEW (1) role_crossover() -------------------------------------------------
def role_crossover(pool: pd.DataFrame, first: Sequence[int], second: Sequence[int]) -> List[int]:
    """Recombine within blocks without duplicate players.

    Args:
        pool: the candidate table, used to fill a block if both parents
            share too few distinct players of that role.
        first, second: parent rosters with the same role layout.

    Returns:
        One child. Uniqueness is preserved; budget may break.

    Example:
        All 100 children remain legal after repair; mean skill is 985.3.
    """
    child: List[int] = []; offset = 0
    for role, count in ROLE_COUNTS.items():
        candidates = list(dict.fromkeys(list(first[offset:offset + count]) + list(second[offset:offset + count])))
        candidates += [int(i) for i in pool.index[pool.role == role] if i not in candidates]
        child.extend(candidates[:count]); offset += count
    return child
# ------------------------------------------------------------------------------


# --- NEW (2) mutate() ---------------------------------------------------------
def mutate(pool: pd.DataFrame, source: Sequence[int]) -> List[int]:
    """Replace one player with an unused player of the same role.

    Args:
        pool: the candidate table.
        source: a roster, usually a child of crossover.

    Returns:
        A new eleven-player roster. The swapped-in player is not already
        on the team, so uniqueness holds; budget may break.
    """
    team = list(source); slot = random.randrange(len(team)); role = pool.loc[team[slot], "role"]
    choices = [int(i) for i in pool.index[pool.role == role] if i not in team]
    team[slot] = random.choice(choices); return team
# ------------------------------------------------------------------------------


# --- NEW (3) one generation ---------------------------------------------------
random.seed(SEED); pool = load_pool()
parents = [repair_budget(pool, random_team(pool)) for _ in range(POPULATION_SIZE)]
select = lambda: max(random.sample(parents, 3), key=lambda team: team_metrics(pool, team)[0])
children = [repair_budget(pool, mutate(pool, role_crossover(pool, select(), select()))) for _ in range(POPULATION_SIZE)]
# ------------------------------------------------------------------------------

print("Lesson 10 - Team 4: ordered operators")
print(f"Legal children: {sum(len(set(t)) == 11 and team_metrics(pool, t)[1] <= BUDGET_EUR for t in children)}/{POPULATION_SIZE}")
print(f"Mean parent skill: {sum(team_metrics(pool, t)[0] for t in parents) / POPULATION_SIZE:.1f}")
print(f"Mean child skill:  {sum(team_metrics(pool, t)[0] for t in children) / POPULATION_SIZE:.1f}")

FIGURES.mkdir(exist_ok=True); fig, ax = plt.subplots(figsize=(7, 4))
ax.boxplot([[team_metrics(pool, t)[0] for t in parents], [team_metrics(pool, t)[0] for t in children]], tick_labels=["parents", "children"])
ax.set(ylabel="skill total", title="Role-aware operators preserve a legal ordered roster")
fig.tight_layout(); fig.savefig(FIGURES / "team_04_ordered_operators.png", dpi=160); plt.close(fig)
