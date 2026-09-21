"""
Lesson 10 - Team 2: Random legal roles
=======================================
NEW IN THIS STEP: random_team(), team_metrics(), and a budget census.

CHANGES FROM team_01_the_roster.py
Introduce them in this order:
    1. random_team()         fill each ordered role block without duplicates
    2. team_metrics()        keep skill and price as separate evidence
    3. the census            measure budget feasibility before optimisation

Run it:  python team_02_random_rosters.py

All 1,000 rosters contain eleven distinct players; only 440 meet the
budget.
"""
from pathlib import Path
import random
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd

SEED = 3
DATA = Path(__file__).with_name("players_20.csv")
FIGURES = Path(__file__).resolve().parent.parent / "figures"
ROLE_COUNTS = {"GK": 1, "DEF": 4, "MID": 3, "FWD": 3}
POOL_PER_ROLE = 12
BUDGET_EUR = 600_000_000
POPULATION_SIZE = 1000


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
    columns = ["sofifa_id", "short_name", "overall", "value_eur", "player_positions"]
    frame = pd.read_csv(path, usecols=columns).dropna(subset=["value_eur", "player_positions"])
    frame["role"] = frame.player_positions.map(primary_role)
    return pd.concat([frame[frame.role == role].sort_values(
        ["overall", "value_eur"], ascending=[False, True]).head(POOL_PER_ROLE)
        for role in ROLE_COUNTS], ignore_index=True)


# --- NEW (1) random_team() ----------------------------------------------------
def random_team(pool: pd.DataFrame) -> List[int]:
    """Fill each ordered role block without duplicates.

    Args:
        pool: the 48-player candidate table.

    Returns:
        Eleven distinct pool indices: 1 GK, 4 DEF, 3 MID, 3 FWD. Budget
        is not checked.

    Example:
        All 1,000 such rosters have eleven distinct players; only 440
        meet the €600m budget.
    """
    team: List[int] = []
    for role, count in ROLE_COUNTS.items():
        candidates = pool.index[pool.role == role].tolist()
        team.extend(random.sample(candidates, count))
    return team
# ------------------------------------------------------------------------------


# --- NEW (2) team_metrics() ---------------------------------------------------
def team_metrics(pool: pd.DataFrame, team: Sequence[int]) -> Tuple[int, int]:
    """Keep skill and price as separate evidence.

    Args:
        pool: the candidate table.
        team: eleven pool indices.

    Returns:
        ``(skill, cost)``. Feasibility is ``cost <= 600_000_000``.

    Example:
        440 of 1,000 random teams are affordable.
    """
    chosen = pool.loc[list(team)]
    return int(chosen.overall.sum()), int(chosen.value_eur.sum())
# ------------------------------------------------------------------------------


# --- NEW (3) the census -------------------------------------------------------
random.seed(SEED)
pool = load_pool()
population = [random_team(pool) for _ in range(POPULATION_SIZE)]
metrics = [team_metrics(pool, team) for team in population]
feasible = [(skill, cost) for skill, cost in metrics if cost <= BUDGET_EUR]
# ------------------------------------------------------------------------------

print("Lesson 10 - Team 2: random legal roles")
print(f"Teams with eleven distinct players: {sum(len(set(t)) == 11 for t in population)}/{POPULATION_SIZE}")
print(f"Teams within budget: {len(feasible)}/{POPULATION_SIZE}")
print(f"Best affordable skill total: {max(skill for skill, _ in feasible)}")

FIGURES.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter([cost / 1e6 for _, cost in metrics], [skill for skill, _ in metrics], alpha=0.3)
ax.axvline(BUDGET_EUR / 1e6, color="tab:red", label="budget")
ax.set(xlabel="team value (€m)", ylabel="total overall rating",
       title="Role legality does not guarantee budget feasibility")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "team_02_random_rosters.png", dpi=160)
plt.close(fig)
