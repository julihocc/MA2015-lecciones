"""
Lesson 10 - Team 3: Budget repair
==================================
NEW IN THIS STEP: repair_budget() and a before/after audit.

CHANGES FROM team_02_random_rosters.py
Introduce them in this order:
    1. repair_budget()       replace costly players within the same role block
    2. the audit             prove price changes without duplicates or role drift

Run it:  python team_03_budget_repair.py

Repair makes 1,000/1,000 teams affordable and preserves uniqueness.
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


def random_team(pool: pd.DataFrame) -> List[int]:
    """Fill each ordered role block without duplicates.

    Args:
        pool: the 48-player candidate table.

    Returns:
        Eleven distinct pool indices. Budget is not checked.
    """
    team: List[int] = []
    for role, count in ROLE_COUNTS.items():
        team.extend(random.sample(pool.index[pool.role == role].tolist(), count))
    return team


def team_metrics(pool: pd.DataFrame, team: Sequence[int]) -> Tuple[int, int]:
    """Keep skill and price as separate evidence.

    Args:
        pool: the candidate table.
        team: eleven pool indices.

    Returns:
        ``(skill, cost)``.
    """
    chosen = pool.loc[list(team)]
    return int(chosen.overall.sum()), int(chosen.value_eur.sum())


# --- NEW (1) repair_budget() --------------------------------------------------
def repair_budget(pool: pd.DataFrame, source: Sequence[int]) -> List[int]:
    """Replace costly players within the same role block until the budget holds.

    Args:
        pool: the candidate table.
        source: an eleven-player roster that may overspend.

    Returns:
        A roster of eleven distinct players costing at most €600m. Roles
        do not drift; uniqueness is preserved.

    Example:
        1,000/1,000 teams become affordable; uniqueness stays 1,000/1,000.
    """
    team = list(source)
    while team_metrics(pool, team)[1] > BUDGET_EUR:
        options = []
        for slot, current in enumerate(team):
            role = pool.loc[current, "role"]
            for replacement in pool.index[(pool.role == role) & (~pool.index.isin(team))]:
                saving = int(pool.loc[current, "value_eur"] - pool.loc[replacement, "value_eur"])
                skill_loss = int(pool.loc[current, "overall"] - pool.loc[replacement, "overall"])
                if saving > 0:
                    options.append((skill_loss / saving, -saving, slot, replacement))
        if not options:
            raise RuntimeError("Candidate pool cannot satisfy the budget")
        _, _, slot, replacement = min(options)
        team[slot] = int(replacement)
    return team
# ------------------------------------------------------------------------------


# --- NEW (2) the audit --------------------------------------------------------
random.seed(SEED)
pool = load_pool()
raw = [random_team(pool) for _ in range(POPULATION_SIZE)]
repaired = [repair_budget(pool, team) for team in raw]
raw_metrics = [team_metrics(pool, team) for team in raw]
repaired_metrics = [team_metrics(pool, team) for team in repaired]
# ------------------------------------------------------------------------------

print("Lesson 10 - Team 3: budget repair")
print(f"Within budget before: {sum(cost <= BUDGET_EUR for _, cost in raw_metrics)}/{POPULATION_SIZE}")
print(f"Within budget after:  {sum(cost <= BUDGET_EUR for _, cost in repaired_metrics)}/{POPULATION_SIZE}")
print(f"Distinct after repair: {sum(len(set(team)) == 11 for team in repaired)}/{POPULATION_SIZE}")
print(f"Mean skill change: {sum(a[0] - b[0] for a, b in zip(repaired_metrics, raw_metrics)) / POPULATION_SIZE:+.2f}")

FIGURES.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter([cost / 1e6 for _, cost in raw_metrics], [skill for skill, _ in raw_metrics], alpha=0.2, label="before")
ax.scatter([cost / 1e6 for _, cost in repaired_metrics], [skill for skill, _ in repaired_metrics], alpha=0.2, label="after")
ax.axvline(BUDGET_EUR / 1e6, color="tab:red", label="budget")
ax.set(xlabel="team value (€m)", ylabel="skill total", title="Repair enforces price while preserving role slots")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "team_03_budget_repair.png", dpi=160)
plt.close(fig)
