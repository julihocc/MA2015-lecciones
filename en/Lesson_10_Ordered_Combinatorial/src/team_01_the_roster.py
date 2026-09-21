"""
Lesson 10 - Team 1: Eleven ordered slots
=========================================
NEW IN THIS STEP: load_pool(), role blocks, and a source-backed candidate pool.

The chromosome is an ordered list of eleven distinct player identifiers. Slot
order encodes one goalkeeper, four defenders, three midfielders and three
forwards, so membership and position must both survive every operator.

Run it:  python team_01_the_roster.py

The source-backed pool contains 12 candidates for each role block and an
explicit €600m budget.
"""
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd

SEED = 3
DATA = Path(__file__).with_name("players_20.csv")
FIGURES = Path(__file__).resolve().parent.parent / "figures"
ROLE_COUNTS = {"GK": 1, "DEF": 4, "MID": 3, "FWD": 3}
POOL_PER_ROLE = 12
BUDGET_EUR = 600_000_000


def primary_role(positions: str) -> str:
    """Map FIFA position codes onto one of four roster blocks.

    Args:
        positions: comma-separated FIFA codes; only the first is used.

    Returns:
        ``"GK"``, ``"DEF"``, ``"MID"`` or ``"FWD"``. Anything not a
        keeper, defender or midfielder is treated as a forward.

    Example:
        The pool keeps 12 candidates in each of those four blocks.
    """
    first = positions.split(",")[0].strip()
    if first == "GK":
        return "GK"
    if first in {"CB", "LB", "RB", "LWB", "RWB"}:
        return "DEF"
    if first in {"CM", "CDM", "CAM", "LM", "RM"}:
        return "MID"
    return "FWD"


def load_pool(path: Path = DATA) -> pd.DataFrame:
    """Load the source-backed candidate pool, 12 players per role.

    Args:
        path: ``players_20.csv`` beside this script.

    Returns:
        48 rows: 12 keepers, 12 defenders, 12 midfielders, 12 forwards,
        ranked by overall then cheaper first.

    Example:
        The pool feeds an eleven-slot roster under a €600m budget.
    """
    columns = ["sofifa_id", "short_name", "overall", "value_eur", "player_positions"]
    frame = pd.read_csv(path, usecols=columns).dropna(subset=["value_eur", "player_positions"])
    frame["role"] = frame["player_positions"].map(primary_role)
    pieces = []
    for role in ROLE_COUNTS:
        group = frame[frame["role"] == role].sort_values(
            ["overall", "value_eur"], ascending=[False, True]
        ).head(POOL_PER_ROLE)
        pieces.append(group)
    return pd.concat(pieces, ignore_index=True)


pool = load_pool()
summary: Dict[str, str] = {}
for role in ROLE_COUNTS:
    group = pool[pool.role == role]
    summary[role] = (f"{len(group)} candidates, overall {group.overall.min()}..{group.overall.max()}, "
                     f"value €{group.value_eur.min() / 1e6:.1f}m..€{group.value_eur.max() / 1e6:.1f}m")

print("Lesson 10 - Team 1: eleven ordered slots")
print(f"Roster slots: {ROLE_COUNTS}; total={sum(ROLE_COUNTS.values())}")
print(f"Budget: €{BUDGET_EUR / 1e6:.0f}m")
for role, text in summary.items():
    print(f"{role}: {text}")

FIGURES.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 5))
for role in ROLE_COUNTS:
    group = pool[pool.role == role]
    ax.scatter(group.value_eur / 1e6, group.overall, label=role, s=55)
ax.set(xlabel="market value (€m)", ylabel="overall rating",
       title="The candidate pool contains a price-skill tradeoff")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "team_01_the_roster.png", dpi=160)
plt.close(fig)
