"""
Lesson 09 - Radar 1: A coverage landscape
==========================================
NEW IN THIS STEP: targets, candidate sites, and the coverage relation.

Each bit decides whether to install a radar at one candidate site. A radar
covers a target when their Euclidean distance is at most the fixed radius.

Run it:  python radar_01_the_landscape.py

Nine sites define 512 plans and jointly cover all 36 targets.
"""
from math import dist
from pathlib import Path
from typing import List, Sequence, Set, Tuple

import matplotlib.pyplot as plt

SEED = 3
RADIUS = 3.2
SITES = [(x, y) for y in (1, 5, 9) for x in (1, 5, 9)]
TARGETS = [(x, y) for y in range(0, 11, 2) for x in range(0, 11, 2)]
FIGURES = Path(__file__).resolve().parent.parent / "figures"


def covered_targets(bits: Sequence[int]) -> Set[int]:
    """Targets within radius 3.2 of at least one active site.

    Args:
        bits: length-9 installation plan, one bit per candidate site.

    Returns:
        Indices of covered targets. Full coverage is a set of size 36.

    Example:
        All nine sites together cover all 36 targets; the search space is
        512 plans.
    """
    covered: Set[int] = set()
    for site_index, active in enumerate(bits):
        if active:
            covered.update(i for i, target in enumerate(TARGETS)
                           if dist(SITES[site_index], target) <= RADIUS)
    return covered


single_site_coverage = [len(covered_targets([int(i == site) for i in range(len(SITES))]))
                        for site in range(len(SITES))]

print("Lesson 09 - Radar 1: the coverage landscape")
print(f"Candidate sites: {len(SITES)}; binary search space: {2 ** len(SITES):,}")
print(f"Targets: {len(TARGETS)}; radius: {RADIUS}")
print(f"Targets covered by one site: min={min(single_site_coverage)}, "
      f"max={max(single_site_coverage)}")
print(f"All sites together cover {len(covered_targets([1] * len(SITES)))}/{len(TARGETS)} targets.")

FIGURES.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(*zip(*TARGETS), marker="x", s=45, label="targets")
ax.scatter(*zip(*SITES), marker="^", s=90, label="candidate sites")
for index, (x, y) in enumerate(SITES):
    ax.text(x + 0.15, y + 0.15, str(index))
ax.set(xlim=(-1, 11), ylim=(-1, 11), aspect="equal", title="Discrete radar sites and coverage targets")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "radar_01_the_landscape.png", dpi=160)
plt.close(fig)
