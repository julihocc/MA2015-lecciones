"""
Lección 09 - Radar 1: Un paisaje de cobertura
=============================================
NUEVO EN ESTE PASO: objetivos, sitios candidatos, y la relación de cobertura.

Cada bit decide si instalar un radar en un sitio candidato. Un radar
cubre un objetivo cuando su distancia euclidiana es a lo sumo el radio fijo.

Ejecútalo:  python radar_01_el_paisaje.py

Nueve sitios definen 512 planes y juntos cubren los 36 objetivos. El espacio es pequeño a propósito: el paso 5 podrá enumerarlo.
"""
from math import dist
from pathlib import Path
from typing import List, Sequence, Set, Tuple

import matplotlib.pyplot as plt

SEMILLA = 3
RADIO = 3.2
SITIOS = [(x, y) for y in (1, 5, 9) for x in (1, 5, 9)]
OBJETIVOS = [(x, y) for y in range(0, 11, 2) for x in range(0, 11, 2)]
FIGURES = Path(__file__).resolve().parent.parent / "figures"


def objetivos_cubiertos(bits: Sequence[int]) -> Set[int]:
    """Índices de objetivos a distancia ≤ 3.2 de algún radar encendido.

    Un bit no cubre un "área": cubre un conjunto de puntos. Sin esta
    función, el cromosoma sería solo un conteo de 1s.

    Args:
        bits: 9 bits, uno por sitio candidato.

    Returns:
        Conjunto de índices 0..35 cubiertos.

    Example:
        Los nueve sitios juntos cubren 36/36. Un solo sitio cubre entre
        el mínimo y el máximo impresos: ninguno cubre todo.
    """
    cubiertos: Set[int] = set()
    for indice_sitio, activo in enumerate(bits):
        if activo:
            cubiertos.update(i for i, objetivo in enumerate(OBJETIVOS)
                             if dist(SITIOS[indice_sitio], objetivo) <= RADIO)
    return cubiertos


cobertura_un_sitio = [len(objetivos_cubiertos([int(i == sitio) for i in range(len(SITIOS))]))
                      for sitio in range(len(SITIOS))]

print("Lección 09 - Radar 1: el paisaje de cobertura")
print(f"Sitios candidatos: {len(SITIOS)}; espacio de búsqueda binario: {2 ** len(SITIOS):,}")
print(f"Objetivos: {len(OBJETIVOS)}; radio: {RADIO}")
print(f"Objetivos cubiertos por un sitio: min={min(cobertura_un_sitio)}, "
      f"max={max(cobertura_un_sitio)}")
print(f"Todos los sitios juntos cubren {len(objetivos_cubiertos([1] * len(SITIOS)))}/{len(OBJETIVOS)} objetivos.")

FIGURES.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(*zip(*OBJETIVOS), marker="x", s=45, label="objetivos")
ax.scatter(*zip(*SITIOS), marker="^", s=90, label="sitios candidatos")
for indice, (x, y) in enumerate(SITIOS):
    ax.text(x + 0.15, y + 0.15, str(indice))
ax.set(xlim=(-1, 11), ylim=(-1, 11), aspect="equal", title="Sitios discretos de radar y objetivos de cobertura")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "radar_01_el_paisaje.png", dpi=160)
plt.close(fig)

