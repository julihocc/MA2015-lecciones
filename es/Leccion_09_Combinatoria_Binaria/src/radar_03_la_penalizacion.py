"""
Lección 09 - Radar 3: Una penalización lexicográfica
====================================================
NUEVO EN ESTE PASO: objetivo() y una auditoría de clasificación.

CAMBIOS RESPECTO A radar_02_el_cromosoma.py
Introdúzcalos en este orden:
    1. objetivo()              un objetivo perdido debe pesar más que el ahorro de cualquier radar
    2. la auditoría de clas.   verifica que la factibilidad realmente va primero

Ejecútalo:  python radar_03_la_penalizacion.py

La penalización auditada clasifica a cada plan factible por encima de cada plan no factible. Un radar ahorrado no puede compensar un objetivo perdido.
"""
from dataclasses import dataclass
from math import dist
from pathlib import Path
import random
from typing import List, Sequence, Set

import matplotlib.pyplot as plt

SEMILLA = 3
RADIO = 3.2
TAMANO_POBLACION = 500
SITIOS = [(x, y) for y in (1, 5, 9) for x in (1, 5, 9)]
OBJETIVOS = [(x, y) for y in range(0, 11, 2) for x in range(0, 11, 2)]
FIGURES = Path(__file__).resolve().parent.parent / "figures"


def objetivos_cubiertos(bits: Sequence[int]) -> Set[int]:
    """Índices cubiertos por los radares encendidos.

    Args:
        bits: 9 bits, uno por sitio.

    Returns:
        Conjunto de objetivos cubiertos.

    Example:
        ``no_cubiertos`` se deriva de aquí. Un hueco cuesta 10 en
        ``objetivo``, más que apagar los 9 radares.
    """
    cubiertos: Set[int] = set()
    for indice_sitio, activo in enumerate(bits):
        if activo:
            cubiertos.update(i for i, objetivo in enumerate(OBJETIVOS)
                             if dist(SITIOS[indice_sitio], objetivo) <= RADIO)
    return cubiertos


@dataclass(frozen=True)
class Individuo:
    """Un plan: radares y huecos. ``objetivo`` los combina sin mezclarlos.

    Args:
        bits: 9 decisiones de sitio.
        conteo_radares: suma de bits.
        no_cubiertos: objetivos fuera de radio.

    Example:
        La auditoría imprime que todo factible vence a todo infactible.
    """
    bits: List[int]
    conteo_radares: int
    no_cubiertos: int

    @property
    def factible(self) -> bool:
        """True solo si no queda ningún objetivo fuera."""
        return self.no_cubiertos == 0


def crear_individuo(bits: List[int]) -> Individuo:
    """Calcula conteo y huecos.

    Args:
        bits: 9 decisiones de sitio.

    Returns:
        Un Individuo.

    Example:
        500 aleatorios alimentan la auditoría de clasificación.
    """
    return Individuo(bits, sum(bits), len(OBJETIVOS) - len(objetivos_cubiertos(bits)))


def individuo_aleatorio() -> Individuo:
    """Nueve monedas. La muestra tiene factibles e infactibles a la vez.

    Returns:
        Un plan aleatorio.

    Example:
        SEMILLA = 3, 500 sorteos: la auditoría necesita ambos bandos.
    """
    return crear_individuo([random.choice((0, 1)) for _ in SITIOS])


# --- NUEVO (1) objetivo() -----------------------------------------------------
def objetivo(individuo: Individuo) -> int:
    """Menor es mejor; la cobertura domina sobre el conteo de radares por construcción.

    Un hueco cuesta ``len(SITIOS)+1 = 10``. Apagar los 9 radares ahorra 9.
    Por eso ningún infactible puede ganar a un factible: la penalización
    es lexicográfica, no un "peso" que se afina.

    Args:
        individuo: plan con conteo y huecos.

    Returns:
        ``10 * no_cubiertos + conteo_radares``.

    Example:
        En 500 aleatorios, todo factible queda por encima de todo
        infactible. La auditoría imprime True.
    """
    return individuo.no_cubiertos * (len(SITIOS) + 1) + individuo.conteo_radares
# ------------------------------------------------------------------------------


# --- NUEVO (2) la auditoría de clasificación ----------------------------------
random.seed(SEMILLA)
poblacion = [individuo_aleatorio() for _ in range(TAMANO_POBLACION)]
clasificados = sorted(poblacion, key=objetivo)
mejor = clasificados[0]
mejor_no_factible = min((c for c in poblacion if not c.factible), key=objetivo)
# ------------------------------------------------------------------------------

print("Lección 09 - Radar 3: la penalización")
print(f"Mejor plan: radares={mejor.conteo_radares}, no_cubiertos={mejor.no_cubiertos}, objetivo={objetivo(mejor)}")
print(f"Mejor plan no factible: radares={mejor_no_factible.conteo_radares}, "
      f"no_cubiertos={mejor_no_factible.no_cubiertos}, objetivo={objetivo(mejor_no_factible)}")
print(f"Todo plan factible supera a todo plan no factible: "
      f"{max(objetivo(c) for c in poblacion if c.factible) < min(objetivo(c) for c in poblacion if not c.factible)}")

FIGURES.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter([c.conteo_radares for c in poblacion], [objetivo(c) for c in poblacion],
           c=[c.no_cubiertos for c in poblacion], cmap="viridis", alpha=0.45)
ax.set(xlabel="radares instalados", ylabel="objetivo",
       title="La penalización ordena la cobertura antes que la economía")
fig.tight_layout()
fig.savefig(FIGURES / "radar_03_la_penalizacion.png", dpi=160)
plt.close(fig)

