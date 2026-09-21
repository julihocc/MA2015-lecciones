"""
Lección 09 - Radar 2: El cromosoma
==================================
NUEVO EN ESTE PASO: Individuo (Candidate), individuo_aleatorio(), y el censo de cobertura.

CAMBIOS RESPECTO A radar_01_el_paisaje.py
Introdúzcalos en este orden:
    1. Individuo                 mantiene conteo de radares y objetivos no cubiertos juntos
    2. individuo_aleatorio()     muestrea un plan de instalación
    3. el censo                  mide el balance (tradeoff) antes de asignar una aptitud

Ejecútalo:  python radar_02_el_cromosoma.py

21 de 500 planes aleatorios cubren todos los objetivos. Cobertura y número de radares son hechos separados: el paso 3 los ordenará.
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
        21 de 500 aleatorios cubren los 36. El resto deja huecos: eso es
        ``no_cubiertos`` del Individuo.
    """
    cubiertos: Set[int] = set()
    for indice_sitio, activo in enumerate(bits):
        if activo:
            cubiertos.update(i for i, objetivo in enumerate(OBJETIVOS)
                             if dist(SITIOS[indice_sitio], objetivo) <= RADIO)
    return cubiertos


# --- NUEVO (1) Individuo ------------------------------------------------------
@dataclass(frozen=True)
class Individuo:
    """Un plan de instalación: cuántos radares y cuántos huecos, juntos.

    Sin este objeto, "barato" y "cubre" viven en columnas distintas y la
    selección puede copiar un plan de un radar que deja 20 huecos.

    Args:
        bits: 9 decisiones de sitio.
        conteo_radares: suma de bits.
        no_cubiertos: 36 menos los cubiertos.

    Example:
        21 de 500 tienen ``no_cubiertos == 0``. Factible no significa
        óptimo: un plan de nueve radares también cubre todo.
    """
    bits: List[int]
    conteo_radares: int
    no_cubiertos: int

    @property
    def factible(self) -> bool:
        """True solo si no queda ningún objetivo fuera."""
        return self.no_cubiertos == 0
# ------------------------------------------------------------------------------


# --- NUEVO (2) individuo_aleatorio() ------------------------------------------
def crear_individuo(bits: List[int]) -> Individuo:
    """Calcula conteo y huecos. No repara: un hueco es un dato.

    Args:
        bits: 9 decisiones de sitio.

    Returns:
        Un Individuo, factible o no.

    Example:
        21 de 500 aleatorios salen factibles. El resto se queda con
        ``no_cubiertos > 0``.
    """
    return Individuo(bits, sum(bits), len(OBJETIVOS) - len(objetivos_cubiertos(bits)))


def individuo_aleatorio() -> Individuo:
    """Nueve monedas independientes. El 4.2% cubre todo por casualidad.

    Returns:
        Un plan aleatorio.

    Example:
        SEMILLA = 3, 500 sorteos: 21 cubren los 36 objetivos.
    """
    return crear_individuo([random.choice((0, 1)) for _ in SITIOS])
# ------------------------------------------------------------------------------


# --- NUEVO (3) el censo -------------------------------------------------------
random.seed(SEMILLA)
poblacion = [individuo_aleatorio() for _ in range(TAMANO_POBLACION)]
factibles = [individuo for individuo in poblacion if individuo.factible]
# ------------------------------------------------------------------------------

print("Lección 09 - Radar 2: el cromosoma")
print(f"Planes aleatorios con cobertura total: {len(factibles)}/{TAMANO_POBLACION}")
print(f"Conteos de radares muestreados: {min(c.conteo_radares for c in poblacion)}.."
      f"{max(c.conteo_radares for c in poblacion)}")
print(f"Mejor fallo de cobertura: {min(c.no_cubiertos for c in poblacion)} objetivos")

FIGURES.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter([c.conteo_radares for c in poblacion], [c.no_cubiertos for c in poblacion], alpha=0.35)
ax.set(xlabel="radares instalados", ylabel="objetivos no cubiertos",
       title="La cobertura y el conteo de instalaciones son hechos separados")
fig.tight_layout()
fig.savefig(FIGURES / "radar_02_el_cromosoma.png", dpi=160)
plt.close(fig)

