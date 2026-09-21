"""
Lección 09 - Radar 4: Operadores binarios
=========================================
NUEVO EN ESTE PASO: seleccion_torneo(), cruza(), y mutar().

CAMBIOS RESPECTO A radar_03_la_penalizacion.py
Introdúzcalos en este orden:
    1. seleccion_torneo()    selecciona usando el objetivo auditado
    2. cruza()               intercambia decisiones de sitio en un corte
    3. mutar()               invierte una decisión de instalación

Ejecútalo:  python radar_04_operadores.py

Una generación reduce el objetivo medio de 86.0 a 42.6. La selección usa la penalización auditada, no un conteo de radares a secas.
"""
from dataclasses import dataclass
from math import dist
from pathlib import Path
import random
from typing import List, Sequence, Set

import matplotlib.pyplot as plt

SEMILLA = 3
RADIO = 3.2
TAMANO_POBLACION = 100
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
        Cruza y mutación reconstruyen el Individuo con esta función:
        un bit volteado cambia los huecos, no solo el conteo.
    """
    cubiertos: Set[int] = set()
    for indice_sitio, activo in enumerate(bits):
        if activo:
            cubiertos.update(i for i, objetivo in enumerate(OBJETIVOS)
                             if dist(SITIOS[indice_sitio], objetivo) <= RADIO)
    return cubiertos


@dataclass(frozen=True)
class Individuo:
    """Un plan. El torneo lee ``objetivo``, no estos campos por separado.

    Args:
        bits: 9 decisiones de sitio.
        conteo_radares: suma de bits.
        no_cubiertos: objetivos fuera de radio.

    Example:
        Una generación baja el objetivo medio de 86.0 a 42.6.
    """
    bits: List[int]
    conteo_radares: int
    no_cubiertos: int


def crear_individuo(bits: List[int]) -> Individuo:
    """Calcula conteo y huecos.

    Args:
        bits: 9 decisiones de sitio.

    Returns:
        Un Individuo.

    Example:
        Cada hijo de cruza o mutación pasa por aquí antes de medirse.
    """
    return Individuo(bits, sum(bits), len(OBJETIVOS) - len(objetivos_cubiertos(bits)))


def individuo_aleatorio() -> Individuo:
    """Nueve monedas. Población inicial de 100, SEMILLA = 3.

    Returns:
        Un plan aleatorio.

    Example:
        Objetivo medio de los padres: 86.0.
    """
    return crear_individuo([random.choice((0, 1)) for _ in SITIOS])


def objetivo(individuo: Individuo) -> int:
    """Penalización lexicográfica: un hueco pesa más que apagar todos los radares.

    Args:
        individuo: plan con conteo y huecos.

    Returns:
        ``10 * no_cubiertos + conteo_radares``.

    Example:
        El torneo minimiza esto. Media 86.0 → 42.6 en una generación.
    """
    return individuo.no_cubiertos * (len(SITIOS) + 1) + individuo.conteo_radares


# --- NUEVO (1) seleccion_torneo() ---------------------------------------------
def seleccion_torneo(poblacion: List[Individuo], tamano: int = 3) -> Individuo:
    """Minimiza ``objetivo`` entre tres. Un factible siempre vence a un infactible.

    Args:
        poblacion: planes actuales.
        tamano: contendientes; 3.

    Returns:
        El de menor objetivo.

    Example:
        Una generación con esta presión baja la media de 86.0 a 42.6.
    """
    return min(random.sample(poblacion, tamano), key=objetivo)
# ------------------------------------------------------------------------------


# --- NUEVO (2) cruza() --------------------------------------------------------
def cruza(primero: Individuo, segundo: Individuo) -> Individuo:
    """Un punto de corte entre sitios. No repara: un hueco se hereda o se crea.

    Args:
        primero: primer padre.
        segundo: segundo padre.

    Returns:
        Un hijo, factible o no.

    Example:
        Combinada con mutación, 100 hijos bajan la media a 42.6.
    """
    corte = random.randrange(1, len(SITIOS))
    return crear_individuo(primero.bits[:corte] + segundo.bits[corte:])
# ------------------------------------------------------------------------------


# --- NUEVO (3) mutar() --------------------------------------------------------
def mutar(individuo: Individuo) -> Individuo:
    """Invierte un sitio. Puede abrir un hueco o apagar un radar de más.

    Args:
        individuo: plan de partida.

    Returns:
        Un Individuo con un bit volteado.

    Example:
        Un flip es barato en 9 bits: suficiente para explorar, no para
        destruir la presión de una generación.
    """
    bits = individuo.bits.copy()
    indice = random.randrange(len(bits))
    bits[indice] = 1 - bits[indice]
    return crear_individuo(bits)
# ------------------------------------------------------------------------------


random.seed(SEMILLA)
poblacion = [individuo_aleatorio() for _ in range(TAMANO_POBLACION)]
hijos = [mutar(cruza(seleccion_torneo(poblacion), seleccion_torneo(poblacion)))
         for _ in range(TAMANO_POBLACION)]

print("Lección 09 - Radar 4: operadores binarios")
print(f"Objetivo medio de los padres: {sum(map(objetivo, poblacion)) / TAMANO_POBLACION:.1f}")
print(f"Objetivo medio de los hijos:  {sum(map(objetivo, hijos)) / TAMANO_POBLACION:.1f}")
print(f"Padres con cobertura total: {sum(c.no_cubiertos == 0 for c in poblacion)}")
print(f"Hijos con cobertura total: {sum(c.no_cubiertos == 0 for c in hijos)}")

FIGURES.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(7, 5))
ax.boxplot([[objetivo(c) for c in poblacion], [objetivo(c) for c in hijos]],
           tick_labels=["padres", "hijos"])
ax.set(ylabel="objetivo", title="La selección desplaza el objetivo en una generación")
fig.tight_layout()
fig.savefig(FIGURES / "radar_04_operadores.png", dpi=160)
plt.close(fig)

