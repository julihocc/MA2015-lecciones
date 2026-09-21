"""
Lección 09 - Radar 5: La búsqueda completa
==========================================
NUEVO EN ESTE PASO: optimo_exacto(), ejecutar(), y el mapa de cobertura.

CAMBIOS RESPECTO A radar_04_operadores.py
Introdúzcalos en este orden:
    1. optimo_exacto()       enumera esta pequeña instancia como referencia
    2. ejecutar()            repite cruza seleccionada y mutación con elitismo
    3. el mapa de cobertura  muestra la geometría final, no solo su puntuación

Ejecútalo:  python radar_05_la_busqueda_completa.py

El AG y la enumeración exhaustiva encuentran cinco radares con cobertura completa. 512 planes caben en una clase; esa coincidencia es verificable, no un acto de fe.
"""
from dataclasses import dataclass
from itertools import product
from math import dist
from pathlib import Path
import random
from typing import List, Sequence, Set, Tuple

import matplotlib.pyplot as plt

SEMILLA = 3
RADIO = 3.2
TAMANO_POBLACION = 100
GENERACIONES = 30
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
        El campeón de cinco radares cubre 36/36. El mapa del paso 5 lo
        dibuja, no solo lo puntúa.
    """
    cubiertos: Set[int] = set()
    for indice_sitio, activo in enumerate(bits):
        if activo:
            cubiertos.update(i for i, objetivo in enumerate(OBJETIVOS)
                             if dist(SITIOS[indice_sitio], objetivo) <= RADIO)
    return cubiertos


@dataclass(frozen=True)
class Individuo:
    """Un plan. El AG y la enumeración se comparan por ``objetivo``.

    Args:
        bits: 9 decisiones de sitio.
        conteo_radares: suma de bits.
        no_cubiertos: objetivos fuera de radio.

    Example:
        AG y referencia exacta: 5 radares, 0 huecos.
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
        Los 512 planes de ``optimo_exacto`` pasan por aquí.
    """
    return Individuo(bits, sum(bits), len(OBJETIVOS) - len(objetivos_cubiertos(bits)))


def individuo_aleatorio() -> Individuo:
    """Nueve monedas. Población inicial de 100, SEMILLA = 3.

    Returns:
        Un plan aleatorio.

    Example:
        El AG llega a 5 radares desde esta muestra, no desde el óptimo.
    """
    return crear_individuo([random.choice((0, 1)) for _ in SITIOS])


def objetivo(individuo: Individuo) -> int:
    """Penalización lexicográfica: cobertura primero, economía después.

    Args:
        individuo: plan con conteo y huecos.

    Returns:
        ``10 * no_cubiertos + conteo_radares``.

    Example:
        AG y enumeración coinciden: 5 radares, 0 huecos, objetivo 5.
    """
    return individuo.no_cubiertos * (len(SITIOS) + 1) + individuo.conteo_radares


def seleccion_torneo(poblacion: List[Individuo], tamano: int = 3) -> Individuo:
    """Minimiza ``objetivo`` entre tres.

    Args:
        poblacion: planes actuales.
        tamano: contendientes; 3.

    Returns:
        El de menor objetivo.

    Example:
        30 generaciones con esta presión coinciden con la enumeración.
    """
    return min(random.sample(poblacion, tamano), key=objetivo)


def cruza(primero: Individuo, segundo: Individuo) -> Individuo:
    """Un punto de corte entre sitios.

    Args:
        primero: primer padre.
        segundo: segundo padre.

    Returns:
        Un hijo.

    Example:
        9 bits: un corte basta para mezclar dos planes de 5 radares.
    """
    corte = random.randrange(1, len(SITIOS))
    return crear_individuo(primero.bits[:corte] + segundo.bits[corte:])


def mutar(individuo: Individuo) -> Individuo:
    """Con probabilidad 0.35 invierte un sitio. En 9 bits, eso es un empujón suave.

    Args:
        individuo: plan de partida.

    Returns:
        El mismo plan o uno con un bit volteado.

    Example:
        Suficiente para salir de un plan de 6 radares hacia uno de 5.
    """
    bits = individuo.bits.copy()
    if random.random() < 0.35:
        indice = random.randrange(len(bits)); bits[indice] = 1 - bits[indice]
    return crear_individuo(bits)


# --- NUEVO (1) optimo_exacto() ------------------------------------------------
def optimo_exacto() -> Individuo:
    """Enumera los 512 planes. Esta instancia cabe en una clase; por eso se eligió.

    Returns:
        El plan de menor ``objetivo``: 5 radares, 0 huecos.

    Example:
        El AG coincide con esta referencia. En la mochila la enumeración
        también era posible; aquí 2^9 = 512 es todavía más barato.
    """
    individuos = (crear_individuo(list(bits)) for bits in product((0, 1), repeat=len(SITIOS)))
    return min(individuos, key=objetivo)
# ------------------------------------------------------------------------------


# --- NUEVO (2) ejecutar() -----------------------------------------------------
def ejecutar() -> Tuple[Individuo, List[int]]:
    """30 generaciones desde SEMILLA = 3, con elitismo.

    Returns:
        ``(campeón, historia de objetivos)``.

    Example:
        Cinco radares, cobertura completa, mismo objetivo que la
        enumeración.
    """
    random.seed(SEMILLA)
    poblacion = [individuo_aleatorio() for _ in range(TAMANO_POBLACION)]
    mejor = min(poblacion, key=objetivo)
    historia = [objetivo(mejor)]
    for _ in range(GENERACIONES):
        hijos = [mejor]
        while len(hijos) < TAMANO_POBLACION:
            hijo = mutar(cruza(seleccion_torneo(poblacion), seleccion_torneo(poblacion)))
            hijos.append(hijo)
        poblacion = hijos
        mejor = min((mejor, min(poblacion, key=objetivo)), key=objetivo)
        historia.append(objetivo(mejor))
    return mejor, historia
# ------------------------------------------------------------------------------


mejor, historia = ejecutar()
referencia = optimo_exacto()
print("Lección 09 - Radar 5: la búsqueda completa")
print(f"Resultado del AG: radares={mejor.conteo_radares}, no_cubiertos={mejor.no_cubiertos}, objetivo={objetivo(mejor)}")
print(f"Referencia exacta: radares={referencia.conteo_radares}, no_cubiertos={referencia.no_cubiertos}, "
      f"objetivo={objetivo(referencia)}")
print(f"AG coincide con el objetivo exacto: {objetivo(mejor) == objetivo(referencia)}")
print(f"Evaluaciones de individuos: {TAMANO_POBLACION * (GENERACIONES + 1):,}")

# --- NUEVO (3) el mapa de cobertura -------------------------------------------
FIGURES.mkdir(exist_ok=True)
fig, (izq, der) = plt.subplots(1, 2, figsize=(11, 5))
izq.step(range(len(historia)), historia, where="post")
izq.axhline(objetivo(referencia), color="black", linestyle="--")
izq.set(xlabel="generación", ylabel="objetivo", title="Historia de búsqueda")
sitios_activos = [sitio for bit, sitio in zip(mejor.bits, SITIOS) if bit]
der.scatter(*zip(*OBJETIVOS), marker="x", s=45, label="objetivos")
der.scatter(*zip(*sitios_activos), marker="^", s=100, label="radares seleccionados")
for x, y in sitios_activos:
    der.add_patch(plt.Circle((x, y), RADIO, fill=False, alpha=0.25))
der.set(xlim=(-1, 11), ylim=(-1, 11), aspect="equal", title="Geometría de cobertura final")
der.legend()
fig.tight_layout()
fig.savefig(FIGURES / "radar_05_la_busqueda_completa.png", dpi=160)
plt.close(fig)
# ------------------------------------------------------------------------------

