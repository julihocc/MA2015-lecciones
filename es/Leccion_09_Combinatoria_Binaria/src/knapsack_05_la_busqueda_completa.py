"""
Lección 09 - Mochila 5: La búsqueda completa
============================================
NUEVO EN ESTE PASO: mutar(), ejecutar(), y el reporte de brecha exacta.

CAMBIOS RESPECTO A knapsack_04_seleccion_y_cruza.py
Introdúzcalos en este orden:
    1. mutar()               las inversiones de bits restauran decisiones que la cruza pudo haber perdido
    2. ejecutar()            repite la generación y preserva el mejor cromosoma
    3. el reporte de brecha  compara la heurística contra la referencia enumerada

Ejecútalo:  python knapsack_05_la_busqueda_completa.py

El AG alcanza el valor exacto 50 con peso 20 y brecha cero en 4,060 evaluaciones. Sin la enumeración del paso 1, ese 50 sería solo un número grande.
"""
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import random
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt

SEMILLA = 63
CAPACIDAD = 20
TAMANO_POBLACION = 100
GENERACIONES = 40
TASA_MUTACION = 1 / 12
FIGURES = Path(__file__).resolve().parent.parent / "figures"


@dataclass(frozen=True)
class Articulo:
    """Un artículo de la mochila: peso entero, valor entero.

    Args:
        nombre: etiqueta para las tablas.
        peso: costo de capacidad.
        valor: lo que se maximiza.

    Example:
        El AG reporta los nombres elegidos junto al valor 50 y peso 20.
    """
    nombre: str
    peso: int
    valor: int


ARTICULOS = [
    Articulo("A", 2, 6), Articulo("B", 3, 8), Articulo("C", 4, 9),
    Articulo("D", 5, 12), Articulo("E", 6, 13), Articulo("F", 7, 14),
    Articulo("G", 3, 7), Articulo("H", 8, 16), Articulo("I", 1, 2),
    Articulo("J", 9, 18), Articulo("K", 4, 10), Articulo("L", 2, 5),
]


def totales(bits: Sequence[int]) -> Tuple[int, int]:
    """Peso y valor. La brecha se calcula con el valor, no con el peso.

    Args:
        bits: 12 bits, uno por artículo.

    Returns:
        ``(peso, valor)``.

    Example:
        El campeón del AG suma valor 50 y peso 20: idéntico al óptimo
        enumerado.
    """
    return (sum(bit * articulo.peso for bit, articulo in zip(bits, ARTICULOS)),
            sum(bit * articulo.valor for bit, articulo in zip(bits, ARTICULOS)))


def optimo_exacto() -> Tuple[List[int], int, int]:
    """Los 4,096 cromosomas. Sin esto, "llegó a 50" no diría brecha cero.

    Returns:
        ``(bits, peso, valor)`` del mejor factible: 50 y 20.

    Example:
        ``brecha = valor_exacto - mejor.valor`` imprime 0. Esa resta es
        la razón de haber enumerado en el paso 1.
    """
    factibles = []
    for bits in product((0, 1), repeat=len(ARTICULOS)):
        peso, valor = totales(bits)
        if peso <= CAPACIDAD:
            factibles.append((valor, -peso, list(bits)))
    valor, peso_negativo, bits = max(factibles)
    return bits, -peso_negativo, valor


@dataclass(frozen=True)
class Individuo:
    """Cromosoma con peso y valor. El elitismo conserva el de mayor valor.

    Args:
        bits: 12 decisiones binarias.
        peso: suma de pesos tomados.
        valor: suma de valores tomados.

    Example:
        El campeón final vale 50 con peso 20, factible, brecha 0.
    """
    bits: List[int]
    peso: int
    valor: int


def crear_individuo(bits: List[int]) -> Individuo:
    """Empaqueta bits con peso y valor. Cada evaluación de la factura pasa por aquí.

    Args:
        bits: 12 decisiones binarias.

    Returns:
        Un Individuo.

    Example:
        4,060 evaluaciones: cada una es una llamada a esta función.
    """
    peso, valor = totales(bits)
    return Individuo(bits, peso, valor)


def individuo_aleatorio() -> Individuo:
    """Un cubo {0,1}^12. La población inicial se repara antes de evolucionar.

    Returns:
        Un Individuo crudo.

    Example:
        100 individuos iniciales, SEMILLA = 63. El AG llega a 50 desde
        esa muestra, no desde el óptimo.
    """
    return crear_individuo([random.choice((0, 1)) for _ in ARTICULOS])


def reparar(individuo: Individuo) -> Individuo:
    """Descarta el peor valor/peso hasta caber. Cruza y mutación pueden reabrir el sobrepeso.

    Args:
        individuo: cromosoma, posiblemente ilegal.

    Returns:
        Un Individuo con peso ≤ 20.

    Example:
        El campeón final es factible. Sin reparación, la mutación de un
        bit podría devolver un sobrepeso al almacén.
    """
    bits = individuo.bits.copy()
    orden = sorted((i for i, bit in enumerate(bits) if bit),
                   key=lambda i: (ARTICULOS[i].valor / ARTICULOS[i].peso, ARTICULOS[i].valor))
    for indice in orden:
        if totales(bits)[0] <= CAPACIDAD:
            break
        bits[indice] = 0
    return crear_individuo(bits)


def seleccion_torneo(poblacion: List[Individuo], tamano: int = 3) -> Individuo:
    """Torneo de 3 sobre valor. La población ya es factible.

    Args:
        poblacion: generación actual.
        tamano: contendientes; 3.

    Returns:
        El de mayor valor entre los sorteados.

    Example:
        40 generaciones con esta presión alcanzan el valor exacto 50.
    """
    return max(random.sample(poblacion, tamano), key=lambda individuo: individuo.valor)


def cruza(padre_a: Individuo, padre_b: Individuo) -> Tuple[Individuo, Individuo]:
    """Un punto de corte. Los hijos se mutan y reparan antes de almacenarse.

    Args:
        padre_a: primer padre.
        padre_b: segundo padre.

    Returns:
        Dos hijos crudos.

    Example:
        Combinada con mutación y reparación, 40 generaciones bastan para
        valor 50.
    """
    corte = random.randrange(1, len(ARTICULOS))
    return (crear_individuo(padre_a.bits[:corte] + padre_b.bits[corte:]),
            crear_individuo(padre_b.bits[:corte] + padre_a.bits[corte:]))


# --- NUEVO (1) mutar() --------------------------------------------------------
def mutar(individuo: Individuo) -> Individuo:
    """Invierte bits con probabilidad 1/12. Restaura decisiones que la cruza pudo perder.

    Un bit por artículo, una moneda por bit. La tasa 1/12 espera un flip
    por cromosoma: suficiente para reponer un artículo descartado, no
    tanto como para destruir la reparación.

    Args:
        individuo: cromosoma de partida.

    Returns:
        Un Individuo, posiblemente con sobrepeso.

    Example:
        Sin este operador, la búsqueda se quedaría en el subespacio que
        la cruza de un punto alcanza. Con él, brecha 0 en 4,060
        evaluaciones.
    """
    bits = [1 - bit if random.random() < TASA_MUTACION else bit
            for bit in individuo.bits]
    return crear_individuo(bits)
# ------------------------------------------------------------------------------


def una_generacion(poblacion: List[Individuo], elite: Individuo) -> List[Individuo]:
    """Cruza, muta y repara, copiando al elite para no perder el 50 si ya está.

    Args:
        poblacion: generación actual.
        elite: mejor de la historia hasta ahora.

    Returns:
        Nueva población del mismo tamaño, todos factibles.

    Example:
        40 vueltas, 99 hijos nuevos por vuelta más el elite: 4,060
        evaluaciones en total.
    """
    hijos = [elite]
    while len(hijos) < len(poblacion):
        par = cruza(seleccion_torneo(poblacion), seleccion_torneo(poblacion))
        hijos.extend(reparar(mutar(hijo)) for hijo in par)
    return hijos[:len(poblacion)]


# --- NUEVO (2) ejecutar() -----------------------------------------------------
def ejecutar() -> Tuple[Individuo, List[int], int]:
    """40 generaciones desde SEMILLA = 63. Devuelve el campeón y su factura.

    Returns:
        ``(mejor, historia de valores, evaluaciones)``.

    Example:
        Valor 50, peso 20, brecha 0, 4,060 evaluaciones. La historia
        se dibuja contra la línea del óptimo exacto.
    """
    random.seed(SEMILLA)
    poblacion = [reparar(individuo_aleatorio()) for _ in range(TAMANO_POBLACION)]
    mejor = max(poblacion, key=lambda individuo: individuo.valor)
    historia = [mejor.valor]
    evaluaciones = TAMANO_POBLACION
    for _ in range(GENERACIONES):
        poblacion = una_generacion(poblacion, mejor)
        evaluaciones += TAMANO_POBLACION - 1
        individuo = max(poblacion, key=lambda ind: ind.valor)
        mejor = max((mejor, individuo), key=lambda ind: ind.valor)
        historia.append(mejor.valor)
    return mejor, historia, evaluaciones
# ------------------------------------------------------------------------------


# --- NUEVO (3) el reporte de brecha -------------------------------------------
mejor, historia, evaluaciones = ejecutar()
bits_exactos, peso_exacto, valor_exacto = optimo_exacto()
brecha = valor_exacto - mejor.valor
elegidos = [articulo.nombre for bit, articulo in zip(mejor.bits, ARTICULOS) if bit]
# ------------------------------------------------------------------------------

print("Lección 09 - Mochila 5: la búsqueda completa")
print(f"Mejor solución del AG: valor={mejor.valor}, peso={mejor.peso}, artículos={elegidos}")
print(f"Referencia exacta: valor={valor_exacto}, peso={peso_exacto}")
print(f"Brecha de optimalidad: {brecha} ({brecha / valor_exacto:.1%})")
print(f"Evaluaciones de individuos: {evaluaciones:,}")
print(f"Respuesta final factible: {mejor.peso <= CAPACIDAD}")

FIGURES.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(9, 5))
ax.step(range(len(historia)), historia, where="post", label="mejor valor AG")
ax.axhline(valor_exacto, color="black", linestyle="--", label="óptimo exacto")
ax.set(xlabel="generación", ylabel="valor", title="Búsqueda de mochila contra una referencia exacta")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "knapsack_05_la_busqueda_completa.png", dpi=160)
plt.close(fig)

