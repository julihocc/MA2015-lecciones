"""
Lección 09 - Mochila 4: Selección y cruza
=========================================
NUEVO EN ESTE PASO: seleccion_torneo(), cruza(), y una_generacion().

CAMBIOS RESPECTO A knapsack_03_reparacion.py
Introdúzcalos en este orden:
    1. seleccion_torneo()    prefiere valor solo después de garantizar factibilidad
    2. cruza()               recombina dos decisiones binarias legales en un punto de corte
    3. una_generacion()      expone la tubería de selección-cruza-reparación

Ejecútalo:  python knapsack_04_seleccion_y_cruza.py

Una generación eleva el valor medio de 39.00 a 40.73. Todos los hijos almacenados siguen siendo factibles: la reparación cierra la tubería que la cruza abre.
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
FIGURES = Path(__file__).resolve().parent.parent / "figures"


@dataclass(frozen=True)
class Articulo:
    """Un artículo de la mochila: peso entero, valor entero.

    Args:
        nombre: etiqueta para las tablas.
        peso: costo de capacidad.
        valor: lo que se maximiza.

    Example:
        El torneo compara ``valor``; la reparación usa valor/peso. Los dos
        enteros no son intercambiables.
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
    """Peso y valor. La cruza de un punto puede devolver sobrepeso; esto lo detecta.

    Args:
        bits: 12 bits, uno por artículo.

    Returns:
        ``(peso, valor)``.

    Example:
        Tras una generación, todos los hijos almacenados tienen peso ≤ 20
        y valor medio 40.73.
    """
    return (sum(bit * articulo.peso for bit, articulo in zip(bits, ARTICULOS)),
            sum(bit * articulo.valor for bit, articulo in zip(bits, ARTICULOS)))


def optimo_exacto() -> Tuple[List[int], int, int]:
    """Referencia enumerada: valor 50. Una generación no tiene por qué llegar.

    Returns:
        ``(bits, peso, valor)`` del mejor factible.

    Example:
        El mejor de una generación se compara contra 50, no contra el
        mejor aleatorio 47 del paso 2.
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
    """Cromosoma factible o no. El torneo lee ``valor``; la reparación, el peso.

    Args:
        bits: 12 decisiones binarias.
        peso: suma de pesos tomados.
        valor: suma de valores tomados.

    Example:
        Población de 100, todos reparados. Valor medio 39.00 antes de la
        generación, 40.73 después.
    """
    bits: List[int]
    peso: int
    valor: int


def crear_individuo(bits: List[int]) -> Individuo:
    """Empaqueta bits con peso y valor. No repara: la cruza produce ilegales.

    Args:
        bits: 12 decisiones binarias.

    Returns:
        Un Individuo, a veces con sobrepeso.

    Example:
        Los hijos crudos de ``cruza`` pasan por aquí antes de ``reparar``.
    """
    peso, valor = totales(bits)
    return Individuo(bits, peso, valor)


def individuo_aleatorio() -> Individuo:
    """Un cubo {0,1}^12. La población inicial se repara antes de seleccionar.

    Returns:
        Un Individuo crudo.

    Example:
        100 sorteos, todos reparados, valor medio 39.00. Esa es la línea
        base de la generación.
    """
    return crear_individuo([random.choice((0, 1)) for _ in ARTICULOS])


def reparar(individuo: Individuo) -> Individuo:
    """Descarta el peor valor/peso hasta caber. Cierra lo que la cruza abre.

    Args:
        individuo: cromosoma, a menudo con sobrepeso tras el corte.

    Returns:
        Un Individuo con peso ≤ 20.

    Example:
        Los 100 hijos almacenados son factibles. La cruza sí produce
        sobrepeso; este operador no deja que se almacene.
    """
    bits = individuo.bits.copy()
    orden = sorted((i for i, bit in enumerate(bits) if bit),
                   key=lambda i: (ARTICULOS[i].valor / ARTICULOS[i].peso, ARTICULOS[i].valor))
    for indice in orden:
        if totales(bits)[0] <= CAPACIDAD:
            break
        bits[indice] = 0
    return crear_individuo(bits)


# --- NUEVO (1) seleccion_torneo() ---------------------------------------------
def seleccion_torneo(poblacion: List[Individuo], tamano: int = 3) -> Individuo:
    """Prefiere valor solo después de que todos son factibles.

    Si la población mezclara sobrepesos, el torneo copiaría el más valioso
    e ilegal. Por eso el paso 3 repara primero.

    Args:
        poblacion: individuos ya legales.
        tamano: contendientes; 3 es la presión de la Lección 03.

    Returns:
        El de mayor valor entre ``tamano`` sorteados.

    Example:
        Una generación con esta presión sube el valor medio de 39.00 a
        40.73.
    """
    return max(random.sample(poblacion, tamano), key=lambda individuo: individuo.valor)
# ------------------------------------------------------------------------------


# --- NUEVO (2) cruza() --------------------------------------------------------
def cruza(padre_a: Individuo, padre_b: Individuo) -> Tuple[Individuo, Individuo]:
    """Un punto de corte. Dos padres legales no garantizan dos hijos legales.

    Args:
        padre_a: primer padre factible.
        padre_b: segundo padre factible.

    Returns:
        Dos hijos, a menudo con sobrepeso. ``una_generacion`` los cuenta
        antes de reparar.

    Example:
        La tubería imprime cuántos hijos crudos exceden 20 y luego
        almacena 100/100 factibles.
    """
    corte = random.randrange(1, len(ARTICULOS))
    primero = crear_individuo(padre_a.bits[:corte] + padre_b.bits[corte:])
    segundo = crear_individuo(padre_b.bits[:corte] + padre_a.bits[corte:])
    return primero, segundo
# ------------------------------------------------------------------------------


# --- NUEVO (3) una_generacion() -----------------------------------------------
def una_generacion(poblacion: List[Individuo]) -> Tuple[List[Individuo], int]:
    """Selección, cruza y reparación. El entero es cuántos hijos nacieron ilegales.

    Args:
        poblacion: generación actual, ya legal.

    Returns:
        ``(hijos, no_factibles_antes_reparacion)``. Los hijos almacenados
        son todos factibles.

    Example:
        Valor medio 39.00 → 40.73. El conteo de ilegales crudos prueba
        que la cruza no preserva la capacidad por sí sola.
    """
    hijos: List[Individuo] = []
    no_factibles_antes_reparacion = 0
    while len(hijos) < len(poblacion):
        hijos_crudos = cruza(seleccion_torneo(poblacion), seleccion_torneo(poblacion))
        no_factibles_antes_reparacion += sum(hijo.peso > CAPACIDAD for hijo in hijos_crudos)
        hijos.extend(reparar(hijo) for hijo in hijos_crudos)
    return hijos[:len(poblacion)], no_factibles_antes_reparacion
# ------------------------------------------------------------------------------


random.seed(SEMILLA)
poblacion = [reparar(individuo_aleatorio()) for _ in range(TAMANO_POBLACION)]
antes = [individuo.valor for individuo in poblacion]
poblacion, hijos_ilegales = una_generacion(poblacion)
despues = [individuo.valor for individuo in poblacion]
_, _, valor_exacto = optimo_exacto()

print("Lección 09 - Mochila 4: selección y cruza")
print(f"Valor medio antes: {sum(antes) / len(antes):.2f}")
print(f"Valor medio después: {sum(despues) / len(despues):.2f}")
print(f"La cruza produjo {hijos_ilegales} hijos con sobrepeso antes de reparar.")
print(f"Los {len(poblacion)} hijos almacenados son factibles después de la reparación.")
print(f"Mejor después de una generación: {max(despues)}; referencia exacta: {valor_exacto}")

FIGURES.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(antes, bins=range(0, valor_exacto + 3, 3), alpha=0.55, label="antes")
ax.hist(despues, bins=range(0, valor_exacto + 3, 3), alpha=0.55, label="después")
ax.axvline(valor_exacto, color="black", linestyle="--", label="óptimo exacto")
ax.set(xlabel="valor factible", ylabel="conteo",
       title="Una generación seleccionada desplaza el valor sin perder legalidad")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "knapsack_04_seleccion_y_cruza.png", dpi=160)
plt.close(fig)

