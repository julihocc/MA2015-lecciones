"""
Lección 09 - Mochila 3: La reparación crea presión de selección
===============================================================
NUEVO EN ESTE PASO: reparar() y un experimento de antes/después.

CAMBIOS RESPECTO A knapsack_02_poblacion_aleatoria.py
Introdúzcalos en este orden:
    1. reparar()             elimina opciones débiles de valor-por-peso hasta que sea legal
    2. la comparación        mide qué arregla la reparación y qué descarta

Ejecútalo:  python knapsack_03_reparacion.py

La reparación hace factibles a 1,000/1,000 y cambia 767 cromosomas. Restaura legalidad, pero la muestra que busca el AG ya no es la del cubo aleatorio.
"""
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import random
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt

SEMILLA = 63
CAPACIDAD = 20
TAMANO_POBLACION = 1000
FIGURES = Path(__file__).resolve().parent.parent / "figures"


@dataclass(frozen=True)
class Articulo:
    """Un artículo de la mochila: peso entero, valor entero.

    Args:
        nombre: etiqueta para las tablas.
        peso: costo de capacidad.
        valor: lo que se maximiza.

    Example:
        ``reparar`` descarta primero el menor valor/peso. Esa regla usa
        estos dos enteros, no el nombre.
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
    """Peso y valor de un cromosoma. ``reparar`` lo consulta en cada descarte.

    Args:
        bits: 12 bits, uno por artículo.

    Returns:
        ``(peso, valor)``.

    Example:
        Tras reparar, los 1,000 pesos son ≤ 20. Antes, 767 no lo eran.
    """
    return (sum(bit * articulo.peso for bit, articulo in zip(bits, ARTICULOS)),
            sum(bit * articulo.valor for bit, articulo in zip(bits, ARTICULOS)))


def optimo_exacto() -> Tuple[List[int], int, int]:
    """Referencia enumerada: valor 50, peso 20. La reparación no la toca.

    Returns:
        ``(bits, peso, valor)`` del mejor factible.

    Example:
        El mejor reparado de 1,000 aleatorios no tiene por qué ser 50:
        reparar legaliza, no optimiza.
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
    """Cromosoma con peso, valor y factibilidad. ``reparar`` devuelve otro.

    Args:
        bits: 12 decisiones binarias.
        peso: suma de pesos tomados.
        valor: suma de valores tomados.

    Example:
        1,000 crudos vs 1,000 reparados: 233 factibles antes, 1,000 después.
    """
    bits: List[int]
    peso: int
    valor: int

    @property
    def factible(self) -> bool:
        """True solo si el peso cabe en la capacidad 20."""
        return self.peso <= CAPACIDAD


def crear_individuo(bits: List[int]) -> Individuo:
    """Empaqueta bits con su peso y valor. No repara: el que llama decide.

    Args:
        bits: 12 decisiones binarias.

    Returns:
        Un Individuo, factible o no.

    Example:
        ``reparar`` construye el resultado por aquí, una vez que el peso
        ya es ≤ 20.
    """
    peso, valor = totales(bits)
    return Individuo(bits, peso, valor)


def individuo_aleatorio() -> Individuo:
    """Un cubo {0,1}^12. La mayoría nace ilegal; por eso existe ``reparar``.

    Returns:
        Un Individuo, de entrada a menudo con sobrepeso.

    Example:
        De 1,000 sorteos, 767 cambian al repararse. Esa es la fracción
        que el paso 2 ya había contado como infactible.
    """
    return crear_individuo([random.choice((0, 1)) for _ in ARTICULOS])


# --- NUEVO (1) reparar() ------------------------------------------------------
def reparar(individuo: Individuo) -> Individuo:
    """Descarta los artículos seleccionados con el valor más débil por unidad de peso.

    La regla es greedy y sesgada: no explora alternativas, tira primero lo
    menos eficiente. Eso legaliza a todos y cambia la distribución que
    verá la selección. El 76.7% de la muestra se mueve; eso no es fontanería
    gratuita.

    Args:
        individuo: cromosoma crudo, a menudo con sobrepeso.

    Returns:
        Un Individuo con peso ≤ 20. Puede ser idéntico si ya era legal.

    Example:
        1,000/1,000 factibles después; 767 cromosomas cambiados. La
        factibilidad se compró descartando bits, no buscándolos.
    """
    bits = individuo.bits.copy()
    orden = sorted((i for i, bit in enumerate(bits) if bit),
                   key=lambda i: (ARTICULOS[i].valor / ARTICULOS[i].peso, ARTICULOS[i].valor))
    for indice in orden:
        if totales(bits)[0] <= CAPACIDAD:
            break
        bits[indice] = 0
    return crear_individuo(bits)
# ------------------------------------------------------------------------------


# --- NUEVO (2) la comparación -------------------------------------------------
random.seed(SEMILLA)
crudos = [individuo_aleatorio() for _ in range(TAMANO_POBLACION)]
reparados = [reparar(individuo) for individuo in crudos]
cambiados = sum(antes.bits != despues.bits for antes, despues in zip(crudos, reparados))
valor_removido = [antes.valor - despues.valor for antes, despues in zip(crudos, reparados)]
# ------------------------------------------------------------------------------

_, peso_exacto, valor_exacto = optimo_exacto()
print("Lección 09 - Mochila 3: reparación")
print(f"Factibles antes de la reparación: {sum(c.factible for c in crudos)}/{TAMANO_POBLACION}")
print(f"Factibles después de la reparación: {sum(c.factible for c in reparados)}/{TAMANO_POBLACION}")
print(f"Cromosomas cambiados:       {cambiados}/{TAMANO_POBLACION}")
print(f"Valor medio removido:       {sum(valor_removido) / TAMANO_POBLACION:.2f}")
print(f"Mejor valor reparado:       {max(c.valor for c in reparados)}")
print(f"Referencia exacta:          valor={valor_exacto}, peso={peso_exacto}")

FIGURES.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter([c.peso for c in crudos], [c.valor for c in crudos], alpha=0.25,
           s=18, label="antes de reparación")
ax.scatter([c.peso for c in reparados], [c.valor for c in reparados], alpha=0.35,
           s=18, label="después de reparación")
ax.axvline(CAPACIDAD, color="tab:red", linewidth=2, label="capacidad")
ax.set(xlabel="peso", ylabel="valor", title="La reparación restaura la factibilidad pero cambia la muestra")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "knapsack_03_reparacion.png", dpi=160)
plt.close(fig)

