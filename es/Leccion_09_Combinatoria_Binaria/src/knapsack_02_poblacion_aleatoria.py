"""
Lección 09 - Mochila 2: Una población aleatoria
===============================================
NUEVO EN ESTE PASO: Individuo (Candidate), individuo_aleatorio(), y un censo de factibilidad.

CAMBIOS RESPECTO A knapsack_01_la_instancia.py
Introdúzcalos en este orden:
    1. Individuo                 mantiene cromosoma, peso, valor y factibilidad juntos
    2. individuo_aleatorio()     muestrea cada decisión de forma independiente
    3. TAMANO_POBLACION          hace explícito el tamaño del experimento
    4. el censo                  mide la factibilidad antes de diseñar la selección

Ejecútalo:  python knapsack_02_poblacion_aleatoria.py

De 1,000 cromosomas aleatorios, 233 son factibles y el mejor vale 47. La selección no puede rescatar una muestra que casi no contiene respuestas legales.
"""
from dataclasses import dataclass
from itertools import product
from pathlib import Path
import random
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt

SEMILLA = 63
CAPACIDAD = 20
FIGURES = Path(__file__).resolve().parent.parent / "figures"


@dataclass(frozen=True)
class Articulo:
    """Un artículo de la mochila: peso entero, valor entero.

    Args:
        nombre: etiqueta para las tablas.
        peso: costo de capacidad.
        valor: lo que se maximiza.

    Example:
        Doce artículos, capacidad 20. El óptimo exacto sigue valiendo 50.
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
    """Peso y valor de un cromosoma. Un sobrepeso es un dato, no un error.

    Args:
        bits: 12 bits, uno por artículo.

    Returns:
        ``(peso, valor)``.

    Example:
        767 de 1,000 aleatorios pesan más de 20: esos no entran al censo
        de "mejor valor factible".
    """
    peso = sum(bit * articulo.peso for bit, articulo in zip(bits, ARTICULOS))
    valor = sum(bit * articulo.valor for bit, articulo in zip(bits, ARTICULOS))
    return peso, valor


def optimo_exacto() -> Tuple[List[int], int, int]:
    """Enumera los 4,096 cromosomas. La referencia no cambia entre pasos.

    Returns:
        ``(bits, peso, valor)`` del mejor factible: valor 50, peso 20.

    Example:
        El mejor aleatorio factible vale 47. Sin esta referencia, 47
        parecería un éxito.
    """
    factibles = []
    for bits in product((0, 1), repeat=len(ARTICULOS)):
        peso, valor = totales(bits)
        if peso <= CAPACIDAD:
            factibles.append((valor, -peso, list(bits)))
    valor, peso_negativo, bits = max(factibles)
    return bits, -peso_negativo, valor


# --- NUEVO (1) Individuo ------------------------------------------------------
@dataclass(frozen=True)
class Individuo:
    """Un cromosoma con su peso, su valor y su factibilidad juntos.

    Sin este objeto, "mejor" y "legal" viven en columnas distintas y la
    selección puede copiar un sobrepeso porque su valor se ve grande.

    Args:
        bits: 12 decisiones binarias.
        peso: suma de pesos tomados.
        valor: suma de valores tomados.

    Example:
        233 de 1,000 individuos aleatorios son factibles. El mejor de
        esos 233 vale 47.
    """
    bits: List[int]
    peso: int
    valor: int

    @property
    def factible(self) -> bool:
        """True solo si el peso cabe en la capacidad 20."""
        return self.peso <= CAPACIDAD
# ------------------------------------------------------------------------------


# --- NUEVO (2) individuo_aleatorio() ------------------------------------------
def individuo_aleatorio() -> Individuo:
    """Muestrea cada bit de forma independiente. La independencia es el problema.

    Un bit por artículo, moneda justa. El producto de doce monedas no
    respeta la capacidad: por eso el 23.3% factible no es un accidente
    de semilla, es la geometría del cubo {0,1}^12 contra el plano peso=20.

    Returns:
        Un Individuo, factible o no.

    Example:
        SEMILLA = 63, 1,000 sorteos: 233 factibles, mejor valor 47.
    """
    bits = [random.choice((0, 1)) for _ in ARTICULOS]
    peso, valor = totales(bits)
    return Individuo(bits, peso, valor)
# ------------------------------------------------------------------------------


# --- NUEVO (3) TAMANO_POBLACION -----------------------------------------------
TAMANO_POBLACION = 1000
# ------------------------------------------------------------------------------


# --- NUEVO (4) el censo -------------------------------------------------------
random.seed(SEMILLA)
poblacion = [individuo_aleatorio() for _ in range(TAMANO_POBLACION)]
factibles = [individuo for individuo in poblacion if individuo.factible]
mejor_aleatorio = max(factibles, key=lambda individuo: individuo.valor)
# ------------------------------------------------------------------------------

_, peso_exacto, valor_exacto = optimo_exacto()
print("Lección 09 - Mochila 2: una población aleatoria")
print(f"Cromosomas aleatorios factibles: {len(factibles)}/{TAMANO_POBLACION} "
      f"({len(factibles) / TAMANO_POBLACION:.1%})")
print(f"Mejor valor aleatorio factible: {mejor_aleatorio.valor} con peso {mejor_aleatorio.peso}")
print(f"Referencia exacta: valor={valor_exacto}, peso={peso_exacto}")

FIGURES.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist([individuo.peso for individuo in poblacion], bins=range(0, 61, 3),
        color="tab:blue", alpha=0.8)
ax.axvline(CAPACIDAD, color="tab:red", linewidth=2, label="capacidad")
ax.set(xlabel="peso del cromosoma", ylabel="conteo",
       title="Los bits aleatorios independientes normalmente exceden la capacidad")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURES / "knapsack_02_poblacion_aleatoria.png", dpi=160)
plt.close(fig)

