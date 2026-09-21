"""
Lección 09 - Mochila 1: La instancia
====================================
NUEVO EN ESTE PASO: un problema de decisión binaria fijo y su referencia exacta.

Cada bit responde a una pregunta: tomar este artículo o dejarlo. Antes de introducir un
AG, enumere esta instancia deliberadamente pequeña para que las ejecuciones posteriores tengan una
respuesta honesta contra la cual comparar.

Ejecútalo:  python knapsack_01_la_instancia.py

La instancia de 12 artículos tiene 4,096 cromosomas. El óptimo exacto vale 50 con peso 20: esa referencia es la que medirá cada paso posterior.
"""
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt

SEMILLA = 63
CAPACIDAD = 20
FIGURES = Path(__file__).resolve().parent.parent / "figures"


@dataclass(frozen=True)
class Articulo:
    """Un artículo de la mochila: peso entero, valor entero.

    Doce artículos caben en 2^12 = 4,096 cromosomas, lo bastante pocos para
    enumerar el óptimo exacto antes de confiar en el AG.

    Args:
        nombre: etiqueta para las tablas.
        peso: costo de capacidad.
        valor: lo que se maximiza.

    Example:
        Capacidad 20. El óptimo exacto vale 50 con peso 20.
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
    """Peso y valor de un cromosoma binario. No repara: un sobrepeso es un dato.

    Args:
        bits: 12 bits, uno por artículo, 1 = tomarlo.

    Returns:
        ``(peso, valor)``. Factible solo si peso ≤ 20.

    Example:
        El óptimo exacto suma peso 20 y valor 50. Un cromosoma de puros 1
        pesa mucho más que la capacidad: eso es lo que el paso 2 cuenta.
    """
    peso = sum(bit * articulo.peso for bit, articulo in zip(bits, ARTICULOS))
    valor = sum(bit * articulo.valor for bit, articulo in zip(bits, ARTICULOS))
    return peso, valor


def optimo_exacto() -> Tuple[List[int], int, int]:
    """Enumera 2^12 cromosomas; esta es una referencia, no el AG.

    Una instancia de 12 bits se eligió a propósito: 4,096 es barato de
    recorrer y caro de adivinar. Sin esta línea, el paso 5 no podría
    decir "brecha cero".

    Returns:
        ``(bits, peso, valor)`` del mejor factible. Empates se rompen
        preferiendo el menor peso.

    Example:
        Valor 50, peso 20. Esa terna es la línea contra la que el AG
        del paso 5 reporta brecha 0 en 4,060 evaluaciones.
    """
    factibles = []
    for bits in product((0, 1), repeat=len(ARTICULOS)):
        peso, valor = totales(bits)
        if peso <= CAPACIDAD:
            factibles.append((valor, -peso, list(bits)))
    valor, peso_negativo, bits = max(factibles)
    return bits, -peso_negativo, valor


mejores_bits, mejor_peso, mejor_valor = optimo_exacto()
elegidos = [articulo.nombre for bit, articulo in zip(mejores_bits, ARTICULOS) if bit]

print("Lección 09 - Mochila 1: la instancia")
print(f"Artículos: {len(ARTICULOS)}; espacio de búsqueda binario: 2^{len(ARTICULOS)} = {2 ** len(ARTICULOS):,}")
print(f"Capacidad: {CAPACIDAD}")
print(f"Referencia exacta: valor={mejor_valor}, peso={mejor_peso}, artículos={elegidos}")

FIGURES.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter([articulo.peso for articulo in ARTICULOS], [articulo.valor for articulo in ARTICULOS], s=70)
for articulo in ARTICULOS:
    ax.annotate(articulo.nombre, (articulo.peso, articulo.valor), xytext=(4, 4),
                textcoords="offset points")
ax.set(xlabel="peso", ylabel="valor", title="Artículos de mochila: valor vs peso")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(FIGURES / "knapsack_01_la_instancia.png", dpi=160)
plt.close(fig)

