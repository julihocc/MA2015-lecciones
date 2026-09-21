"""
Lección 11 - Ecuaciones 2: Lo que vale una muestra aleatoria aquí
=================================================================
NUEVO EN ESTE PASO: tripleta_aleatoria() y una muestra de 400 extracciones de la caja.

Antes de ejecutar un algoritmo genético, averigua qué se logra solo con suerte. En la lección 09
el muestreo aleatorio fue una línea base respetable. Aquí no lo es, y la razón es
visible en un histograma: el residual se mide en dígitos decimales, por lo que una extracción
que parece cercana sigue estando astronómicamente lejos.

CAMBIOS RESPECTO A ecuaciones_01_el_sistema.py
Introdúcelos en este orden:
    1. tripleta_aleatoria() extrae un candidato entero uniformemente de la caja
    2. la muestra           400 extracciones, y lo que el mejor de ellos realmente vale

Ejecútalo:  python ecuaciones_02_poblacion_aleatoria.py

400 extracciones (0.58% de la caja) encuentran 0 soluciones. El mejor es (0, 1, 2) con residual 453; la mediana tiene 957 dígitos y la peor 14,187. Cercano no es cercano.
"""
from math import factorial
from pathlib import Path
import random
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEMILLA = 3
FIGURAS = Path(__file__).resolve().parent.parent / "figures"
TAMANO_MUESTRA = 400

CAJA_BAJA, CAJA_ALTA = -20, 20


def f(x: int, y: int, z: int) -> int:
    """Primera ecuación del sistema; una solución la hace exactamente cero.

    Args:
        x, y, z: enteros en [-20, 20].

    Returns:
        Residual de la primera ecuación. En la raíz (-6, 2, 3) vale 0.
    """
    return (x * y + 2) ** 2 + (x + y) ** (abs(z - 2) ** 3 + 2) + x * y * z


def g(x: int, y: int, z: int) -> int:
    """Segunda ecuación. El factorial obliga a mantener z pequeña.

    Args:
        x, y, z: enteros en [-20, 20].

    Returns:
        Residual. En la raíz (-6, 2, 3) vale 0.
    """
    return x * (y * z + 10) - factorial(abs(z - 3)) + y ** abs(x) + 11 * z


def w(x: int, y: int, z: int) -> int:
    """Tercera ecuación del sistema.

    Args:
        x, y, z: enteros en [-20, 20].

    Returns:
        Residual. En la raíz (-6, 2, 3) vale 0.
    """
    return (x + 7 * y) ** abs(z + x) - (z + 16) ** 2 - 151


def error_total(x: int, y: int, z: int) -> int:
    """Suma de residuales absolutos. Cero es una solución exacta.

    Args:
        x, y, z: enteros en [-20, 20].

    Returns:
        |f|+|g|+|w|, o 10**100 si el factorial/potencia explota.

    Example:
        400 extracciones: mejor residual 453 en (0, 1, 2); 0 soluciones.
    """
    try:
        return abs(f(x, y, z)) + abs(g(x, y, z)) + abs(w(x, y, z))
    except (ValueError, ZeroDivisionError):
        return 10 ** 100


def digitos(error: int) -> int:
    """Longitud decimal del residual, calculada a partir de la longitud en bits porque
    estos enteros son rutinariamente demasiado largos para que str() los convierta."""
    if error == 0:
        return 1
    estimacion = int(error.bit_length() * 0.30103) + 1
    while 10 ** (estimacion - 1) > error:
        estimacion -= 1
    while 10 ** estimacion <= error:
        estimacion += 1
    return estimacion


def acotar(valor: float) -> int:
    """Los genes son enteros dentro de la caja; la cruza y la mutación no lo son."""
    return max(CAJA_BAJA, min(CAJA_ALTA, round(valor)))


# --- NUEVO (1) tripleta_aleatoria() -------------------------------------------
def tripleta_aleatoria() -> Tuple[int, int, int]:
    """Una extracción uniforme de la caja. Esto es todo lo que es la 'búsqueda aleatoria'.

    Returns:
        (x, y, z) en [-20, 20]^3.

    Example:
        400 extracciones, 0.58% de la caja, 0 soluciones. El mejor residual
        es 453 en (0, 1, 2); la mediana tiene 957 dígitos.
    """
    return tuple(random.randint(CAJA_BAJA, CAJA_ALTA) for _ in range(3))
# ------------------------------------------------------------------------------


# --- NUEVO (2) la muestra -----------------------------------------------------
random.seed(SEMILLA)
muestra: List[Tuple[Tuple[int, int, int], int]] = [
    (tripleta, error_total(*tripleta)) for tripleta in (tripleta_aleatoria() for _ in range(TAMANO_MUESTRA))
]
exactas = [tripleta for tripleta, error in muestra if error == 0]
mejor_tripleta, mejor_error = min(muestra, key=lambda par: par[1])
tamanos = sorted(digitos(error) for _, error in muestra)
mediana = tamanos[len(tamanos) // 2]
bajo_diez = [error for _, error in muestra if error < 10]
# ------------------------------------------------------------------------------

lado = CAJA_ALTA - CAJA_BAJA + 1
print("Lección 11 - Ecuaciones 2: lo que vale una muestra aleatoria aquí")
print(f"Semilla {SEMILLA}; {TAMANO_MUESTRA} extracciones uniformes de una caja de {lado ** 3:,} tripletas")
print(f"Soluciones exactas encontradas:      {len(exactas)}")
print(f"Mejor extracción:                  {mejor_tripleta} con residual {mejor_error:,}")
print(f"Tamaño del residual, dígitos:      min {tamanos[0]}, mediana {mediana}, max {tamanos[-1]}")
print(f"Extracciones con residual < 10:   {len(bajo_diez)}")
print(f"Fracción de la caja vista:   {TAMANO_MUESTRA / lado ** 3:.2%}")
print("Un residual con una mediana de "
      f"{mediana} dígitos no está 'casi resuelto'; es un número completamente diferente.")

FIGURAS.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(tamanos, bins=40, color="#4c72b0", edgecolor="white")
ax.axvline(digitos(mejor_error), color="crimson", linestyle="--",
           label=f"mejor extracción: {digitos(mejor_error)} dígitos")
ax.set(xlabel="dígitos decimales del residual", ylabel="extracciones",
       title=f"{TAMANO_MUESTRA} candidatos aleatorios, ninguno de ellos es solución")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURAS / "ecuaciones_02_poblacion_aleatoria.png", dpi=160)
plt.close(fig)

