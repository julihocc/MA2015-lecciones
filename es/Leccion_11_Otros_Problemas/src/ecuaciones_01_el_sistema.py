"""
Lección 11 - Ecuaciones 1: Un problema cuya respuesta puede verificarse
=======================================================================
NUEVO EN ESTE PASO: f(), g(), w(), error_total(), digitos() y la caja de búsqueda.

Hasta ahora, cada problema ha sido de optimización: producíamos una respuesta y no
teníamos forma de saber si era la mejor. La Lección 08 lo hizo explícito. Un
sistema de ecuaciones es la rara excepción. Un candidato (x, y, z) es una solución
si y solo si las tres ecuaciones se evalúan exactamente en cero, y eso es una
verificación de una línea, no una opinión.

Los genes son enteros, así que todo el espacio de búsqueda es una caja finita. Recuerda ese
número: volverá en el paso 4.

Ejecútalo:  python ecuaciones_01_el_sistema.py

Los genes son enteros en [-20, 20], así que la caja tiene 68,921 tripletas. Los residuales son enteros, no flotantes: a lo largo de (y, z) = (1, 1) van de 3 dígitos en x = 2 a 31 dígitos en x = 20.
"""
from math import factorial
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path

FIGURAS = Path(__file__).resolve().parent.parent / "figures"

# Los genes están acotados a esta caja de enteros. La cota no es decoración: los
# exponentes a continuación crecen con |z| y |x|, así que un gen sin acotar hace que una sola
# evaluación de aptitud tome más tiempo que toda la lección.
CAJA_BAJA, CAJA_ALTA = -20, 20


def f(x: int, y: int, z: int) -> int:
    """Primera ecuación del sistema; una solución la hace cero.

    Args:
        x, y, z: enteros en [-20, 20].

    Returns:
        Un entero. Cero si y solo si esta ecuación se cumple.

    Example:
        En (-6, 2, 3) vale 0, igual que g y w. Eso se verifica en el paso 3.
    """
    return (x * y + 2) ** 2 + (x + y) ** (abs(z - 2) ** 3 + 2) + x * y * z


def g(x: int, y: int, z: int) -> int:
    """Segunda ecuación. El factorial es la razón por la que z debe mantenerse pequeña.

    Args:
        x, y, z: enteros en [-20, 20].

    Returns:
        Un entero. ``factorial(|z-3|)`` explota si z se sale de la caja.

    Example:
        Sin la caja, una sola evaluación de g tarda más que enumerar
        las 68,921 tripletas.
    """
    return x * (y * z + 10) - factorial(abs(z - 3)) + y ** abs(x) + 11 * z


def w(x: int, y: int, z: int) -> int:
    """Tercera ecuación.

    Args:
        x, y, z: enteros en [-20, 20].

    Returns:
        Un entero. El exponente |z+x| es la otra razón de acotar.

    Example:
        A lo largo de (y, z) = (1, 1), el residual conjunto pasa de 3
        dígitos en x = 2 a 31 dígitos en x = 20.
    """
    return (x + 7 * y) ** abs(z + x) - (z + 16) ** 2 - 151


def error_total(x: int, y: int, z: int) -> int:
    """Suma de residuales absolutos. Exactamente cero significa una solución exacta.

    Args:
        x, y, z: enteros en [-20, 20].

    Returns:
        |f|+|g|+|w|, o 10^100 si la evaluación revienta.

    Example:
        68,921 tripletas. El paso 4 prueba que exactamente una da 0.
    """
    try:
        return abs(f(x, y, z)) + abs(g(x, y, z)) + abs(w(x, y, z))
    except (ValueError, ZeroDivisionError):
        return 10 ** 100


def digitos(error: int) -> int:
    """Longitud decimal del residual, calculada a partir de la longitud en bits porque
    estos enteros son rutinariamente demasiado largos para que str() los convierta.

    Args:
        error: residual absoluto, posiblemente enorme.

    Returns:
        Número de dígitos decimales.

    Example:
        En (y, z) = (1, 1), x = 20 produce 31 dígitos. str() no es la
        herramienta.
    """
    if error == 0:
        return 1
    estimacion = int(error.bit_length() * 0.30103) + 1
    while 10 ** (estimacion - 1) > error:
        estimacion -= 1
    while 10 ** estimacion <= error:
        estimacion += 1
    return estimacion


def acotar(valor: float) -> int:
    """Los genes son enteros dentro de la caja; la cruza y la mutación no lo son.

    Args:
        valor: real propuesto por un operador.

    Returns:
        Entero en [-20, 20].

    Example:
        Sin esto, g llamaría factorial de un z enorme. La caja no es
        decoración.
    """
    return max(CAJA_BAJA, min(CAJA_ALTA, round(valor)))


def describir(punto: Tuple[int, int, int]) -> str:
    """Una línea por candidato, en las unidades en que realmente se plantea el problema.

    Args:
        punto: (x, y, z).

    Returns:
        Texto con residual y si es solución.

    Example:
        Los puntos de la línea (y, z) = (1, 1) se imprimen así: 3 dígitos
        en x = 2, 31 en x = 20.
    """
    error = error_total(*punto)
    mostrado = str(error) if error < 10 ** 9 else f"~10^{digitos(error) - 1}"
    return f"  {str(punto):<14} |f|+|g|+|w| = {mostrado:<12} solución: {error == 0}"


lado = CAJA_ALTA - CAJA_BAJA + 1
print("Lección 11 - Ecuaciones 1: un problema cuya respuesta puede verificarse")
print(f"Caja de búsqueda: enteros en [{CAJA_BAJA}, {CAJA_ALTA}] para cada uno de x, y, z")
print(f"Tamaño de caja:   {lado} x {lado} x {lado} = {lado ** 3:,} tripletas candidatas")
print("Un candidato es una solución exactamente cuando el residual total es 0.")
print("Cuatro candidatos elegidos a mano:")
for candidato in [(0, 0, 0), (1, 1, 1), (-3, 4, 2), (5, -2, 7)]:
    print(describir(candidato))

barrido = [(x, error_total(x, 1, 1)) for x in range(CAJA_BAJA, CAJA_ALTA + 1)]
peor = max(barrido, key=lambda par: par[1])
mejor = min(barrido, key=lambda par: par[1])
print(f"Barriendo x con (y, z) = (1, 1): el residual va desde "
      f"{digitos(mejor[1])} dígitos en x={mejor[0]} hasta {digitos(peor[1])} dígitos en x={peor[0]}.")
print("El objetivo no es suave y sus valores no son flotantes. Eso importa.")

FIGURAS.mkdir(exist_ok=True)
cuadricula = [[digitos(error_total(x, y, 0)) for x in range(CAJA_BAJA, CAJA_ALTA + 1)]
        for y in range(CAJA_BAJA, CAJA_ALTA + 1)]
fig, ejes = plt.subplots(1, 2, figsize=(11, 4.5))
imagen = ejes[0].imshow(cuadricula, origin="lower", extent=(CAJA_BAJA, CAJA_ALTA, CAJA_BAJA, CAJA_ALTA),
                       cmap="magma")
ejes[0].set(xlabel="x", ylabel="y", title="Dígitos del residual en el corte z = 0")
fig.colorbar(imagen, ax=ejes[0], label="dígitos decimales")
ejes[1].plot([x for x, _ in barrido], [digitos(e) for _, e in barrido], "o-")
ejes[1].set(xlabel="x", ylabel="dígitos decimales del residual",
            title="Una línea a través de la caja, (y, z) = (1, 1)")
fig.tight_layout()
fig.savefig(FIGURAS / "ecuaciones_01_el_sistema.png", dpi=160)
plt.close(fig)

