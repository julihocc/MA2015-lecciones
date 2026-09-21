"""
Lección 10 - TSP 4: La mutación cambia el orden, no la pertenencia
=========================================================
NUEVO EN ESTE PASO: mutacion_intercambio(), mutacion_inversion(), y una auditoría de distancia.

CAMBIOS RESPECTO A tsp_03_cruza_ordenada.py
Introdúcelos en este orden:
    1. mutacion_intercambio()   intercambiar dos posiciones
    2. mutacion_inversion()     revertir un segmento contiguo
    3. la auditoria de distancia  medir cuán diferente se mueven los operadores

Ejecútalo:  python tsp_04_mutacion.py

El intercambio y la inversión preservan cada ciudad. Sus cambios medios de longitud son +451.0 y +238.9: legales las dos, no equivalentes.
"""
from math import dist
from pathlib import Path
import random
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt

SEMILLA = 60
DATOS = Path(__file__).with_name("att48_xy.txt")
FIGURAS = Path(__file__).resolve().parent.parent / "figures"
PRUEBAS = 1000


def cargar_puntos(ruta: Path = DATOS) -> List[Tuple[int, int]]:
    """Las 48 ciudades ATT.

    Args:
        ruta: ``att48_xy.txt``.

    Returns:
        48 pares (x, y).

    Example:
        Las longitudes +451.0 y +238.9 se miden sobre estas coordenadas.
    """
    return [tuple(map(int, linea.split())) for linea in ruta.read_text().splitlines() if linea.strip()]


def longitud_ruta(puntos: Sequence[Tuple[int, int]], ruta: Sequence[int]) -> float:
    """Longitud del ciclo. Aquí mide cuánto se mueve cada mutación.

    Args:
        puntos: coordenadas.
        ruta: permutación legal.

    Returns:
        Longitud euclidiana cerrada.

    Example:
        Delta medio intercambio +451.0, inversión +238.9.
    """
    return sum(dist(puntos[a], puntos[b]) for a, b in zip(ruta, ruta[1:] + ruta[:1]))


def ruta_aleatoria(tamano: int) -> List[int]:
    """Una permutación legal. La auditoría muta una sola base mil veces.

    Args:
        tamano: 48.

    Returns:
        Una permutación de 0..47.

    Example:
        SEMILLA = 60. Intercambio e inversión se aplican a la misma base
        para que los deltas sean comparables.
    """
    ruta = list(range(tamano)); random.shuffle(ruta); return ruta


def cruza_ordenada(primero: Sequence[int], segundo: Sequence[int]) -> List[int]:
    """Permutación legal. Este paso no la llama: viaja para el paso 5.

    Args:
        primero: padre legal.
        segundo: padre legal.

    Returns:
        Una permutación.

    Example:
        1,000/1,000 legales en el paso 3. La mutación no debe romper
        lo que la cruza ya garantiza.
    """
    izquierdo, derecho = sorted(random.sample(range(len(primero)), 2))
    hijo = [-1] * len(primero); hijo[izquierdo:derecho] = primero[izquierdo:derecho]
    restantes = [ciudad for ciudad in segundo if ciudad not in hijo]
    huecos = list(range(derecho, len(primero))) + list(range(izquierdo))
    for indice, ciudad in zip(huecos, restantes): hijo[indice] = ciudad
    return hijo


# --- NUEVO (1) mutacion_intercambio() -----------------------------------------
def mutacion_intercambio(fuente: Sequence[int]) -> List[int]:
    """Intercambia dos posiciones. Preserva la permutación; mueve dos aristas.

    Args:
        fuente: ruta legal.

    Returns:
        Una permutación legal.

    Example:
        1,000 mutaciones: todas legales, cambio medio de longitud +451.0.
    """
    ruta = list(fuente)
    primera, segunda = random.sample(range(len(ruta)), 2)
    ruta[primera], ruta[segunda] = ruta[segunda], ruta[primera]
    return ruta
# ------------------------------------------------------------------------------


# --- NUEVO (2) mutacion_inversion() ------------------------------------------
def mutacion_inversion(fuente: Sequence[int]) -> List[int]:
    """Invierte un segmento. Preserva la permutación; mueve dos aristas, no todas.

    2-opt disfrazado: solo cambian los extremos del segmento. Por eso
    su delta medio (+238.9) es menor que el del intercambio (+451.0).

    Args:
        fuente: ruta legal.

    Returns:
        Una permutación legal.

    Example:
        El paso 5 usa esta, no el intercambio: se mueve menos y sigue
        legal.
    """
    ruta = list(fuente)
    izquierdo, derecho = sorted(random.sample(range(len(ruta)), 2))
    ruta[izquierdo:derecho] = reversed(ruta[izquierdo:derecho])
    return ruta
# ------------------------------------------------------------------------------


# --- NUEVO (3) la auditoria de distancia --------------------------------------
random.seed(SEMILLA)
puntos = cargar_puntos()
base = ruta_aleatoria(len(puntos))
longitud_base = longitud_ruta(puntos, base)
delta_intercambio = [longitud_ruta(puntos, mutacion_intercambio(base)) - longitud_base for _ in range(PRUEBAS)]
delta_inversion = [longitud_ruta(puntos, mutacion_inversion(base)) - longitud_base for _ in range(PRUEBAS)]
# ------------------------------------------------------------------------------

print("Lección 10 - TSP 4: mutación")
print(f"Todos los mutantes de intercambio son legales: {all(sorted(mutacion_intercambio(base)) == list(range(len(puntos))) for _ in range(100))}")
print(f"Todos los mutantes de inversión son legales: {all(sorted(mutacion_inversion(base)) == list(range(len(puntos))) for _ in range(100))}")
print(f"Cambio medio en longitud, intercambio: {sum(delta_intercambio) / PRUEBAS:+.1f}")
print(f"Cambio medio en longitud, inversión: {sum(delta_inversion) / PRUEBAS:+.1f}")

FIGURAS.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 4))
ax.boxplot([delta_intercambio, delta_inversion], tick_labels=["intercambio", "inversión"], showfliers=False)
ax.axhline(0, color="black", linewidth=1)
ax.set(ylabel="cambio en longitud de ruta", title="Ambas mutaciones se mantienen legales pero se mueven diferente")
fig.tight_layout()
fig.savefig(FIGURAS / "tsp_04_mutacion.png", dpi=160)
plt.close(fig)

