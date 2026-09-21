"""
Lección 10 - TSP 3: La cruza debe preservar una permutación
=========================================================
NUEVO EN ESTE PASO: un_punto(), cruza_ordenada(), y un experimento de legalidad.

CAMBIOS RESPECTO A tsp_02_poblacion_aleatoria.py
Introdúcelos en este orden:
    1. un_punto()            exponer por qué un corte estilo binario duplica ciudades
    2. cruza_ordenada()      mantener un segmento y llenar huecos en el orden del otro padre
    3. el experimento        contar los hijos legales de ambos operadores

Ejecútalo:  python tsp_03_cruza_ordenada.py

La cruza de un punto produce 0/1,000 hijos legales. La cruza ordenada produce 1,000/1,000. El operador determina la legalidad del cromosoma.
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
        El experimento usa solo la longitud 48, no las coordenadas.
    """
    return [tuple(map(int, linea.split())) for linea in ruta.read_text().splitlines() if linea.strip()]


def longitud_ruta(puntos: Sequence[Tuple[int, int]], ruta: Sequence[int]) -> float:
    """Longitud del ciclo. Este paso no la usa: aquí se cuenta legalidad.

    Args:
        puntos: coordenadas.
        ruta: permutación, o no.

    Returns:
        Longitud euclidiana cerrada.

    Example:
        Un hijo de un punto ni siquiera es una permutación: medir su
        longitud mentiría.
    """
    return sum(dist(puntos[a], puntos[b]) for a, b in zip(ruta, ruta[1:] + ruta[:1]))


def ruta_aleatoria(tamano: int) -> List[int]:
    """Una permutación legal. Los padres del experimento salen de aquí.

    Args:
        tamano: 48.

    Returns:
        Una permutación de 0..47.

    Example:
        1,000 pares de padres legales. Un punto destruye esa legalidad;
        la cruza ordenada no.
    """
    ruta = list(range(tamano)); random.shuffle(ruta); return ruta


# --- NUEVO (1) un_punto() -----------------------------------------------------
def un_punto(primero: Sequence[int], segundo: Sequence[int]) -> List[int]:
    """El corte binario de la Lección 04. En una permutación duplica y pierde ciudades.

    Args:
        primero: padre legal.
        segundo: padre legal.

    Returns:
        Un hijo, casi nunca una permutación.

    Example:
        0/1,000 hijos legales. Un hijo roto típico tiene menos de 48
        ciudades distintas.
    """
    corte = random.randrange(1, len(primero))
    return list(primero[:corte]) + list(segundo[corte:])
# ------------------------------------------------------------------------------


# --- NUEVO (2) cruza_ordenada() ---------------------------------------------
def cruza_ordenada(primero: Sequence[int], segundo: Sequence[int]) -> List[int]:
    """Copia un segmento del primero y llena huecos en el orden del segundo.

    Así no se duplica ninguna ciudad: las que ya están en el segmento
    se saltan al recorrer al segundo padre.

    Args:
        primero: padre legal; dona el segmento.
        segundo: padre legal; dona el orden de las restantes.

    Returns:
        Una permutación legal.

    Example:
        1,000/1,000 hijos legales. El operador, no la reparación,
        garantiza la permutación.
    """
    izquierdo, derecho = sorted(random.sample(range(len(primero)), 2))
    hijo = [-1] * len(primero)
    hijo[izquierdo:derecho] = primero[izquierdo:derecho]
    restantes = [ciudad for ciudad in segundo if ciudad not in hijo]
    huecos = list(range(derecho, len(primero))) + list(range(0, izquierdo))
    for indice, ciudad in zip(huecos, restantes):
        hijo[indice] = ciudad
    return hijo
# ------------------------------------------------------------------------------


# --- NUEVO (3) el experimento -------------------------------------------------
random.seed(SEMILLA)
tamano = len(cargar_puntos())
ordinaria, ordenada = [], []
for _ in range(PRUEBAS):
    padres = ruta_aleatoria(tamano), ruta_aleatoria(tamano)
    ordinaria.append(un_punto(*padres))
    ordenada.append(cruza_ordenada(*padres))
legal = lambda ruta: sorted(ruta) == list(range(tamano))
# ------------------------------------------------------------------------------

print("Lección 10 - TSP 3: cruza ordenada")
print(f"Hijos legales por un punto: {sum(map(legal, ordinaria))}/{PRUEBAS}")
print(f"Hijos legales por cruza ordenada:   {sum(map(legal, ordenada))}/{PRUEBAS}")
print(f"Ciudades distintas en un hijo roto típico: {len(set(ordinaria[0]))}/{tamano}")

FIGURAS.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(["un punto", "ordenada"], [sum(map(legal, ordinaria)), sum(map(legal, ordenada))], color=["tab:red", "tab:green"])
ax.set(ylabel=f"hijos legales de {PRUEBAS}", title="El operador determina la legalidad del cromosoma")
fig.tight_layout()
fig.savefig(FIGURAS / "tsp_03_cruza_ordenada.png", dpi=160)
plt.close(fig)

