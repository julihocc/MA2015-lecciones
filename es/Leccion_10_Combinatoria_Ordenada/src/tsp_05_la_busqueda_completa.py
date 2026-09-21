"""
Lección 10 - TSP 5: La búsqueda ordenada completa
===========================================
NUEVO EN ESTE PASO: vecino_mas_cercano(), seleccion_torneo(), y ejecutar().

CAMBIOS RESPECTO A tsp_04_mutacion.py
Introdúcelos en este orden:
    1. vecino_mas_cercano()  establecer una línea base constructiva barata
    2. seleccion_torneo()    preferir rutas legales más cortas
    3. ejecutar()            combinar cruza ordenada, inversión y elitismo

Ejecútalo:  python tsp_05_la_busqueda_completa.py

La ruta del AG es legal pero 57.2% más larga que el vecino más cercano después de 10,100 evaluaciones. Ese fracaso es la lección, no un bug a reescribir.
"""
from math import dist
from pathlib import Path
import random
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt

SEMILLA = 60
DATOS = Path(__file__).with_name("att48_xy.txt")
FIGURAS = Path(__file__).resolve().parent.parent / "figures"
TAMANO_POBLACION = 100
GENERACIONES = 100
PROBABILIDAD_MUTACION = 0.25


def cargar_puntos(ruta: Path = DATOS) -> List[Tuple[int, int]]:
    """Las 48 ciudades ATT.

    Args:
        ruta: ``att48_xy.txt``.

    Returns:
        48 pares (x, y).

    Example:
        El AG queda 57.2% más largo que el vecino más cercano sobre
        estas mismas coordenadas.
    """
    return [tuple(map(int, linea.split())) for linea in ruta.read_text().splitlines() if linea.strip()]


def longitud_ruta(puntos: Sequence[Tuple[int, int]], ruta: Sequence[int]) -> float:
    """Longitud del ciclo. AG y vecino más cercano se comparan aquí.

    Args:
        puntos: coordenadas.
        ruta: permutación legal.

    Returns:
        Longitud euclidiana cerrada.

    Example:
        AG 57.2% más largo que el vecino más cercano tras 10,100
        evaluaciones.
    """
    return sum(dist(puntos[a], puntos[b]) for a, b in zip(ruta, ruta[1:] + ruta[:1]))


def ruta_aleatoria(tamano: int) -> List[int]:
    """Población inicial: 100 permutaciones, SEMILLA = 60.

    Args:
        tamano: 48.

    Returns:
        Una permutación legal.

    Example:
        El AG parte de aquí, no del vecino más cercano. Por eso perder
        contra esa línea base es informativo.
    """
    ruta = list(range(tamano)); random.shuffle(ruta); return ruta


def cruza_ordenada(primero: Sequence[int], segundo: Sequence[int]) -> List[int]:
    """Hijo legal. Sin esto, 10,100 evaluaciones producirían rutas ilegales.

    Args:
        primero: padre legal.
        segundo: padre legal.

    Returns:
        Una permutación.

    Example:
        La ruta final es legal. El 57.2% extra de longitud no es por
        ciudades duplicadas.
    """
    izquierdo, derecho = sorted(random.sample(range(len(primero)), 2))
    hijo = [-1] * len(primero); hijo[izquierdo:derecho] = primero[izquierdo:derecho]
    restantes = [ciudad for ciudad in segundo if ciudad not in hijo]
    huecos = list(range(derecho, len(primero))) + list(range(izquierdo))
    for indice, ciudad in zip(huecos, restantes): hijo[indice] = ciudad
    return hijo


def mutacion_inversion(fuente: Sequence[int]) -> List[int]:
    """Invierte un segmento. Probabilidad 0.25 por hijo.

    Args:
        fuente: ruta legal.

    Returns:
        Una permutación legal.

    Example:
        El operador más suave del paso 4. Aun así el AG pierde 57.2%
        contra el vecino más cercano.
    """
    ruta = list(fuente)
    izquierdo, derecho = sorted(random.sample(range(len(ruta)), 2))
    ruta[izquierdo:derecho] = reversed(ruta[izquierdo:derecho])
    return ruta


# --- NUEVO (1) vecino_mas_cercano() ------------------------------------------
def vecino_mas_cercano(puntos: Sequence[Tuple[int, int]], inicio: int = 0) -> List[int]:
    """Línea base constructiva: siempre el más cercano no visitado. Barata y legal.

    Un AG que no gana a esto no ha justificado su factura. 48 ciudades
    caben en un greedy; 10,100 evaluaciones del AG no.

    Args:
        puntos: coordenadas.
        inicio: ciudad de partida, 0.

    Returns:
        Una permutación legal.

    Example:
        El AG queda 57.2% más largo. Esa cifra es el veredicto, no un
        "casi".
    """
    ruta, no_vistos = [inicio], set(range(len(puntos))) - {inicio}
    while no_vistos:
        actual = ruta[-1]
        siguiente_ciudad = min(no_vistos, key=lambda ciudad: dist(puntos[actual], puntos[ciudad]))
        ruta.append(siguiente_ciudad); no_vistos.remove(siguiente_ciudad)
    return ruta
# ------------------------------------------------------------------------------


# --- NUEVO (2) seleccion_torneo() ---------------------------------------------
def seleccion_torneo(poblacion: List[List[int]], puntos: Sequence[Tuple[int, int]], tamano: int = 3) -> List[int]:
    """Minimiza longitud entre tres rutas legales.

    Args:
        poblacion: rutas actuales.
        puntos: coordenadas para evaluar.
        tamano: contendientes; 3.

    Returns:
        La más corta de las tres.

    Example:
        100 generaciones no alcanzan al greedy. La presión no es el
        problema: el vecino más cercano ya es una heurística fuerte.
    """
    return min(random.sample(poblacion, tamano), key=lambda ruta: longitud_ruta(puntos, ruta))
# ------------------------------------------------------------------------------


# --- NUEVO (3) ejecutar() -----------------------------------------------------
def ejecutar(puntos: Sequence[Tuple[int, int]]) -> Tuple[List[int], List[float]]:
    """100 generaciones, población 100, SEMILLA = 60. 10,100 evaluaciones.

    Args:
        puntos: las 48 ciudades.

    Returns:
        ``(mejor ruta, historia de longitudes)``.

    Example:
        Ruta legal, 57.2% más larga que el vecino más cercano. El
        fracaso se imprime con esas palabras.
    """
    random.seed(SEMILLA)
    poblacion = [ruta_aleatoria(len(puntos)) for _ in range(TAMANO_POBLACION)]
    mejor = min(poblacion, key=lambda ruta: longitud_ruta(puntos, ruta))
    historial = [longitud_ruta(puntos, mejor)]
    for _ in range(GENERACIONES):
        hijos = [mejor]
        while len(hijos) < TAMANO_POBLACION:
            hijo = cruza_ordenada(seleccion_torneo(poblacion, puntos), seleccion_torneo(poblacion, puntos))
            if random.random() < PROBABILIDAD_MUTACION:
                hijo = mutacion_inversion(hijo)
            hijos.append(hijo)
        poblacion = hijos
        mejor = min(poblacion, key=lambda ruta: longitud_ruta(puntos, ruta))
        historial.append(longitud_ruta(puntos, mejor))
    return mejor, historial
# ------------------------------------------------------------------------------


puntos = cargar_puntos()
linea_base = vecino_mas_cercano(puntos)
mejor, historial = ejecutar(puntos)
longitud_linea_base = longitud_ruta(puntos, linea_base)
longitud_mejor = longitud_ruta(puntos, mejor)

print("Lección 10 - TSP 5: la búsqueda ordenada completa")
print(f"Línea base de vecino más cercano: {longitud_linea_base:,.1f}")
print(f"Ruta del AG:                   {longitud_mejor:,.1f}")
print(f"AG versus línea base:         {(longitud_linea_base - longitud_mejor) / longitud_linea_base:+.1%}")
print(f"Ruta final es legal:          {sorted(mejor) == list(range(len(puntos)))}")
print(f"Evaluaciones de ruta:          {TAMANO_POBLACION * (GENERACIONES + 1):,}")

FIGURAS.mkdir(exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].plot(historial)
axes[0].axhline(longitud_linea_base, color="black", linestyle="--", label="vecino más cercano")
axes[0].set(xlabel="generación", ylabel="longitud de ruta", title="Historial de búsqueda")
axes[0].legend()
cerrada = mejor + mejor[:1]
axes[1].plot([puntos[i][0] for i in cerrada], [puntos[i][1] for i in cerrada], "o-", markersize=3)
axes[1].set(title="Mejor ruta del AG", aspect="equal")
fig.tight_layout()
fig.savefig(FIGURAS / "tsp_05_la_busqueda_completa.png", dpi=160)
plt.close(fig)

