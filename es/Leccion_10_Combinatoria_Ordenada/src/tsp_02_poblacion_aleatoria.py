"""
Lección 10 - TSP 2: Rutas legales aleatorias
=======================================
NUEVO EN ESTE PASO: ruta_aleatoria(), TAMANO_POBLACION, y un censo de longitud.

CAMBIOS RESPECTO A tsp_01_la_ruta.py
Introdúcelos en este orden:
    1. ruta_aleatoria()        barajar los índices de las ciudades sin perder ninguna
    2. TAMANO_POBLACION        definir el presupuesto de muestreo
    3. el censo                medir cuánto difieren todavía las rutas legales

Ejecútalo:  python tsp_02_poblacion_aleatoria.py

Las 500 rutas barajadas son legales y su distribución de longitudes es amplia. Aquí el cubo aleatorio ya nace legal: el problema no es la factibilidad, es la calidad.
"""
from math import dist
from pathlib import Path
import random
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt

SEMILLA = 60
DATOS = Path(__file__).with_name("att48_xy.txt")
FIGURAS = Path(__file__).resolve().parent.parent / "figures"


def cargar_puntos(ruta: Path = DATOS) -> List[Tuple[int, int]]:
    """Las 48 ciudades ATT.

    Args:
        ruta: ``att48_xy.txt``.

    Returns:
        48 pares (x, y).

    Example:
        500 barajas de estos 48 índices: todas legales.
    """
    return [tuple(map(int, linea.split())) for linea in ruta.read_text().splitlines() if linea.strip()]


def longitud_ruta(puntos: Sequence[Tuple[int, int]], ruta: Sequence[int]) -> float:
    """Longitud del ciclo. Aquí mide calidad, no legalidad: todas las rutas ya son legales.

    Args:
        puntos: coordenadas.
        ruta: permutación de 0..47.

    Returns:
        Longitud euclidiana cerrada.

    Example:
        500 rutas: min, media y max se imprimen. La dispersión es el
        punto: legal no significa parecido.
    """
    pares = zip(ruta, ruta[1:] + ruta[:1])
    return sum(dist(puntos[primera], puntos[segunda]) for primera, segunda in pares)


# --- NUEVO (1) ruta_aleatoria() -----------------------------------------------
def ruta_aleatoria(tamano: int) -> List[int]:
    """Baraja los índices. Shuffle no pierde ciudades: por eso 500/500 son legales.

    Args:
        tamano: número de ciudades, 48.

    Returns:
        Una permutación de 0..tamano-1.

    Example:
        SEMILLA = 60, 500 barajas: 500 permutaciones legales, longitudes
        muy distintas.
    """
    ruta = list(range(tamano))
    random.shuffle(ruta)
    return ruta
# ------------------------------------------------------------------------------


# --- NUEVO (2) TAMANO_POBLACION -----------------------------------------------
TAMANO_POBLACION = 500
# ------------------------------------------------------------------------------


# --- NUEVO (3) el censo -------------------------------------------------------
random.seed(SEMILLA)
puntos = cargar_puntos()
poblacion = [ruta_aleatoria(len(puntos)) for _ in range(TAMANO_POBLACION)]
longitudes = [longitud_ruta(puntos, ruta) for ruta in poblacion]
# ------------------------------------------------------------------------------

print("Lección 10 - TSP 2: rutas legales aleatorias")
print(f"Permutaciones legales: {sum(sorted(r) == list(range(len(puntos))) for r in poblacion)}/{TAMANO_POBLACION}")
print(f"Longitud de ruta: min={min(longitudes):,.1f}, media={sum(longitudes) / len(longitudes):,.1f}, max={max(longitudes):,.1f}")

FIGURAS.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(longitudes, bins=25, color="tab:blue")
ax.set(xlabel="longitud de la ruta cerrada", ylabel="rutas", title="Toda ruta es legal; la calidad aún varía")
fig.tight_layout()
fig.savefig(FIGURAS / "tsp_02_poblacion_aleatoria.png", dpi=160)
plt.close(fig)

