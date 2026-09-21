"""
Lección 10 - TSP 1: Una ruta es un ordenamiento
==========================================
NUEVO EN ESTE PASO: cargar_puntos(), longitud_ruta(), y una ruta de referencia.

Un cromosoma del TSP contiene cada ciudad exactamente una vez. Sus genes no son opciones separadas: cambiar una posición cambia el significado de toda la ruta.

Ejecútalo:  python tsp_01_la_ruta.py

La instancia tiene 48 ciudades. La ruta identidad es legal pero larga: legalidad y calidad no son lo mismo.
"""
from math import dist
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt

SEMILLA = 60
DATOS = Path(__file__).with_name("att48_xy.txt")
FIGURAS = Path(__file__).resolve().parent.parent / "figures"


def cargar_puntos(ruta: Path = DATOS) -> List[Tuple[int, int]]:
    """Carga las coordenadas de las 48 ciudades ATT que vienen con la lección.

    Args:
        ruta: ``att48_xy.txt`` junto al script.

    Returns:
        48 pares (x, y) enteros.

    Example:
        48 ciudades, cromosoma de longitud 48. La Lección 12 reusa esta
        instancia, reescrita, no importada.
    """
    return [tuple(map(int, linea.split())) for linea in ruta.read_text().splitlines() if linea.strip()]


def longitud_ruta(puntos: Sequence[Tuple[int, int]], ruta: Sequence[int]) -> float:
    """Cierra el recorrido regresando de la ciudad final a la primera.

    Un TSP abierto sería más corto y mentiría: el viajante vuelve.

    Args:
        puntos: coordenadas de las 48 ciudades.
        ruta: permutación de 0..47.

    Returns:
        Longitud euclidiana del ciclo.

    Example:
        La ruta identidad es legal y larga. El paso 5 la compara contra
        el vecino más cercano, no contra esta.
    """
    pares = zip(ruta, ruta[1:] + ruta[:1])
    return sum(dist(puntos[primera], puntos[segunda]) for primera, segunda in pares)


puntos = cargar_puntos()
ruta = list(range(len(puntos)))
longitud = longitud_ruta(puntos, ruta)

print("Lección 10 - TSP 1: una ruta es un ordenamiento")
print(f"Ciudades: {len(puntos)}; longitud del cromosoma: {len(ruta)}")
print(f"La ruta identidad contiene cada ciudad una vez: {sorted(ruta) == list(range(len(puntos)))}")
print(f"Longitud de la ruta identidad: {longitud:,.1f}")

FIGURAS.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 5))
cerrada = ruta + ruta[:1]
ax.plot([puntos[i][0] for i in cerrada], [puntos[i][1] for i in cerrada], "o-", markersize=3)
ax.set(title="Una ruta legal aún puede ser muy larga", aspect="equal")
fig.tight_layout()
fig.savefig(FIGURAS / "tsp_01_la_ruta.png", dpi=160)
plt.close(fig)

