"""
Lección 11 - Coloreado 2: La línea base que provee la suerte
=============================================================
NUEVO EN ESTE PASO: poblacion_aleatoria() y una muestra de 500 coloreados.

Un coloreado aleatorio asigna cada uno de los tres colores de forma independiente, así que cada arista
tiene una posibilidad entre tres de romperse. Eso da una cantidad esperada de conflictos
de |E|/3 antes de ejecutar nada, y la muestra debería acercarse a ello. Lo que importa
es el otro extremo del histograma: el mejor de 500 coloreados aleatorios, y qué
tan lejos está todavía del 0 que exige el problema.

CAMBIOS RESPECTO A coloreado_01_el_grafo.py
Introdúcelos en este orden:
    1. poblacion_aleatoria()  obtener toda la población inicial a la vez
    2. la muestra             500 coloreados, sus conflictos y la línea teórica

Ejecútalo:  python coloreado_02_poblacion_aleatoria.py

500 coloreados aleatorios producen 0 legales. La media es 21.51 conflictos frente a |E|/3 = 21.33, y el mejor aún rompe 11 aristas (17.2%).
"""
from collections import Counter
from pathlib import Path
import random
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import igraph

SEMILLA = 7
TAMANO_POBLACION = 500
FIGURAS = Path(__file__).resolve().parent.parent / "figures"
PALETA = ["#d62728", "#2ca02c", "#1f77b4"]

# La instancia es de Gridin, mantenida literalmente para poder comparar la lección con el
# libro: 30 vértices, y las aristas a continuación.
ARISTAS: List[Tuple[int, int]] = [
    (0, 2), (3, 4), (3, 5), (0, 5), (1, 4), (2, 10), (14, 8), (15, 3), (16, 9), (17, 11), (18, 7),
    (19, 7), (6, 3), (7, 6), (8, 0), (11, 5), (11, 9), (12, 1), (13, 4), (20, 5), (21, 17), (22, 15),
    (23, 20), (24, 20), (25, 17), (26, 24), (27, 10), (28, 15), (29, 21), (12, 14), (16, 20), (29, 19),
    (27, 22), (26, 9), (25, 26), (23, 10), (6, 28), (14, 13), (0, 12), (3, 2), (19, 14), (22, 10), (1, 18),
    (7, 21), (15, 12), (11, 1), (23, 28), (6, 11), (9, 25), (17, 9), (24, 16), (27, 28), (18, 20), (19, 21),
    (1, 14), (22, 29), (17, 4), (8, 13), (7, 23), (16, 28), (5, 9), (12, 29), (27, 21), (1, 23)
]
COLORES = 3


def vertices(aristas: Sequence[Tuple[int, int]] = ARISTAS) -> List[int]:
    """Cada vértice mencionado por una arista, en orden."""
    return sorted({extremo for arista in aristas for extremo in arista})


def conflictos(coloreado: Sequence[int],
              aristas: Sequence[Tuple[int, int]] = ARISTAS) -> int:
    """Aristas cuyos dos extremos comparten un color. Cero es el objetivo completo."""
    return sum(1 for uno, dos in aristas if coloreado[uno] == coloreado[dos])


def grados(aristas: Sequence[Tuple[int, int]] = ARISTAS) -> Dict[int, int]:
    """Cuántas aristas tocan cada vértice."""
    contador: Counter = Counter()
    for uno, dos in aristas:
        contador[uno] += 1
        contador[dos] += 1
    return dict(contador)


def dibujar(eje, coloreado: Sequence[int], titulo: str) -> None:
    """Dibuja la instancia con el layout de igraph y el lienzo de matplotlib, para que la
    figura se guarde en lugar de mostrarse y el script se ejecute de forma desatendida."""
    grafo = igraph.Graph(n=len(vertices()), edges=ARISTAS, directed=False)
    layout = grafo.layout_kamada_kawai()
    rotas = {indice for indice, (uno, dos) in enumerate(ARISTAS)
              if coloreado[uno] == coloreado[dos]}
    igraph.plot(
        grafo, target=eje, layout=layout,
        vertex_color=[PALETA[color] for color in coloreado],
        vertex_size=22, vertex_label=[str(v) for v in vertices()],
        vertex_label_size=7, vertex_frame_width=0.5,
        edge_color=["#000000" if indice in rotas else "#bbbbbb"
                    for indice in range(len(ARISTAS))],
        edge_width=[2.2 if indice in rotas else 0.7 for indice in range(len(ARISTAS))],
    )
    eje.set_title(titulo)


# --- NUEVO (1) poblacion_aleatoria() ------------------------------------------
def poblacion_aleatoria(tamano: int) -> List[List[int]]:
    """Colores uniformes independientes para cada vértice de cada individuo.

    Args:
        tamano: 500 en este paso.

    Returns:
        Lista de cromosomas de 30 genes en {0, 1, 2}.

    Example:
        0/500 legales. Media 21.51 vs |E|/3 = 21.33. El mejor aún rompe
        11 aristas.
    """
    return [[random.randrange(COLORES) for _ in vertices()] for _ in range(tamano)]
# ------------------------------------------------------------------------------


# --- NUEVO (2) la muestra -----------------------------------------------------
todos_los_vertices = vertices()
random.seed(SEMILLA)
poblacion = poblacion_aleatoria(TAMANO_POBLACION)
conteos = [conflictos(coloreado) for coloreado in poblacion]
esperado = len(ARISTAS) / COLORES
legales = [coloreado for coloreado in poblacion if conflictos(coloreado) == 0]
mejor_indice = min(range(len(conteos)), key=lambda indice: conteos[indice])
# ------------------------------------------------------------------------------

print("Lección 11 - Coloreado 2: la línea base que provee la suerte")
print(f"Semilla {SEMILLA}, población {TAMANO_POBLACION}")
print(f"Coloreados legales obtenidos:  {len(legales)}")
print(f"Conflictos: min {min(conteos)}, media {sum(conteos) / len(conteos):.2f}, max {max(conteos)}")
print(f"Conflictos esperados |E|/{COLORES}: {esperado:.2f}")
print(f"La media de la muestra se desvía de la esperanza por "
      f"{abs(sum(conteos) / len(conteos) - esperado):.2f} aristas")
print(f"El mejor de {TAMANO_POBLACION} obtenidos aún rompe {min(conteos)} de {len(ARISTAS)} aristas "
      f"({min(conteos) / len(ARISTAS):.1%})")
print(f"Espacio de búsqueda: {COLORES ** len(todos_los_vertices):,} coloreados. A diferencia de la caja")
print("de ecuaciones, este no se puede recorrer, lo cual es el argumento para usar una búsqueda en primer lugar.")

FIGURAS.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(conteos, bins=range(min(conteos), max(conteos) + 2), color="#4c72b0", edgecolor="white")
ax.axvline(esperado, color="black", linestyle="--", label=f"|E|/{COLORES} = {esperado:.2f}")
ax.axvline(min(conteos), color="crimson", linestyle="--", label=f"mejor obtenido = {min(conteos)}")
ax.set(xlabel="aristas en conflicto", ylabel="coloreados",
       title=f"{TAMANO_POBLACION} coloreados aleatorios; {len(legales)} de ellos son legales")
ax.legend()
fig.tight_layout()
fig.savefig(FIGURAS / "coloreado_02_poblacion_aleatoria.png", dpi=160)
plt.close(fig)

