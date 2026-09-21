"""
Lección 11 - Coloreado 1: Cuando la restricción es el objetivo completo
=======================================================================
NUEVO EN ESTE PASO: ARISTAS, vertices(), conflictos(), dibujar() y un coloreado aleatorio.

El coloreado de grafos es lo opuesto al sistema de ecuaciones. Allí el objetivo era
una cantidad más o menos suave a minimizar; aquí lo único que se mide es una
restricción violada. La aptitud es el número negado de aristas cuyos dos extremos comparten
un color, por lo que un coloreado legal y un coloreado óptimo son el mismo objeto y
no hay nada que optimizar una vez que lo alcanzas.

Esa forma tiene una consecuencia con la que el resto de esta secuencia se sigue topando: cerca
de la respuesta el objetivo es plano. Un conflicto y dos conflictos se ven casi
idénticos para la búsqueda, y no se puede llegar a cero acercándose gradualmente.

Ejecútalo:  python coloreado_01_el_grafo.py

30 vértices, 64 aristas, grado medio 4.27, y 3^30 = 205,891,132,094,649 coloreados. Un aleatorio en la semilla 7 rompe 26 de las 64 aristas.
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
    """Cada vértice mencionado por una arista, en orden.

    Args:
        aristas: la instancia de Gridin, 64 aristas.

    Returns:
        30 índices 0..29.

    Example:
        3^30 = 205,891,132,094,649 coloreados. Eso no se enumera.
    """
    return sorted({extremo for arista in aristas for extremo in arista})


def conflictos(coloreado: Sequence[int],
              aristas: Sequence[Tuple[int, int]] = ARISTAS) -> int:
    """Aristas cuyos dos extremos comparten un color. Cero es el objetivo completo.

    Args:
        coloreado: 30 colores en {0, 1, 2}.
        aristas: las 64 aristas.

    Returns:
        Entero 0..64. Cero es legal y óptimo a la vez.

    Example:
        Semilla 7: un aleatorio rompe 26 de 64. Cerca de cero el paisaje
        es plano: 1 y 2 conflictos se ven casi iguales.
    """
    return sum(1 for uno, dos in aristas if coloreado[uno] == coloreado[dos])


def grados(aristas: Sequence[Tuple[int, int]] = ARISTAS) -> Dict[int, int]:
    """Cuántas aristas tocan cada vértice.

    Args:
        aristas: las 64 aristas.

    Returns:
        Diccionario vértice → grado. Grados de 3 a 6.

    Example:
        Grado medio 4.27. Nada aquí es un vértice aislado.
    """
    contador: Counter = Counter()
    for uno, dos in aristas:
        contador[uno] += 1
        contador[dos] += 1
    return dict(contador)


def dibujar(eje, coloreado: Sequence[int], titulo: str) -> None:
    """Dibuja la instancia con el layout de igraph y el lienzo de matplotlib, para que la
    figura se guarde en lugar de mostrarse y el script se ejecute de forma desatendida.

    Args:
        eje: Axes de matplotlib.
        coloreado: 30 colores.
        titulo: texto de la figura.

    Returns:
        None. Guarda vía el llamador.

    Example:
        Las 26 aristas rotas de la semilla 7 se pintan más gruesas.
    """
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


todos_los_vertices = vertices()
grado = grados()
random.seed(SEMILLA)
coloreado_aleatorio = [random.randrange(COLORES) for _ in todos_los_vertices]
aristas_rotas = [arista for arista in ARISTAS
                if coloreado_aleatorio[arista[0]] == coloreado_aleatorio[arista[1]]]

print("Lección 11 - Coloreado 1: cuando la restricción es el objetivo completo")
print(f"Vértices:            {len(todos_los_vertices)}")
print(f"Aristas:               {len(ARISTAS)} ({len({tuple(sorted(a)) for a in ARISTAS})} distintas)")
print(f"Grado:              min {min(grado.values())}, max {max(grado.values())}, "
      f"media {sum(grado.values()) / len(todos_los_vertices):.2f}")
print(f"Colores disponibles:   {COLORES}")
print(f"Cromosoma:          {len(todos_los_vertices)} genes, cada uno en 0..{COLORES - 1}, "
      f"{COLORES ** len(todos_los_vertices):,} coloreados en total")
print(f"Un coloreado aleatorio en semilla {SEMILLA}: {coloreado_aleatorio}")
print(f"  aristas en conflicto: {conflictos(coloreado_aleatorio)}")
print(f"  aptitud:           {-conflictos(coloreado_aleatorio)} (0 sería un coloreado legal)")
print(f"  primeras tres aristas rotas: {aristas_rotas[:3]}")
print("La aptitud está acotada por arriba por 0 y esa cota es la respuesta, no un objetivo.")

FIGURAS.mkdir(exist_ok=True)
fig, ejes = plt.subplots(1, 2, figsize=(12, 6))
dibujar(ejes[0], [2] * len(todos_los_vertices), "La instancia, sin colorear")
dibujar(ejes[1], coloreado_aleatorio,
     f"Un 3-coloreado aleatorio: {conflictos(coloreado_aleatorio)} aristas en conflicto (negrita)")
for eje in ejes:
    eje.set_axis_off()
fig.tight_layout()
fig.savefig(FIGURAS / "coloreado_01_el_grafo.png", dpi=160)
plt.close(fig)

