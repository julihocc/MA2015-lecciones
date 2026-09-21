"""
Lección 11 - Coloreado 3: Operadores que consultan la función de aptitud
========================================================================
NUEVO EN ESTE PASO: cruza_n_puntos(), los dos operadores basados en aptitud, y una
medición de todos ellos sobre la misma población.

En una meseta, un operador ciego es como lanzar una moneda. La respuesta de Gridin para este problema
es dejar que los operadores miren antes de saltar: una cruza que conserva los mejores
dos de {ambos hijos, ambos padres}, y una mutación que prueba algunos recoloreados
aleatorios y se queda con uno solo si reduce el conteo de conflictos. Ambos son
codiciosos (greedy), y la codicia es una elección real con un costo real, así que hay que medirlo en lugar de
solo afirmarlo.

CAMBIOS RESPECTO A coloreado_02_poblacion_aleatoria.py
Introdúcelos en este orden:
    1. cruza_n_puntos()                       el operador sencillo de dos puntos
    2. cruza_basada_en_aptitud_n_puntos()     los hijos deben superar a sus padres
    3. mutacion_basada_en_aptitud_cambio_aleatorio() conservar un recoloreado solo si ayuda
    4. el experimento                         los tres, en una sola población

Ejecútalo:  python coloreado_03_operadores_basados_en_aptitud.py

La cruza de dos puntos mueve la media +0.03; la basada en aptitud, −2.32. El recoloreado codicioso mueve −1.02 pero deja intactos 206 de 500, y su mejor (11) es peor que el de un recoloreado ciego (9).
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


def poblacion_aleatoria(tamano: int) -> List[List[int]]:
    """Colores uniformes independientes para cada vértice de cada individuo."""
    return [[random.randrange(COLORES) for _ in vertices()] for _ in range(tamano)]


# --- NUEVO (1) cruza_n_puntos() -----------------------------------------------
def cruza_n_puntos(primero: Sequence[int], segundo: Sequence[int],
                      puntos: int = 2) -> Tuple[List[int], List[int]]:
    """Intercambia segmentos alternos. Cualquier coloreado es legal como cromosoma, así que
    a diferencia de la lección 10 no hay nada que reparar - solo calidad que perder.

    Args:
        primero: padre.
        segundo: padre.
        puntos: 2.

    Returns:
        Dos hijos, tan legales como los padres y no necesariamente mejores.

    Example:
        Media de conflictos +0.03. En una meseta, un corte ciego es una
        moneda.
    """
    cortes = sorted(random.sample(range(1, len(primero) - 1), puntos) + [0, len(primero)])
    hijo_uno, hijo_dos = list(primero), list(segundo)
    for indice in range(1, puntos + 1, 2):
        izq, der = cortes[indice], cortes[indice + 1]
        hijo_uno[izq:der] = segundo[izq:der]
        hijo_dos[izq:der] = primero[izq:der]
    return hijo_uno, hijo_dos
# ------------------------------------------------------------------------------


# --- NUEVO (2) cruza_basada_en_aptitud_n_puntos() -----------------------------
def cruza_basada_en_aptitud_n_puntos(primero: Sequence[int], segundo: Sequence[int],
                                     puntos: int = 2) -> List[List[int]]:
    """Produce los dos hijos, luego devuelve los mejores dos de los cuatro. Por lo tanto,
    un par de padres nunca puede ser reemplazado por algo peor.

    Args:
        primero: padre.
        segundo: padre.
        puntos: 2.

    Returns:
        Los dos de menor ``conflictos`` entre {hijos, padres}.

    Example:
        Media −2.32. Compra promedio; cuesta evaluaciones extra (los
        cuatro se evalúan).
    """
    hijo_uno, hijo_dos = cruza_n_puntos(primero, segundo, puntos)
    familia = [list(hijo_uno), list(hijo_dos), list(primero), list(segundo)]
    return sorted(familia, key=conflictos)[:2]
# ------------------------------------------------------------------------------


# --- NUEVO (3) mutacion_basada_en_aptitud_cambio_aleatorio() ------------------
def mutacion_basada_en_aptitud_cambio_aleatorio(coloreado: Sequence[int],
                                          intentos: int = 3) -> List[int]:
    """Recolorea un vértice aleatorio, hasta `intentos` veces, y devuelve el primer
    intento que mejora estrictamente. De lo contrario, deja al individuo solo.

    Args:
        coloreado: 30 colores.
        intentos: 3 recoloreados.

    Returns:
        El mutante si mejoró; si no, una copia del original.

    Example:
        Media −1.02, pero 206 de 500 intactos. El mejor (11) es peor
        que el de un recoloreado ciego (9): la codicia vende la cola.
    """
    actual = conflictos(coloreado)
    for _ in range(intentos):
        mutante = list(coloreado)
        mutante[random.randrange(len(mutante))] = random.randrange(COLORES)
        if conflictos(mutante) < actual:
            return mutante
    return list(coloreado)
# ------------------------------------------------------------------------------


# --- NUEVO (4) el experimento -------------------------------------------------
def media(valores: Sequence[float]) -> float:
    """Promedio de una muestra. El experimento compara tres operadores con esto.

    Args:
        valores: conflictos por individuo.

    Returns:
        Media aritmética.

    Example:
        +0.03, −2.32, −1.02: las tres medias del paso 3.
    """
    return sum(valores) / len(valores)


todos_los_vertices = vertices()
random.seed(SEMILLA)
poblacion = poblacion_aleatoria(TAMANO_POBLACION)
antes = [conflictos(coloreado) for coloreado in poblacion]
pares = list(zip(poblacion[::2], poblacion[1::2]))

hijos_sencillos: List[List[int]] = []
for uno, dos in pares:
    hijos_sencillos += list(cruza_n_puntos(uno, dos))
hijos_dirigidos: List[List[int]] = []
for uno, dos in pares:
    hijos_dirigidos += cruza_basada_en_aptitud_n_puntos(uno, dos)

mutantes_ciegos = []
for coloreado in poblacion:
    mutante = list(coloreado)
    mutante[random.randrange(len(mutante))] = random.randrange(COLORES)
    mutantes_ciegos.append(mutante)
mutantes_dirigidos = [mutacion_basada_en_aptitud_cambio_aleatorio(c) for c in poblacion]
sin_cambios = sum(1 for antes_uno, despues_uno in zip(poblacion, mutantes_dirigidos)
                if antes_uno == despues_uno)
# ------------------------------------------------------------------------------

sencillos = [conflictos(hijo) for hijo in hijos_sencillos]
dirigidos = [conflictos(hijo) for hijo in hijos_dirigidos]
ciegos = [conflictos(mutante) for mutante in mutantes_ciegos]
codiciosos = [conflictos(mutante) for mutante in mutantes_dirigidos]

print("Lección 11 - Coloreado 3: operadores que consultan la función de aptitud")
print(f"Semilla {SEMILLA}, población {TAMANO_POBLACION}, {len(pares)} pares de padres")
print(f"Padres:                       media {media(antes):.2f} conflictos, mejor {min(antes)}")
print(f"Cruza sencilla de dos puntos:     media {media(sencillos):.2f} conflictos, mejor {min(sencillos)}")
print(f"Cruza basada en aptitud:      media {media(dirigidos):.2f} conflictos, mejor {min(dirigidos)}")
print(f"Recoloreado simple ciego:         media {media(ciegos):.2f} conflictos, mejor {min(ciegos)}")
print(f"Recoloreado basado en aptitud:       media {media(codiciosos):.2f} conflictos, mejor {min(codiciosos)}")
print(f"Recoloreado basado en aptitud dejó a {sin_cambios} de {TAMANO_POBLACION} individuos intactos, "
      f"habiendo intentado y rechazado cada recoloreado que probó")
print(f"Cruza sencilla mueve la media en {media(sencillos) - media(antes):+.2f} aristas; "
      f"la versión basada en aptitud en {media(dirigidos) - media(antes):+.2f}.")
print(f"Mejor individuo después de mutar: ciego {min(ciegos)}, codicioso {min(codiciosos)}; la codicia movió"
      f" la media en {media(codiciosos) - media(antes):+.2f} aristas y el mejor en {min(codiciosos) - min(ciegos):+d}.")
print(f"Ningún operador produjo un coloreado legal: "
      f"{sum(1 for c in sencillos + dirigidos + ciegos + codiciosos if c == 0)} de "
      f"{len(sencillos) + len(dirigidos) + len(ciegos) + len(codiciosos)} resultados tuvieron 0 conflictos.")
print("La codicia no es gratis, y el costo es visible en las dos funciones de arriba:")
print("cruza_basada_en_aptitud_n_puntos() cuenta conflictos en cuatro coloreados por")
print("par - dos hijos y ambos padres - donde el operador sencillo cuenta ninguno,")
print("y mutacion_basada_en_aptitud_cambio_aleatorio() cuenta uno más hasta tres más")
print(f"por individuo. En esta población eso es al menos {4 * len(pares):,} conteos extra")
print("solo para la cruza.")

FIGURAS.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.boxplot([antes, sencillos, dirigidos, ciegos, codiciosos], vert=True,
           tick_labels=["padres", "cruza\nsencilla", "cruza\ndirigida",
                        "recoloreado\nciego", "recoloreado\ndirigido"])
ax.set(ylabel="aristas en conflicto",
       title="Los operadores codiciosos mueven la población hacia abajo; ninguno llega a 0")
fig.tight_layout()
fig.savefig(FIGURAS / "coloreado_03_operadores_basados_en_aptitud.png", dpi=160)
plt.close(fig)

