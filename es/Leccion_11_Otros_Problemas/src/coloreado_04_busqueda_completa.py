"""
Lección 11 - Coloreado 4: La búsqueda que no termina, y la que sí
=================================================================
NUEVO EN ESTE PASO: seleccion_rango_con_elite(), ejecutar(), coloreado_exacto() y un
barrido sobre cuatro semillas.

Este es el paso para el que se construyó la lección. El algoritmo genético se ensambla y
se ejecuta, y en la mayoría de las semillas se detiene a una o dos aristas de un coloreado legal y
se queda ahí por el resto de su presupuesto - la meseta que el paso 1 predijo.
Luego, diez líneas de backtracking resuelven la misma pregunta de inmediato, y también prueban
que dos colores son imposibles, lo cual ninguna cantidad de ejecución del AG podría haber
establecido.

El resultado a continuación es el medido, no el planeado. La secuencia fue
diseñada esperando que el AG llegara a cero, porque eso es lo que Gridin reporta.
Lo hace en una de las cuatro semillas probadas aquí. Ambos hechos son la lección.

CAMBIOS RESPECTO A coloreado_03_operadores_basados_en_aptitud.py
Introdúcelos en este orden:
    1. seleccion_rango_con_elite()  selección por rango, dos élites conservadas
    2. ejecutar()                   el bucle, deteniéndose en el momento que conflictos llegue a 0
    3. coloreado_exacto()           backtracking que responde la pregunta de inmediato
    4. el barrido de semillas       con qué frecuencia la búsqueda realmente llega a cero

Ejecútalo:  python coloreado_04_busqueda_completa.py

1 de 4 semillas alcanza un coloreado legal (semilla 11, generación 15). Las otras se estancan a 1 o 2 conflictos. El backtracking encuentra un 3-coloreado en 40 nodos y prueba que no hay 2-coloreado en 4 nodos.
"""
from collections import Counter
from pathlib import Path
import random
import time
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import igraph

SEMILLA = 7
TAMANO_POBLACION = 500
PROBABILIDAD_CRUZA = 0.5
PROBABILIDAD_MUTACION = 0.5
MAX_GENERACIONES = 200
TAMANO_ELITE = 2
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


def cruza_n_puntos(primero: Sequence[int], segundo: Sequence[int],
                      puntos: int = 2) -> Tuple[List[int], List[int]]:
    """Intercambia segmentos alternos. Cualquier coloreado es legal como cromosoma, así que
    a diferencia de la lección 10 no hay nada que reparar - solo calidad que perder."""
    cortes = sorted(random.sample(range(1, len(primero) - 1), puntos) + [0, len(primero)])
    hijo_uno, hijo_dos = list(primero), list(segundo)
    for indice in range(1, puntos + 1, 2):
        izq, der = cortes[indice], cortes[indice + 1]
        hijo_uno[izq:der] = segundo[izq:der]
        hijo_dos[izq:der] = primero[izq:der]
    return hijo_uno, hijo_dos


def cruza_basada_en_aptitud_n_puntos(primero: Sequence[int], segundo: Sequence[int],
                                     puntos: int = 2) -> List[List[int]]:
    """Produce los dos hijos, luego devuelve los mejores dos de los cuatro. Por lo tanto,
    un par de padres nunca puede ser reemplazado por algo peor."""
    hijo_uno, hijo_dos = cruza_n_puntos(primero, segundo, puntos)
    familia = [list(hijo_uno), list(hijo_dos), list(primero), list(segundo)]
    return sorted(familia, key=conflictos)[:2]


def mutacion_basada_en_aptitud_cambio_aleatorio(coloreado: Sequence[int],
                                          intentos: int = 3) -> List[int]:
    """Recolorea un vértice aleatorio, hasta `intentos` veces, y devuelve el primer
    intento que mejora estrictamente. De lo contrario, deja al individuo solo."""
    actual = conflictos(coloreado)
    for _ in range(intentos):
        mutante = list(coloreado)
        mutante[random.randrange(len(mutante))] = random.randrange(COLORES)
        if conflictos(mutante) < actual:
            return mutante
    return list(coloreado)


# --- NUEVO (1) seleccion_rango_con_elite() ------------------------------------
def seleccion_rango_con_elite(poblacion: List[List[int]],
                               tamano_elite: int = 2) -> List[List[int]]:
    """Selección por rango con elitismo. Los conteos de conflictos se encuentran en una banda estrecha, por lo que
    el rango - que ignora el tamaño de las brechas - es la presión sensata aquí.

    Args:
        poblacion: coloreados actuales.
        tamano_elite: 2.

    Returns:
        Nueva población del mismo tamaño.

    Example:
        En la meseta, 1 conflicto y 2 se ven iguales para la aptitud
        cruda. El rango no mira el tamaño de la brecha.
    """
    ordenada = sorted(poblacion, key=conflictos)
    distancia_rango = 1.0 / len(poblacion)
    rangos = [1.0 - indice * distancia_rango for indice in range(len(poblacion))]
    suma_rangos = sum(rangos)
    seleccionados = [list(individuo) for individuo in ordenada[:tamano_elite]]
    for _ in range(len(ordenada) - tamano_elite):
        umbral = random.random() * suma_rangos
        acumulado = 0.0
        for indice, rango in enumerate(rangos):
            acumulado += rango
            if acumulado > umbral:
                seleccionados.append(list(ordenada[indice]))
                break
    return seleccionados
# ------------------------------------------------------------------------------


# --- NUEVO (2) ejecutar() -----------------------------------------------------
def ejecutar(semilla: int) -> Tuple[List[int], int, List[int]]:
    """Devuelve el mejor coloreado encontrado, la generación alcanzada y el mejor
    conteo de conflictos después de cada generación.

    Args:
        semilla: 11, 7, 16 o 1 en el barrido.

    Returns:
        ``(mejor, generación, historia)``. Se detiene al llegar a 0.

    Example:
        Semilla 11 llega a 0 en la generación 15. Semillas 7 y 16 se
        quedan en 2; semilla 1 en 1. 1 de 4.
    """
    random.seed(semilla)
    poblacion = poblacion_aleatoria(TAMANO_POBLACION)
    mejor = min(poblacion, key=conflictos)
    historia = [conflictos(mejor)]
    generacion = 0
    while generacion < MAX_GENERACIONES and conflictos(mejor) > 0:
        generacion += 1
        padres = seleccion_rango_con_elite(poblacion, TAMANO_ELITE)
        cruzados: List[List[int]] = []
        for uno, dos in zip(padres[::2], padres[1::2]):
            if random.random() < PROBABILIDAD_CRUZA:
                cruzados += cruza_basada_en_aptitud_n_puntos(uno, dos)
            else:
                cruzados += [uno, dos]
        poblacion = [mutacion_basada_en_aptitud_cambio_aleatorio(individuo)
                      if random.random() < PROBABILIDAD_MUTACION else individuo
                      for individuo in cruzados]
        campeon = min(poblacion, key=conflictos)
        if conflictos(campeon) < conflictos(mejor):
            mejor = campeon
        historia.append(conflictos(mejor))
    return mejor, generacion, historia
# ------------------------------------------------------------------------------


# --- NUEVO (3) coloreado_exacto() ---------------------------------------------
def coloreado_exacto(k: int) -> Tuple[List[int], int]:
    """Búsqueda por backtracking para un k-coloreado adecuado. Devuelve el coloreado (o
    una lista vacía si no existe ninguno) y el número de nodos de búsqueda visitados.

    El corte de simetría - nunca abrir un color numerado a más de uno por encima del
    más alto ya usado - es lo que mantiene esto en las decenas de nodos.

    Args:
        k: número de colores. 3 existe; 2 no.

    Returns:
        ``(coloreado o [], nodos visitados)``.

    Example:
        3-coloreado en 40 nodos; no hay 2-coloreado en 4 nodos. 0.69 ms
        para ambos. El AG no puede afirmar imposibilidad.
    """
    vecinos: Dict[int, List[int]] = {v: [] for v in vertices()}
    for uno, dos in ARISTAS:
        vecinos[uno].append(dos)
        vecinos[dos].append(uno)
    asignacion = [-1] * len(vertices())
    visitados = 0

    def extender() -> bool:
        nonlocal visitados
        visitados += 1
        vertices_abiertos = [v for v in vertices() if asignacion[v] == -1]
        if not vertices_abiertos:
            return True
        # Más restringido primero: el vértice con menos colores aún libres.
        elegido = min(vertices_abiertos,
                     key=lambda v: (-len({asignacion[u] for u in vecinos[v]
                                          if asignacion[u] >= 0}),
                                    -len(vecinos[v])))
        prohibidos = {asignacion[u] for u in vecinos[elegido] if asignacion[u] >= 0}
        techo = min(k, max(asignacion) + 2)
        for color in range(techo):
            if color in prohibidos:
                continue
            asignacion[elegido] = color
            if extender():
                return True
            asignacion[elegido] = -1
        return False

    return (list(asignacion), visitados) if extender() else ([], visitados)
# ------------------------------------------------------------------------------


# --- NUEVO (4) el barrido de semillas -----------------------------------------
SEMILLAS = [7, 1, 16, 11]
inicio_ag = time.perf_counter()
resultados = {semilla: ejecutar(semilla) for semilla in SEMILLAS}
segundos_ag = time.perf_counter() - inicio_ag
resueltos = [semilla for semilla, (mejor, _, _) in resultados.items() if conflictos(mejor) == 0]

inicio_exacto = time.perf_counter()
adecuado, nodos_tres = coloreado_exacto(3)
dos_colores, nodos_dos = coloreado_exacto(2)
segundos_exacto = time.perf_counter() - inicio_exacto
# ------------------------------------------------------------------------------

todos_los_vertices = vertices()
print("Lección 11 - Coloreado 4: la búsqueda que no termina, y la que sí")
print(f"Población {TAMANO_POBLACION}, límite de generaciones {MAX_GENERACIONES}, "
      f"élite {TAMANO_ELITE}, semillas {SEMILLAS}")
for semilla in SEMILLAS:
    mejor, generacion, historia = resultados[semilla]
    print(f"  semilla {semilla:<3} se detuvo en la generación {generacion:<4} "
          f"mejor {conflictos(mejor)} conflictos   "
          f"(después de 10 generaciones ya estaba en {historia[min(10, len(historia) - 1)]})")
print(f"Alcanzó un coloreado legal:  {len(resueltos)} de {len(SEMILLAS)} semillas {resueltos}")
print(f"Tiempo de pared búsqueda gen.:   {segundos_ag:.1f} s para {len(SEMILLAS)} ejecuciones")
print()
print("La misma pregunta, respondida exactamente por backtracking:")
print(f"  3-coloreado adecuado:       {'encontrado' if adecuado else 'no existe ninguno'}, "
      f"{nodos_tres} nodos de búsqueda")
print(f"  conflictos en él:          {conflictos(adecuado) if adecuado else 'n/a'}")
print(f"  2-coloreado adecuado:       {'encontrado' if dos_colores else 'no existe ninguno'}, "
      f"{nodos_dos} nodos de búsqueda")
print(f"  tiempo de pared para ambos:       {segundos_exacto * 1000:.2f} ms")
print()
print(f"Así que la instancia es 3-coloreable, y el algoritmo genético la falló en "
      f"{len(SEMILLAS) - len(resueltos)} de {len(SEMILLAS)} semillas.")
print("La meseta es la razón. Una vez que la población está en uno o dos conflictos, casi")
print("cada recoloreado es neutral o peor, la mutación basada en aptitud rechaza")
print("casi todo lo que intenta, y el elitismo mantiene quieta a la población.")
print("Al leer la ejecución sola, no podrías distinguir una búsqueda atascada de una instancia")
print("imposible - el problema de la lección 08, en un caso donde la verdad está a una llamada de distancia.")

FIGURAS.mkdir(exist_ok=True)
fig, ejes = plt.subplots(1, 2, figsize=(12, 5))
for semilla in SEMILLAS:
    _, _, historia = resultados[semilla]
    ejes[0].plot(historia, label=f"semilla {semilla}")
ejes[0].axhline(0, color="black", linestyle="--", label="coloreado legal")
ejes[0].set(xlabel="generación", ylabel="aristas en conflicto",
            title=f"{len(resueltos)} de {len(SEMILLAS)} semillas llegan a cero")
ejes[0].legend(fontsize=8)
dibujar(ejes[1], adecuado, f"El 3-coloreado exacto: {conflictos(adecuado)} conflictos, "
                      f"{nodos_tres} nodos de búsqueda")
ejes[1].set_axis_off()
fig.tight_layout()
fig.savefig(FIGURAS / "coloreado_04_busqueda_completa.png", dpi=160)
plt.close(fig)

