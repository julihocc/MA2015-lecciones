"""
Lección 04 - Paso 6: Los mismos operadores, en un cromosoma que es una permutación
====================================================================================
NUEVO EN ESTE PASO: los nueve paradas, longitud_recorrido(), es_legal_recorrido().

Nada de los operadores cambia en este paso. El cromosoma cambia: nueve genes
en lugar de seis, y los genes son las nueve paradas de una ronda de entregas,
cada una de las cuales debe aparecer exactamente una vez.

Cada operador construido hasta ahora recibe los nuevos padres sin modificar,
y la columna `legal` del instrumento enseña. Este es el punto hacia el que
la lección ha estado caminando: los operadores de cruza no son herramientas
intercambiables que difieren en fuerza, están divididos en familias por la
representación que asumen, y una cruza de un punto ingenua en una ruta no
produce una solución pobre - produce un cromosoma que no es una solución en absoluto.

CAMBIOS RESPECTO A descendencia_05_guiada_por_aptitud.py
Introdúcelos en este orden:
    1. las nueve paradas         un cromosoma que es una ruta, no un punto en una caja
    2. longitud_recorrido()      lo que vale una ruta
    3. es_legal_recorrido()      la legalidad deja de ser propiedad de un gen
    4. los mismos operadores     ejecuta los pasos 2 al 4 en las rutas y lee la columna legal
    5. la figura del mapa        las dos rutas parentales, y un hijo que no es una ruta

Ejecútalo: python descendencia_06_la_division.py

En rutas, un punto/mezcla/lineal son legales 0.00%; uniforme 0.95%,
y esos pocos legales eran padres devueltos.
"""
import math
import random
import statistics
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt


SEMILLA = 3                         # semilla de Gridin para el capítulo 4; cada medición la reinicia
CANTIDAD_GENES = 6                  # un cromosoma tiene seis números reales
GEN_MIN, GEN_MAX = 0.0, 10.0       # la caja dentro de la cual debe permanecer un gen
ENSAYOS = 2000                      # extracciones por medición, para porcentajes estables
FIGURAS = Path(__file__).resolve().parent.parent / "figures"

Cromosoma = list[float]
Operador = Callable[[list, list], tuple[list, list]]
Legalidad = Callable[[list], bool]


def crear_padres() -> tuple[Cromosoma, Cromosoma]:
    """Los dos padres que recibe cada operador de esta lección.

    Args:
        (sin argumentos).

    Returns:
        tuple[Cromosoma, Cromosoma].

    Example:
        SEMILLA = 3 produce [2.38 5.44 3.70 6.04 6.26 0.66] y [0.13 8.37 2.59 2.34 9.96 4.70].
    """
    random.seed(SEMILLA)
    p1 = [round(random.uniform(GEN_MIN, GEN_MAX), 2) for _ in range(CANTIDAD_GENES)]
    p2 = [round(random.uniform(GEN_MIN, GEN_MAX), 2) for _ in range(CANTIDAD_GENES)]
    return p1, p2


def es_legal_real(cromosoma: Cromosoma) -> bool:
    """Un cromosoma de valores reales es legal cuando cada gen está dentro de la caja.

    Args:
        cromosoma.

    Returns:
        bool.

    Example:
        En rutas, un punto/mezcla/lineal son legales 0.00%; uniforme 0.95%.
    """
    return all(GEN_MIN <= gen <= GEN_MAX for gen in cromosoma)


def mostrar(cromosoma) -> str:
    """Un cromosoma en una línea, para que padres e hijos queden alineados en columnas.

    Args:
        cromosoma.

    Returns:
        str.

    Example:
        En rutas, un punto/mezcla/lineal son legales 0.00%; uniforme 0.95%.
    """
    partes = [f"{gen:6.2f}" if isinstance(gen, float) else f"{gen:>6d}"
              for gen in cromosoma]
    return "[" + " ".join(partes) + "]"


def cruza_clon(padre1: list, padre2: list) -> tuple[list, list]:
    """La base de cero alcance: los hijos SON los padres.

    Args:
        padre1, padre2.

    Returns:
        tuple[list, list].

    Example:
        2 hijos, 0.00% genes nuevos, 100.00% clones, 100.00% legales.
    """
    return list(padre1), list(padre2)


def reporte_descendencia(etiqueta: str, operador: Operador,
                         padre1: list, padre2: list,
                         es_legal: Legalidad, ensayos: int = ENSAYOS) -> dict:
    """Ejecuta un operador `ensayos` veces sobre el mismo par y describe la nube.

    Args:
        etiqueta, operador, padre1, padre2, es_legal, ensayos.

    Returns:
        dict.

    Example:
        2000 extracciones desde SEMILLA = 3; un punto alcanza 10 hijos.
    """
    random.seed(SEMILLA)
    vistos: set[tuple] = set()
    posiciones = valores_nuevos = clones = legales = producidos = 0
    for _ in range(ensayos):
        for hijo in operador(padre1, padre2):
            producidos += 1
            vistos.add(tuple(hijo))
            if list(hijo) == list(padre1) or list(hijo) == list(padre2):
                clones += 1
            if es_legal(hijo):
                legales += 1
            for indice, gen in enumerate(hijo):
                posiciones += 1
                if gen != padre1[indice] and gen != padre2[indice]:
                    valores_nuevos += 1
    return {"etiqueta": etiqueta, "hijos": len(vistos), "nuevos": 100 * valores_nuevos / posiciones,
            "clon": 100 * clones / producidos, "legal": 100 * legales / producidos,
            "alcanzables": vistos}


def imprimir_encabezado() -> None:
    """Un encabezado para cada tabla de comparación de los pasos 1 al 7.

    Args:
        (sin argumentos).

    Returns:
        None. Imprime o guarda figura.

    Example:
        En rutas, un punto/mezcla/lineal son legales 0.00%; uniforme 0.95%.
    """
    print(f"{'operador':28s} {'hijos':>8} {'genes':>7} {'clon':>7} {'legal':>7}")
    print(f"{'':28s} {'alcanz.':>8} {'nuevos':>7} {'de un':>7} {'':>7}")
    print(f"{'':28s} {'':>8} {'':>7} {'padre':>7} {'':>7}")


def imprimir_fila(fila: dict) -> None:
    """imprimir_fila.

    Args:
        fila.

    Returns:
        None. Imprime o guarda figura.

    Example:
        En rutas, un punto/mezcla/lineal son legales 0.00%; uniforme 0.95%.
    """
    print(f"{fila['etiqueta']:28s} {fila['hijos']:8d} {fila['nuevos']:6.2f}% "
          f"{fila['clon']:6.2f}% {fila['legal']:6.2f}%")


def cruza_un_punto(padre1: list, padre2: list) -> tuple[list, list]:
    """Corta ambos padres en el mismo punto e intercambia las colas.

    Args:
        padre1, padre2.

    Returns:
        tuple[list, list].

    Example:
        2000 extracciones producen 10 hijos distintos; 0.00% genes nuevos.
    """
    punto = random.randint(1, len(padre1) - 1)
    hijo1, hijo2 = list(padre1), list(padre2)
    hijo1[punto:], hijo2[punto:] = padre2[punto:], padre1[punto:]
    return hijo1, hijo2


N_PUNTOS = (2, 3)


def cruza_n_puntos(padre1: list, padre2: list, n: int) -> tuple[list, list]:
    """Corta n veces e intercambia segmentos alternos.

    La cruza canónica permite cortes 1..L-1; esta implementación conserva la
    decisión de Gridin y muestrea range(1, L-1), por lo que excluye L-1.

    Args:
        padre1, padre2, n.

    Returns:
        tuple[list, list].

    Example:
        2 puntos alcanzan 12 hijos; 3 puntos alcanzan 8.
    """
    # Gridin extrae de range(1, len-1): el último gen nunca abre un segmento.
    puntos = sorted(random.sample(range(1, len(padre1) - 1), n) + [0, len(padre1)])
    hijo1, hijo2 = list(padre1), list(padre2)
    for i in range(n + 1):
        if i % 2 == 0:
            continue
        hijo1[puntos[i]:puntos[i + 1]] = padre2[puntos[i]:puntos[i + 1]]
        hijo2[puntos[i]:puntos[i + 1]] = padre1[puntos[i]:puntos[i + 1]]
    return hijo1, hijo2


TASA_UNIFORME = 0.5


def cruza_uniforme(padre1: list, padre2: list,
                   tasa: float = TASA_UNIFORME) -> tuple[list, list]:
    """Decide gen a gen de qué padre proviene.

    Args:
        padre1, padre2, tasa.

    Returns:
        tuple[list, list].

    Example:
        Alcanza 64 de 64 combinaciones; 3.20% de los hijos son un padre.
    """
    hijo1, hijo2 = list(padre1), list(padre2)
    for i in range(len(padre1)):
        if random.random() < tasa:
            hijo1[i], hijo2[i] = padre2[i], padre1[i]
    return hijo1, hijo2


def acotar(gen: float, bajo: float = GEN_MIN, alto: float = GEN_MAX) -> float:
    """La aritmética sobre genes puede producir cualquier valor; el espacio de búsqueda tiene bordes.

    Args:
        gen, bajo, alto.

    Returns:
        float.

    Example:
        acotar(11.4) == 10.0 en la caja [0, 10] de esta lección.
    """
    return max(bajo, min(alto, gen))


ALFAS_MEZCLA = (0.0, 0.5)


def cruza_mezcla(padre1: Cromosoma, padre2: Cromosoma, alfa: float,
                 mantener_dentro: bool = True) -> tuple[Cromosoma, Cromosoma]:
    """Cruza de mezcla (BLX-alfa): extrae cada gen hijo de un intervalo ampliado.

    Para el intervalo parental [m, M] de ancho d > 0, cada gen crudo es uniforme en
    [m-alfa*d, M+alfa*d]; su probabilidad exacta de salir del intervalo es
    2*alfa/(1+2*alfa). Una corrida finita mide alcance muestreado, no soporte.

    Args:
        padre1, padre2, alfa, mantener_dentro.

    Returns:
        tuple[Cromosoma, Cromosoma].

    Example:
        alfa = 0.5: 99.57% genes nuevos, 50.08% legal hasta acotar().
    """
    hijo1, hijo2 = list(padre1), list(padre2)
    for i in range(len(padre1)):
        distancia = abs(padre2[i] - padre1[i])
        bajo = min(padre1[i], padre2[i]) - alfa * distancia
        alto = max(padre1[i], padre2[i]) + alfa * distancia
        hijo1[i] = round(bajo + random.random() * (alto - bajo), 2)
        hijo2[i] = round(bajo + random.random() * (alto - bajo), 2)
    if mantener_dentro:
        hijo1 = [acotar(gen) for gen in hijo1]
        hijo2 = [acotar(gen) for gen in hijo2]
    return hijo1, hijo2


ALFA_LINEAL = 0.3


def cruza_lineal(padre1: Cromosoma, padre2: Cromosoma,
                 alfa: float = ALFA_LINEAL) -> tuple[Cromosoma, Cromosoma]:
    """Avanza una fracción fija alfa a lo largo de la línea que une a los dos padres.

    Args:
        padre1, padre2, alfa.

    Returns:
        tuple[Cromosoma, Cromosoma].

    Example:
        100.00% genes nuevos y exactamente 2 hijos alcanzables.
    """
    hijo1, hijo2 = list(padre1), list(padre2)
    for i in range(len(padre1)):
        diferencia = padre2[i] - padre1[i]
        hijo1[i] = acotar(round(padre1[i] + alfa * diferencia, 2))
        hijo2[i] = acotar(round(padre2[i] - alfa * diferencia, 2))
    return hijo1, hijo2


def objetivo(cromosoma: Cromosoma) -> float:
    """Lo que vale un cromosoma de valores reales. Mayor es mejor.

    Args:
        cromosoma.

    Returns:
        float.

    Example:
        El máximo global del seno está cerca de x = +1.372, f = +0.706.
    """
    return sum(math.sin(g) - 0.2 * abs(g) for g in cromosoma) / len(cromosoma)


def cruza_guiada_por_aptitud(padre1: Cromosoma, padre2: Cromosoma,
                              alfa: float) -> tuple[Cromosoma, Cromosoma]:
    """Mezcla, luego conserva los mejores dos de los cuatro cromosomas involucrados.

    Args:
        padre1, padre2, alfa.

    Returns:
        tuple[Cromosoma, Cromosoma].

    Example:
        Peor que los padres 0.00%; 16.50% de las extracciones los devuelve.
    """
    hijo1, hijo2 = cruza_mezcla(padre1, padre2, alfa)
    clasificados = sorted([list(padre1), list(padre2), hijo1, hijo2],
                          key=objetivo, reverse=True)
    return clasificados[0], clasificados[1]


# --- NUEVO (1) las nueve paradas ----------------------------------------------
CANTIDAD_PARADAS = 9              # nueve paradas de entrega, como en order.py de Gridin
PARADAS_XY = [(4.5, 5.6), (9.2, 4.7), (5.1, 5.9), (1.8, 5.1), (6.3, 7.9),
              (0.9, 3.0), (0.9, 8.1), (6.9, 0.4), (9.8, 9.6)]

Recorrido = list[int]


def crear_recorridos() -> tuple[Recorrido, Recorrido]:
    """Dos rutas por las nueve paradas - los padres de la segunda mitad.

    Una ruta es también un cromosoma: nueve genes, uno por posición en la ronda.
    Lo que ha cambiado no es la longitud ni el tipo sino la RESTRICCIÓN: los nueve
    genes tienen que ser las nueve paradas, cada una exactamente una vez.

    Args:
        (sin argumentos).

    Returns:
        tuple[Recorrido, Recorrido].

    Example:
        SEMILLA = 3, nueve paradas: el tamaño de order.py de Gridin.
    """
    random.seed(SEMILLA)
    return (random.sample(range(1, CANTIDAD_PARADAS + 1), CANTIDAD_PARADAS),
            random.sample(range(1, CANTIDAD_PARADAS + 1), CANTIDAD_PARADAS))
# ------------------------------------------------------------------------------


# --- NUEVO (2) longitud_recorrido() -------------------------------------------
def longitud_recorrido(recorrido: Recorrido) -> float:
    """Distancia total de una ronda cerrada por las paradas. Menor es mejor.

    Args:
        recorrido.

    Returns:
        float.

    Example:
        Los hijos OX1 promedian 52.95 frente a 51.23 de 2000 rutas aleatorias.
    """
    return sum(math.dist(PARADAS_XY[recorrido[i] - 1], PARADAS_XY[recorrido[(i + 1) % len(recorrido)] - 1])
               for i in range(len(recorrido)))
# ------------------------------------------------------------------------------


# --- NUEVO (3) es_legal_recorrido() -------------------------------------------
def es_legal_recorrido(cromosoma) -> bool:
    """Una ruta es legal cuando visita cada una de las nueve paradas exactamente una vez.

    Compara esto con es_legal_real(): allí, la legalidad era una propiedad de cada
    gen por sí solo. Aquí es una propiedad del cromosoma EN SU CONJUNTO, y ningún
    operador que trate los genes de forma independiente puede garantizar preservarla.

    Args:
        cromosoma.

    Returns:
        bool.

    Example:
        Un punto, n puntos, mezcla y lineal: 0.00% de hijos legales.
    """
    return sorted(cromosoma) == list(range(1, CANTIDAD_PARADAS + 1))


def fallos_recorrido(cromosoma) -> tuple[list, list]:
    """Qué paradas repite una ruta rota, y cuáles nunca visita.

    Args:
        cromosoma.

    Returns:
        tuple[list, list].

    Example:
        En rutas, un punto/mezcla/lineal son legales 0.00%; uniforme 0.95%.
    """
    repetidas = sorted({s for s in cromosoma if list(cromosoma).count(s) > 1})
    faltantes = [s for s in range(1, CANTIDAD_PARADAS + 1) if s not in list(cromosoma)]
    return repetidas, faltantes
# ------------------------------------------------------------------------------


padre1, padre2 = crear_padres()
print(f"padre 1   {mostrar(padre1)}")
print(f"padre 2   {mostrar(padre2)}")


# --- NUEVO (4) los mismos operadores ------------------------------------------
recorrido1, recorrido2 = crear_recorridos()
print("\nLos mismos dos padres, en la otra representación - dos rondas de entregas:")
print(f"    ruta 1   {mostrar(recorrido1)}   longitud {longitud_recorrido(recorrido1):.2f}")
print(f"    ruta 2   {mostrar(recorrido2)}   longitud {longitud_recorrido(recorrido2):.2f}")
print(f"    legal: {es_legal_recorrido(recorrido1)} y {es_legal_recorrido(recorrido2)}")

print("\nAhora córtalos una vez e intercambia las colas, exactamente como en el paso 2:")
rotos = 0
for punto in range(1, CANTIDAD_PARADAS):
    hijo = recorrido1[:punto] + recorrido2[punto:]
    repetidas, faltantes = fallos_recorrido(hijo)
    if not es_legal_recorrido(hijo):
        rotos += 1
    print(f"    corte después de la parada {punto}:  {mostrar(hijo)}   "
          f"visita {repetidas} dos veces, nunca visita {faltantes}")
print(f"\n{rotos} de los {CANTIDAD_PARADAS - 1} cortes posibles producen algo que")
print("no es una ruta. No una ruta mala - no es una ruta en absoluto: a la furgoneta")
print("se le dice que llame a una dirección dos veces y a otra nunca. No hay función")
print("de aptitud que pueda puntuar esto, y ningún ajuste de parámetros que ayude.")

filas_recorrido = [
    reporte_descendencia("clon (sin cruza)", cruza_clon,
                         recorrido1, recorrido2, es_legal_recorrido),
    reporte_descendencia("un punto", cruza_un_punto,
                         recorrido1, recorrido2, es_legal_recorrido),
    reporte_descendencia(f"{N_PUNTOS[-1]} puntos",
                         lambda a, b: cruza_n_puntos(a, b, N_PUNTOS[-1]),
                         recorrido1, recorrido2, es_legal_recorrido),
    reporte_descendencia(f"uniforme (tasa {TASA_UNIFORME})", cruza_uniforme,
                         recorrido1, recorrido2, es_legal_recorrido),
    reporte_descendencia(f"mezcla alfa={ALFAS_MEZCLA[-1]} + acotar",
                         lambda a, b: cruza_mezcla(a, b, ALFAS_MEZCLA[-1]),
                         recorrido1, recorrido2, es_legal_recorrido),
    reporte_descendencia(f"lineal alfa={ALFA_LINEAL}", cruza_lineal,
                         recorrido1, recorrido2, es_legal_recorrido),
]
print(f"\nCada operador de los pasos 2 al 4, en las dos rutas, {ENSAYOS} extracciones cada uno:")
imprimir_encabezado()
for fila in filas_recorrido:
    imprimir_fila(fila)

fila_uniforme = next(f for f in filas_recorrido if f["etiqueta"].startswith("uniforme"))
random.seed(SEMILLA)
legal_y_nuevo = 0
for _ in range(ENSAYOS):
    for hijo in cruza_uniforme(recorrido1, recorrido2):
        if es_legal_recorrido(hijo) and hijo != recorrido1 and hijo != recorrido2:
            legal_y_nuevo += 1
print(f"\nLa columna `legal` es todo el paso. La cruza uniforme es el único")
print(f"operador aquí que alguna vez produce una ruta legal - {fila_uniforme['legal']:.2f}%")
print(f"de las veces - y su columna `clon` es {fila_uniforme['clon']:.2f}%. Esas")
print("son las mismas extracciones: a lo largo de toda la ejecución, el número de rutas legales que")
print(f"no eran simplemente un padre devuelto es {legal_y_nuevo}. La cruza uniforme")
print("es legal aquí exactamente cuando no hace nada.")

random.seed(SEMILLA)
ejemplo = cruza_mezcla(recorrido1, recorrido2, ALFAS_MEZCLA[-1])[0]
fraccion = next(g for g in ejemplo if g != int(g))
print("\nLa mezcla hace algo peor que ilegal - es absurdo:")
print(f"    {mostrar(ejemplo)}")
print(f"La parada {fraccion} no existe. Promediar dos números de parada es aritmética sobre")
print("una etiqueta, y una etiqueta no es una cantidad. Los operadores de los pasos 2 y 3 al")
print("menos produjeron números de parada reales; la mezcla ni siquiera hace eso.")

print("\nLo que cambió no es la longitud del cromosoma ni el tipo de sus genes.")
print("Es DÓNDE VIVE LA LEGALIDAD. Para un cromosoma de valores reales la legalidad")
print("es una propiedad de cada gen por sí solo, así que un operador que trata los genes")
print("de forma independiente no puede romperla. Para una ruta es una propiedad del")
print("cromosoma completo - los nueve genes deben ser las nueve paradas - y cada operador")
print("de esta lección hasta ahora trata los genes de forma independiente. Esa es la")
print("división, y ningún parámetro la cruza. El paso 7 construye el operador que lo hace.")
# ------------------------------------------------------------------------------


# --- NUEVO (5) la figura del mapa --------------------------------------------
fig, ejes = plt.subplots(1, 3, figsize=(13, 4.4))
random.seed(SEMILLA)
punto = 4
hijo = recorrido1[:punto] + recorrido2[punto:]
repetidas, faltantes = fallos_recorrido(hijo)
paneles = [(recorrido1, "ruta 1 (legal)", "tab:blue"),
           (recorrido2, "ruta 2 (legal)", "tab:red"),
           (hijo, f"hijo de un punto, corte después de la parada {punto}", "tab:green")]
for eje, (ruta, titulo, color) in zip(ejes, paneles):
    xs = [PARADAS_XY[s - 1][0] for s in ruta] + [PARADAS_XY[ruta[0] - 1][0]]
    ys = [PARADAS_XY[s - 1][1] for s in ruta] + [PARADAS_XY[ruta[0] - 1][1]]
    eje.plot(xs, ys, "-", color=color, linewidth=1.4, alpha=0.9)
    for indice, (x, y) in enumerate(PARADAS_XY, start=1):
        if indice in faltantes and titulo.startswith("hijo"):
            eje.scatter([x], [y], color="black", marker="x", s=70, zorder=3)
        elif indice in repetidas and titulo.startswith("hijo"):
            eje.scatter([x], [y], color="black", s=90, zorder=3)
        else:
            eje.scatter([x], [y], color="grey", s=35, zorder=3)
        eje.annotate(str(indice), (x, y), textcoords="offset points",
                     xytext=(5, 5), fontsize=8)
    eje.set_title(titulo, fontsize=10)
    eje.set_xticks([])
    eje.set_yticks([])
ejes[2].set_xlabel(f"relleno = visitada dos veces {repetidas}   x = nunca visitada {faltantes}",
                   fontsize=8)

FIGURAS.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURAS / "descendencia_06_la_division.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigura guardada en {FIGURAS}/descendencia_06_la_division.png")
# ------------------------------------------------------------------------------

