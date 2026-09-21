"""
Lección 04 - Paso 5: Dejar que el operador vea lo que produjo
=============================================================
NUEVO EN ESTE PASO: objetivo(), cruza_guiada_por_aptitud().

Cada operador hasta ahora era ciego: producía dos hijos y los entregaba
sin preguntar si eran buenos. La variante guiada por aptitud de Gridin
agrega una línea - clasifica los dos padres y los dos hijos juntos, devuelve
los mejores dos - y con ella la garantía de que la cruza nunca puede empeorar
el par.

La garantía es real. Este paso mide lo que cuesta, y el costo no es sutil.

(El listado propio de Gridin tiene un bug aquí: construye ambos hijos a partir
de los genes del primer hijo, así que su segundo hijo es un duplicado. Corregido
en el código de abajo.)

CAMBIOS RESPECTO A descendencia_04_mezcla.py
Introdúcelos en este orden:
    1. objetivo()                   un cromosoma finalmente tiene un valor, no solo una forma
    2. cruza_guiada_por_aptitud()   produce dos hijos, conserva los mejores dos de los cuatro
    3. los sobrevivientes           qué compra la regla, y qué cuesta silenciosamente
    4. la figura de aptitud         lo que devuelve cada regla, como distribución

Ejecútalo: python descendencia_05_guiada_por_aptitud.py

La mezcla ciega es peor que sus padres en 41.70% de las extracciones.
La regla guiada es peor en 0.00% y paga: 16.50% devuelve a los padres.
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
        La mezcla ciega es peor que sus padres en 41.70% de las extracciones.
    """
    return all(GEN_MIN <= gen <= GEN_MAX for gen in cromosoma)


def mostrar(cromosoma) -> str:
    """Un cromosoma en una línea, para que padres e hijos queden alineados en columnas.

    Args:
        cromosoma.

    Returns:
        str.

    Example:
        La mezcla ciega es peor que sus padres en 41.70% de las extracciones.
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
        La mezcla ciega es peor que sus padres en 41.70% de las extracciones.
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
        La mezcla ciega es peor que sus padres en 41.70% de las extracciones.
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


# --- NUEVO (1) objetivo() -----------------------------------------------------
def objetivo(cromosoma: Cromosoma) -> float:
    """Lo que vale un cromosoma de valores reales. Mayor es mejor.

    El paisaje de la Lección 01, sin(x) - 0.2|x|, promediado sobre los seis genes,
    para que el número se mantenga en la escala que usó esa lección y un cromosoma
    finalmente tenga un valor en lugar de solo una forma.

    Args:
        cromosoma.

    Returns:
        float.

    Example:
        El máximo global del seno está cerca de x = +1.372, f = +0.706.
    """
    return sum(math.sin(g) - 0.2 * abs(g) for g in cromosoma) / len(cromosoma)
# ------------------------------------------------------------------------------


# --- NUEVO (2) cruza_guiada_por_aptitud() -------------------------------------
def cruza_guiada_por_aptitud(padre1: Cromosoma, padre2: Cromosoma,
                              alfa: float) -> tuple[Cromosoma, Cromosoma]:
    """Mezcla, luego conserva los mejores dos de los cuatro cromosomas involucrados.

    La cruza guiada por aptitud de Gridin. El operador deja de ser ciego: mira
    lo que produjo y se niega a entregar algo peor que lo que recibió. Esa
    garantía es real, y el paso 5 mide lo que cuesta.

    (El listado propio del libro construye ambos hijos a partir de los genes del
    primer hijo, así que su `hijo2` es un duplicado de `hijo1`. Corregido aquí.)

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
# ------------------------------------------------------------------------------


padre1, padre2 = crear_padres()
print(f"padre 1   {mostrar(padre1)}")
print(f"padre 2   {mostrar(padre2)}")


# --- NUEVO (3) los sobrevivientes ---------------------------------------------
print(f"\nLos padres ahora tienen un valor: f(padre 1) = {objetivo(padre1):+.4f}, "
      f"f(padre 2) = {objetivo(padre2):+.4f}")
MEDIA_PADRES = (objetivo(padre1) + objetivo(padre2)) / 2
print(f"Su media es {MEDIA_PADRES:+.4f}. Ese es el número que el operador tiene")
print("que superar.")

alfa = ALFAS_MEZCLA[-1]


def calidad_par(operador: Operador) -> dict:
    """Valor medio del par que devuelve un operador, y con qué frecuencia devuelve
    los padres sin cambios."""
    random.seed(SEMILLA)
    valores, sin_cambio, hijos_conservados, conservados = [], 0, 0, 0
    for _ in range(ENSAYOS):
        primero, segundo = operador(padre1, padre2)
        valores.append((objetivo(primero) + objetivo(segundo)) / 2)
        devueltos = {tuple(primero), tuple(segundo)}
        if devueltos == {tuple(padre1), tuple(padre2)}:
            sin_cambio += 1
        for sobreviviente in (primero, segundo):
            conservados += 1
            if list(sobreviviente) != padre1 and list(sobreviviente) != padre2:
                hijos_conservados += 1
    peor = sum(1 for valor in valores if valor < MEDIA_PADRES)
    return {"media": statistics.mean(valores),
            "sin_cambio": 100 * sin_cambio / ENSAYOS,
            "hijos": 100 * hijos_conservados / conservados,
            "peor": 100 * peor / ENSAYOS}


ciega = calidad_par(lambda a, b: cruza_mezcla(a, b, alfa))
guiada = calidad_par(lambda a, b: cruza_guiada_por_aptitud(a, b, alfa))

print(f"\n{ENSAYOS} extracciones de cada una, con alfa = {alfa}:")
print(f"{'operador':24s} {'media f del':>11} {'extracc. peores':>12} {'par = los':>11} "
      f"{'sobrevivientes':>10}")
print(f"{'':24s} {'par devuelto':>11} {'que los padres':>12} {'padres':>11} {'que son':>10}")
print(f"{'':24s} {'':>11} {'':>12} {'sin cambio':>11} {'hijos':>10}")
print(f"{'los padres mismos':24s} {MEDIA_PADRES:+11.4f} {0.0:11.2f}% "
      f"{100.0:10.2f}% {0.0:9.2f}%")
print(f"{'mezcla, ciega':24s} {ciega['media']:+11.4f} {ciega['peor']:11.2f}% "
      f"{ciega['sin_cambio']:10.2f}% {ciega['hijos']:9.2f}%")
print(f"{'mezcla, guiada por aptitud':24s} {guiada['media']:+11.4f} {guiada['peor']:11.2f}% "
      f"{guiada['sin_cambio']:10.2f}% {guiada['hijos']:9.2f}%")

print(f"\nLa mezcla ciega promedia {ciega['media']:+.4f} aquí, por encima de la")
print(f"{MEDIA_PADRES:+.4f} que recibió - pero eso es este par en este")
print("paisaje, no una propiedad del operador, y la columna siguiente lo dice:")
print(f"en {ciega['peor']:.2f}% de las extracciones el par que devuelve es peor que el")
print("par que recibió. La cruza ciega no ofrece ninguna garantía.")

print(f"\nLa regla guiada por aptitud promedia {guiada['media']:+.4f} y es peor")
print(f"que los padres en {guiada['peor']:.2f}% de las extracciones - no puede serlo,")
print("porque los padres son dos de los cuatro candidatos que clasifica. Esa")
print("garantía es exactamente lo que cuesta:")
print(f"    en {guiada['sin_cambio']:.2f}% de las extracciones devuelve los dos padres")
print("    sin cambios - el operador corrió y la población no se movió")
print(f"    solo {guiada['hijos']:.2f}% de los cromosomas que devuelve son")
print(f"    hijos, contra {ciega['hijos']:.2f}% de la mezcla ciega")
print("\nAsí que la regla convierte exploración en seguridad a una tasa medible. En un")
print("paisaje difícil eso es una ganga; en una ejecución que ya convergió es")
print("una máquina para estar quieto. La Lección 03 midió este intercambio para")
print("la selección; este es el mismo intercambio, movido dentro del operador.")
# ------------------------------------------------------------------------------


# --- NUEVO (4) la figura de aptitud -------------------------------------------
random.seed(SEMILLA)
valores_ciegos, valores_guiados = [], []
for _ in range(ENSAYOS):
    primero, segundo = cruza_mezcla(padre1, padre2, alfa)
    valores_ciegos.extend([objetivo(primero), objetivo(segundo)])
random.seed(SEMILLA)
for _ in range(ENSAYOS):
    primero, segundo = cruza_guiada_por_aptitud(padre1, padre2, alfa)
    valores_guiados.extend([objetivo(primero), objetivo(segundo)])

fig, eje = plt.subplots(figsize=(9, 4))
eje.hist(valores_ciegos, bins=50, alpha=0.6, color="tab:green",
         label="mezcla ciega: lo que salió")
eje.hist(valores_guiados, bins=50, alpha=0.6, color="tab:purple",
         label="guiada por aptitud: los mejores dos de cuatro")
eje.axvline(objetivo(padre1), color="tab:blue", linewidth=2, label="padre 1")
eje.axvline(objetivo(padre2), color="tab:red", linewidth=2, label="padre 2")
eje.set_xlabel("f de un cromosoma devuelto")
eje.set_ylabel("cuenta")
eje.set_title("Lo que devuelve cada regla, en "
              f"{ENSAYOS} extracciones de los mismos dos padres")
eje.legend(fontsize=8)
eje.grid(True, linestyle=":", alpha=0.5)

FIGURAS.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURAS / "descendencia_05_guiada_por_aptitud.png", dpi=150,
            bbox_inches="tight")
plt.close(fig)
print(f"\nFigura guardada en {FIGURAS}/descendencia_05_guiada_por_aptitud.png")
# ------------------------------------------------------------------------------

