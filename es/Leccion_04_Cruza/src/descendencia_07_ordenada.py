"""
Lección 04 - Paso 7: Cruza ordenada, y la división leída en ambas direcciones
===============================================================================
NUEVO EN ESTE PASO: cruza_ordenada(), y la comparación para la que se construyó la lección.

La cruza ordenada (OX1) es la respuesta al paso 6: nunca intercambia valores de genes
posición a posición, así que una permutación entra y una permutación sale, por
construcción en lugar de por suerte. Este paso verifica que es legal, verifica lo que
realmente hereda, y luego hace lo que cierra la lección - entrega OX1 los padres de
valores reales del paso 1 y observa qué pasa.

No se bloquea. Ese es el hallazgo.

Nota sobre el listado de Gridin: su order.py indexa un arreglo POR VALOR DE GEN
(`spaces1[t2[i]]`), así que sobre genes de valores reales genera TypeError en cambio. La
implementación de abajo es la formulación usual de OX1 y falla silenciosamente, que es
la falla más instructiva y la más peligrosa.

CAMBIOS RESPECTO A descendencia_06_la_division.py
Introdúcelos en este orden:
    1. cruza_ordenada()          OX1: copia un segmento, rellena el resto en el orden del otro padre
    2. la prueba de adyacencia   lo que OX1 realmente hereda, contra una ruta aleatoria
    3. la otra dirección         OX1 recibe los padres de valores reales del paso 1
    4. la comparación completa   siete operadores, dos representaciones, una tabla
    5. la figura de la división  la columna útil de esa tabla, lado a lado

Ejecútalo: python descendencia_07_ordenada.py

OX1 es legal el 100.00% en las rutas y mantiene 7.17 de 9 tramos.
En genes reales alcanza 2 hijos, 100.00% un padre. Útiles en ambas: 0.
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
        OX1 es legal el 100.00% en las rutas y mantiene 7.17 de 9 tramos.
    """
    return all(GEN_MIN <= gen <= GEN_MAX for gen in cromosoma)


def mostrar(cromosoma) -> str:
    """Un cromosoma en una línea, para que padres e hijos queden alineados en columnas.

    Args:
        cromosoma.

    Returns:
        str.

    Example:
        OX1 es legal el 100.00% en las rutas y mantiene 7.17 de 9 tramos.
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
        OX1 es legal el 100.00% en las rutas y mantiene 7.17 de 9 tramos.
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
        OX1 es legal el 100.00% en las rutas y mantiene 7.17 de 9 tramos.
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


CANTIDAD_PARADAS = 9
PARADAS_XY = [(4.5, 5.6), (9.2, 4.7), (5.1, 5.9), (1.8, 5.1), (6.3, 7.9),
              (0.9, 3.0), (0.9, 8.1), (6.9, 0.4), (9.8, 9.6)]

Recorrido = list[int]


def crear_recorridos() -> tuple[Recorrido, Recorrido]:
    """Dos rutas por las nueve paradas - los padres de la segunda mitad.

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


def es_legal_recorrido(cromosoma) -> bool:
    """Una ruta es legal cuando visita cada una de las nueve paradas exactamente una vez.

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
        OX1 es legal el 100.00% en las rutas y mantiene 7.17 de 9 tramos.
    """
    repetidas = sorted({s for s in cromosoma if list(cromosoma).count(s) > 1})
    faltantes = [s for s in range(1, CANTIDAD_PARADAS + 1) if s not in list(cromosoma)]
    return repetidas, faltantes


# --- NUEVO (1) cruza_ordenada() -----------------------------------------------
def cruza_ordenada(padre1: list, padre2: list) -> tuple[list, list]:
    """Cruza ordenada (OX1): copia un segmento, rellena el resto en el orden del otro.

    Elige dos puntos de corte. El hijo 1 toma el segmento entre ellos del padre 2
    verbatim, y rellena las posiciones restantes - avanzando desde el final del segmento,
    dando la vuelta - con las paradas del padre 1 en el orden en que padre 1 las visita,
    saltando las que el segmento ya usó. El hijo 2 es lo mismo con los roles intercambiados.

    Nada aquí intercambia valores de genes posición a posición, que es por qué el
    resultado es una permutación por construcción en lugar de por suerte. Nota el
    supuesto que esconde: ambos padres deben llevar el MISMO CONJUNTO de valores. El paso 7
    termina rompiendo ese supuesto a propósito.

    Args:
        padre1, padre2.

    Returns:
        tuple[list, list].

    Example:
        Legal 100.00% en rutas; 7.17 de 9 tramos. En reales: 2 hijos, ambos padres.
    """
    longitud = len(padre1)
    inicio, fin = sorted(random.randrange(longitud) for _ in range(2))

    def construir(donante: list, guardián: list) -> list:
        hijo = [None] * longitud
        hijo[inicio:fin + 1] = donante[inicio:fin + 1]
        tomados = set(hijo[inicio:fin + 1])
        ranura = (fin + 1) % longitud
        for paso in range(longitud):
            gen = guardián[(fin + 1 + paso) % longitud]
            if gen not in tomados:
                hijo[ranura] = gen
                ranura = (ranura + 1) % longitud
        return hijo

    return construir(padre2, padre1), construir(padre1, padre2)
# ------------------------------------------------------------------------------


padre1, padre2 = crear_padres()
print(f"padre 1   {mostrar(padre1)}")
print(f"padre 2   {mostrar(padre2)}")


# --- NUEVO (2) la prueba de adyacencia ----------------------------------------
recorrido1, recorrido2 = crear_recorridos()
print(f"\nruta 1   {mostrar(recorrido1)}   longitud {longitud_recorrido(recorrido1):.2f}")
print(f"ruta 2   {mostrar(recorrido2)}   longitud {longitud_recorrido(recorrido2):.2f}")

random.seed(SEMILLA)
print("\nCuatro cruzas ordenadas de esas dos rutas:")
for _ in range(4):
    hijo1, hijo2 = cruza_ordenada(recorrido1, recorrido2)
    print(f"    {mostrar(hijo1)}  legal={es_legal_recorrido(hijo1)}    "
          f"{mostrar(hijo2)}  legal={es_legal_recorrido(hijo2)}")


def aristas(recorrido: Recorrido) -> set:
    """Los pares no ordenados de paradas que una ruta visita de forma consecutiva.

    Esto es lo que una ruta realmente es, desde el punto de vista de su longitud:
    un orden importa solo a través de las adyacencias que crea.

    Args:
        recorrido.

    Returns:
        set.

    Example:
        OX1 es legal el 100.00% en las rutas y mantiene 7.17 de 9 tramos.
    """
    return {frozenset((recorrido[i], recorrido[(i + 1) % len(recorrido)])) for i in range(len(recorrido))}


aristas_padres = aristas(recorrido1) | aristas(recorrido2)
random.seed(SEMILLA)
conservadas, longitudes = [], []
for _ in range(ENSAYOS):
    for hijo in cruza_ordenada(recorrido1, recorrido2):
        conservadas.append(len(aristas(hijo) & aristas_padres))
        longitudes.append(longitud_recorrido(hijo))
random.seed(SEMILLA)
conservadas_aleatorias, longitudes_aleatorias = [], []
for _ in range(ENSAYOS):
    recorrido = random.sample(range(1, CANTIDAD_PARADAS + 1), CANTIDAD_PARADAS)
    conservadas_aleatorias.append(len(aristas(recorrido) & aristas_padres))
    longitudes_aleatorias.append(longitud_recorrido(recorrido))

longitud_padres = (longitud_recorrido(recorrido1) + longitud_recorrido(recorrido2)) / 2
print(f"\nLo que hereda OX1, en {ENSAYOS} extracciones:")
print(f"    tramos padre-a-padre conservados por un hijo OX1:  "
      f"{statistics.mean(conservadas):.2f} de {CANTIDAD_PARADAS}")
print(f"    tramos padre-a-padre conservados por una ruta aleatoria: "
      f"{statistics.mean(conservadas_aleatorias):.2f} de {CANTIDAD_PARADAS}")
print("Un hijo conserva la mayoría de los tramos que condujeron sus padres, que es exactamente lo que")
print("se supone que significa la recombinación y exactamente lo que un hijo de un punto reparado")
print("no haría.")

print(f"\nLa longitud de la ruta, en cambio, casi no dice nada aquí:")
print(f"    los dos padres promedian       {longitud_padres:.2f}")
print(f"    los hijos de OX1 promedian     {statistics.mean(longitudes):.2f}")
print(f"    {ENSAYOS} rutas aleatorias promedian {statistics.mean(longitudes_aleatorias):.2f}")
print("En nueve paradas las rutas buenas y malas están cerca, y estos")
print("padres no son buenos, así que una ruta aleatoria supera a sus hijos en")
print("promedio. Heredar estructura no es lo mismo que heredar calidad -")
print("la cruza aporta lo primero, y solo la selección aporta lo segundo.")
# ------------------------------------------------------------------------------


# --- NUEVO (3) la otra dirección ----------------------------------------------
print("\nY ahora la misma pregunta en la otra dirección. La cruza ordenada recibe")
print("los dos padres de VALORES REALES del paso 1:")
cruzado = reporte_descendencia("ordenada (OX1)", cruza_ordenada,
                               padre1, padre2, es_legal_real)
imprimir_encabezado()
imprimir_fila(cruzado)
random.seed(SEMILLA)
muestra1, muestra2 = cruza_ordenada(padre1, padre2)
print(f"    hijo 1   {mostrar(muestra1)}")
print(f"    hijo 2   {mostrar(muestra2)}")
print(f"\nNo se bloqueó, y {cruzado['legal']:.2f}% de lo que produjo es")
print(f"legal. También alcanzó {cruzado['hijos']} hijos distintos y")
print(f"{cruzado['clon']:.2f}% de ellos son un padre. Esos dos hechos son el")
print("mismo hecho: OX1 rellena las posiciones fuera de su segmento con los genes del otro")
print("padre EN EL ORDEN EN QUE ESE PADRE LOS TIENE, saltando los valores que")
print("el segmento ya usó - y cuando los dos padres no comparten ningún valor")
print("en absoluto, nunca se salta nada y el relleno sobreescribe el segmento. El")
print("operador devuelve los padres.")
print("\nEsta es la peor mitad de la división. Una parada duplicada es ruidosa: el")
print("programa se detiene o la puntuación no tiene sentido y alguien investiga. Un")
print("operador que devuelve silenciosamente sus entradas no rompe nada, no registra nada,")
print("y simplemente hace que la ejecución deje de buscar.")
# ------------------------------------------------------------------------------


# --- NUEVO (4) la comparación completa ----------------------------------------
def proporcion_util(operador: Operador, primero: list, segundo: list,
                    es_legal: Legalidad) -> float:
    """% de hijos que son legales Y no son una copia de un padre.

    Ninguna columna sola capta ambas mitades de la división: los hijos ilegales
    fallan la primera prueba, y un operador que devuelve los padres falla la segunda.
    Esta multiplica las dos, y es la columna de la que se lee la última tabla de la lección.

    Args:
        operador, primero, segundo, es_legal.

    Returns:
        float.

    Example:
        OX1 es legal el 100.00% en las rutas y mantiene 7.17 de 9 tramos.
    """
    random.seed(SEMILLA)
    utiles = producidos = 0
    for _ in range(ENSAYOS):
        for hijo in operador(primero, segundo):
            producidos += 1
            if es_legal(hijo) and list(hijo) != list(primero) \
                    and list(hijo) != list(segundo):
                utiles += 1
    return 100 * utiles / producidos


comunes = [
    ("clon (sin cruza)", cruza_clon),
    ("un punto", cruza_un_punto),
    (f"{N_PUNTOS[-1]} puntos", lambda a, b: cruza_n_puntos(a, b, N_PUNTOS[-1])),
    (f"uniforme (tasa {TASA_UNIFORME})", cruza_uniforme),
    (f"mezcla alfa={ALFAS_MEZCLA[-1]} + acotar",
     lambda a, b: cruza_mezcla(a, b, ALFAS_MEZCLA[-1])),
    (f"lineal alfa={ALFA_LINEAL}", cruza_lineal),
    ("ordenada (OX1)", cruza_ordenada),
]
solo_reales = [("mezcla guiada por aptitud",
                lambda a, b: cruza_guiada_por_aptitud(a, b, ALFAS_MEZCLA[-1]))]

print(f"\nSiete operadores, dos representaciones, {ENSAYOS} extracciones en cada celda.")
print("`útil` es la proporción de hijos que son legales Y no son un padre:")
print(f"\n{'operador':28s} {'VALORES REALES, 6 genes':>28} {'RUTA, 9 paradas':>26}")
print(f"{'':28s} {'legal':>13} {'útil':>14} {'legal':>13} {'útil':>12}")
resumen = []
for etiqueta, operador in comunes:
    real = reporte_descendencia(etiqueta, operador, padre1, padre2, es_legal_real)
    recorrido = reporte_descendencia(etiqueta, operador, recorrido1, recorrido2, es_legal_recorrido)
    real_util = proporcion_util(operador, padre1, padre2, es_legal_real)
    recorrido_util = proporcion_util(operador, recorrido1, recorrido2, es_legal_recorrido)
    resumen.append((etiqueta, real["legal"], real_util, recorrido["legal"], recorrido_util))
    print(f"{etiqueta:28s} {real['legal']:12.2f}% {real_util:13.2f}% "
          f"{recorrido['legal']:12.2f}% {recorrido_util:11.2f}%")
for etiqueta, operador in solo_reales:
    real = reporte_descendencia(etiqueta, operador, padre1, padre2, es_legal_real)
    real_util = proporcion_util(operador, padre1, padre2, es_legal_real)
    print(f"{etiqueta:28s} {real['legal']:12.2f}% {real_util:13.2f}% "
          f"{'n/d':>13} {'n/d':>12}")
print(f"{'':28s} la guiada por aptitud no tiene columna de ruta: tiene que PUNTUAR sus")
print(f"{'':28s} hijos, y una ruta ilegal no tiene longitud.")

ambos = [fila for fila in resumen if fila[2] > 0 and fila[4] > 0]
print(f"\nFilas con una puntuación útil por encima de cero en AMBOS lados: {len(ambos)}.")
print("Esa es la lección, en un entero. Lee las columnas por separado y cada una")
print("parece ordinaria; léelas juntas y no se superponen:")
for etiqueta, _, real_util, _, recorrido_util in resumen:
    if etiqueta.startswith("clon"):
        continue
    lado = "genes de valores reales" if real_util > recorrido_util else "rutas"
    print(f"    {etiqueta:28s} funciona en {lado}, y solo allí")
print("\nLa cruza no es un menú de operadores intercambiables de diferente")
print("fuerza. Son dos familias, elegidas por la representación, y")
print("los fallos que producen se ven completamente diferentes: cromosomas ilegales")
print("en un lado de la línea, no-ops silenciosos en el otro.")
# ------------------------------------------------------------------------------


# --- NUEVO (5) la figura de la división ---------------------------------------
fig, eje = plt.subplots(figsize=(10, 4.6))
etiquetas = [fila[0] for fila in resumen]
posiciones = range(len(etiquetas))
eje.bar([p - 0.2 for p in posiciones], [fila[2] for fila in resumen], width=0.4,
        color="tab:blue", label="genes de valores reales")
eje.bar([p + 0.2 for p in posiciones], [fila[4] for fila in resumen], width=0.4,
        color="tab:orange", label="ruta de nueve paradas")
eje.set_xticks(list(posiciones))
eje.set_xticklabels(etiquetas, rotation=20, ha="right", fontsize=8)
eje.set_ylabel("% de hijos legales y no un padre")
eje.set_ylim(0, 105)
eje.set_title("La división: ningún operador es útil en ambas representaciones")
eje.legend(fontsize=9)
eje.grid(True, axis="y", linestyle=":", alpha=0.5)

FIGURAS.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(FIGURAS / "descendencia_07_ordenada.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nFigura guardada en {FIGURAS}/descendencia_07_ordenada.png")
# ------------------------------------------------------------------------------

